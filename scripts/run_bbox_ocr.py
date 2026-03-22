"""
Run OCR on bounding box crops using a HuggingFace VLM.

Reads a predictions CSV with columns [image_path, x, y, width, height, ...],
crops each region from the source image, runs OCR via a VLM, and writes
a parquet file containing all original columns plus `ocr_text` and `ocr_model`.

Usage:
    python scripts/run_bbox_ocr.py \\
        --predictions data/results_spiritualist/spiritualist_heron_predictions.csv \\
        --output data/results_spiritualist/spiritualist_heron_ocr.parquet

    # Override image root (if CSV paths are not locally accessible):
    python scripts/run_bbox_ocr.py \\
        --predictions data/results_spiritualist/spiritualist_heron_predictions.csv \\
        --image-dir /local/path/to/images \\
        --output data/results_spiritualist/spiritualist_heron_ocr.parquet
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

try:
    # transformers >= 5.0
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    try:
        # transformers 4.36–4.x
        from transformers import AutoModelForVision2Seq
    except ImportError:
        try:
            from transformers import Idefics3ForConditionalGeneration as AutoModelForVision2Seq
        except ImportError:
            raise ImportError(
                "Could not import a vision-language model class from transformers. "
                "Please upgrade: pip install --upgrade transformers"
            )

DEFAULT_MODEL = "ibm-granite/granite-docling-258m"
OCR_PROMPT = "Convert the text in this image to plain text. Output only the transcribed text with no commentary."


def load_model(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()
    return processor, model


def build_prompt(processor) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": OCR_PROMPT},
            ],
        }
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def run_ocr_batch(
    crops: list[Image.Image],
    prompt: str,
    processor,
    model,
    device: str,
    max_new_tokens: int,
) -> list[str]:
    inputs = processor(
        text=[prompt] * len(crops),
        images=crops,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the newly generated tokens for each item in the batch
    input_len = inputs["input_ids"].shape[1]
    results = []
    for ids in output_ids:
        text = processor.decode(ids[input_len:], skip_special_tokens=True).strip()
        results.append(text)
    return results


def crop_region(image: Image.Image, x: float, y: float, w: float, h: float) -> Image.Image:
    left = max(0, int(x))
    top = max(0, int(y))
    right = min(image.width, int(x + w))
    bottom = min(image.height, int(y + h))
    return image.crop((left, top, right, bottom))


def resolve_image_path(row_path: str, image_dir: Path | None) -> Path:
    if image_dir is not None:
        return image_dir / Path(row_path).name
    return Path(row_path)


def process_image_group(
    group_df: pd.DataFrame,
    image_dir: Path | None,
    prompt: str,
    processor,
    model,
    device: str,
    max_new_tokens: int,
    batch_size: int,
) -> pd.DataFrame:
    img_path = resolve_image_path(str(group_df.iloc[0]["image_path"]), image_dir)
    image = Image.open(img_path).convert("RGB")

    rows = list(group_df.itertuples(index=False))
    ocr_texts: list[str] = []

    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        crops = [crop_region(image, row.x, row.y, row.width, row.height) for row in batch_rows]
        ocr_texts.extend(run_ocr_batch(crops, prompt, processor, model, device, max_new_tokens))

    result = group_df.copy()
    result["ocr_text"] = ocr_texts
    return result


def merge_parts(parts_dir: Path, output_path: Path) -> None:
    part_files = sorted(parts_dir.glob("*.parquet"))
    if not part_files:
        print("No parts found to merge.")
        return
    df = pd.concat([pd.read_parquet(f) for f in part_files], ignore_index=True)
    df.to_parquet(output_path, index=False)
    print(f"Merged {len(part_files)} parts ({len(df):,} rows) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run VLM OCR on bounding box crops.")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV")
    parser.add_argument("--output", required=True, help="Path to final merged parquet file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Override image directory (uses filename from image_path column)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16, help="Number of crops per inference batch")
    args = parser.parse_args()

    image_dir = Path(args.image_dir) if args.image_dir else None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-image results are saved here; survives interruptions
    parts_dir = output_path.parent / output_path.stem / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    print(f"Loaded {len(df):,} rows from {args.predictions}")

    # Determine which images still need processing
    images = df["filename"].unique()
    remaining = [f for f in images if not (parts_dir / f"{Path(f).stem}.parquet").exists()]
    print(
        f"{len(images)} images total | {len(images) - len(remaining)} already done | "
        f"{len(remaining)} remaining"
    )
    print(f"Device: {args.device}  |  Model: {args.model}  |  Batch size: {args.batch_size}")

    if remaining:
        processor, model = load_model(args.model, args.device)
        prompt = build_prompt(processor)

        for filename in tqdm(remaining, desc="Images"):
            group_df = df[df["filename"] == filename].copy()
            result_df = process_image_group(
                group_df, image_dir, prompt, processor, model,
                args.device, args.max_new_tokens, args.batch_size,
            )
            result_df["ocr_model"] = args.model
            part_path = parts_dir / f"{Path(filename).stem}.parquet"
            result_df.to_parquet(part_path, index=False)

    print(f"\nAll images processed. Merging into {output_path} ...")
    merge_parts(parts_dir, output_path)


if __name__ == "__main__":
    main()

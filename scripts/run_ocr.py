"""
Run traditional OCR models on bounding box crops.

Reads a predictions CSV (produced by cotescore benchmarking) with columns
[filename, image_path, x, y, width, height, ...], crops each region from
the source image, runs OCR via the chosen model, and writes a parquet file
containing all original columns plus `ocr_text` and `ocr_model`.

Supported models: tesseract, trocr, paddleocr, easyocr

Usage:
    python scripts/run_ocr.py \\
        --predictions data/results_spiritualist/spiritualist_yolo_predictions.csv \\
        --model tesseract \\
        --output data/results_spiritualist/spiritualist_yolo_tesseract_ocr.parquet

    # TrOCR on GPU with custom batch size:
    python scripts/run_ocr.py \\
        --predictions data/results_spiritualist/spiritualist_heron_predictions.csv \\
        --model trocr \\
        --device cuda \\
        --batch-size 16 \\
        --output data/results_spiritualist/spiritualist_heron_trocr_ocr.parquet

    # Override image root (if CSV paths are not locally accessible):
    python scripts/run_ocr.py \\
        --predictions data/results_spiritualist/spiritualist_yolo_predictions.csv \\
        --model tesseract \\
        --image-dir /local/path/to/images \\
        --output data/results_spiritualist/spiritualist_yolo_tesseract_ocr.parquet
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# Allow imports from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_models.base import OCRModel


def load_model(model_name: str, device: str, batch_size: int, extra_kwargs: dict) -> OCRModel:
    if model_name == "tesseract":
        from ocr_models.tesseract import TesseractOCR
        return TesseractOCR(**extra_kwargs)
    elif model_name == "trocr":
        from ocr_models.trocr import TrOCROCR
        return TrOCROCR(device=device, batch_size=batch_size, **extra_kwargs)
    elif model_name == "paddleocr":
        from ocr_models.paddleocr import PaddleOCROCR
        return PaddleOCROCR(device=device, **extra_kwargs)
    elif model_name == "easyocr":
        from ocr_models.easyocr import EasyOCROCR
        gpu = device != "cpu"
        return EasyOCROCR(gpu=gpu, **extra_kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name!r}. Choose from: tesseract, trocr, paddleocr, easyocr")


def crop_region(image: Image.Image, x: float, y: float, w: float, h: float) -> Image.Image:
    left = max(0, int(x))
    top = max(0, int(y))
    right = min(image.width, int(x + w))
    bottom = min(image.height, int(y + h))
    if right <= left or bottom <= top:
        return Image.new("RGB", (1, 1), color=255)
    return image.crop((left, top, right, bottom))


def resolve_image_path(row_path: str, image_dir: Path | None) -> Path:
    if image_dir is not None:
        return image_dir / Path(row_path).name
    return Path(row_path)


def merge_parts(parts_dir: Path, output_path: Path) -> None:
    part_files = sorted(parts_dir.glob("*.parquet"))
    if not part_files:
        print("No parts found to merge.")
        return
    df = pd.concat([pd.read_parquet(f) for f in part_files], ignore_index=True)
    df.to_parquet(output_path, index=False)
    print(f"Merged {len(part_files)} parts ({len(df):,} rows) -> {output_path}")


def prepare_crops(group_df: pd.DataFrame, image_dir: Path | None) -> list:
    """Load image and crop all regions. Runs on a background thread for prefetching."""
    img_path = resolve_image_path(str(group_df.iloc[0]["image_path"]), image_dir)
    image = Image.open(img_path).convert("RGB")
    rows = list(group_df.itertuples(index=False))
    return [crop_region(image, row.x, row.y, row.width, row.height) for row in rows]


def process_image_group(
    group_df: pd.DataFrame,
    crops: list,
    model: OCRModel,
    batch_size: int,
) -> pd.DataFrame:
    ocr_texts: list[str] = []

    for batch_start in range(0, len(crops), batch_size):
        batch_crops = crops[batch_start : batch_start + batch_size]
        try:
            ocr_texts.extend(model.run_batch(batch_crops))
        except Exception as e:
            print(f"  Warning: batch OCR failed ({e}), falling back to per-crop")
            for crop in batch_crops:
                try:
                    ocr_texts.append(model.run(crop))
                except Exception:
                    ocr_texts.append("")

    result = group_df.copy()
    result["ocr_text"] = ocr_texts
    return result


def main():
    parser = argparse.ArgumentParser(description="Run OCR on bounding box crops.")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV")
    parser.add_argument(
        "--model",
        required=True,
        choices=["tesseract", "trocr", "paddleocr", "easyocr"],
        help="OCR model to use",
    )
    parser.add_argument("--output", required=True, help="Path to final merged parquet file")
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Override image directory (uses filename from image_path column)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for GPU-capable models: cpu, cuda, gpu (default: cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of crops per inference batch (default: 8)",
    )
    # Tesseract-specific
    parser.add_argument("--lang", default=None, help="Tesseract/EasyOCR language (default: eng/en)")
    parser.add_argument("--psm", type=int, default=None, help="Tesseract PSM mode (default: 6)")
    # TrOCR-specific
    parser.add_argument("--model-name", default=None, help="HuggingFace model ID for TrOCR")
    parser.add_argument(
        "--no-split-lines",
        action="store_true",
        help="Disable TrOCR automatic line splitting (enabled by default)",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir) if args.image_dir else None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts_dir = output_path.parent / output_path.stem / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    # Build model-specific kwargs from CLI flags
    extra_kwargs: dict = {}
    if args.lang is not None:
        extra_kwargs["lang"] = args.lang
    if args.psm is not None:
        extra_kwargs["psm"] = args.psm
    if args.model_name is not None:
        extra_kwargs["model_name"] = args.model_name
    if args.model == "trocr":
        extra_kwargs["split_lines"] = not args.no_split_lines

    model = load_model(args.model, args.device, args.batch_size, extra_kwargs)
    print(f"Loading {args.model} model...")
    model.load()

    df = pd.read_csv(args.predictions)
    print(f"Loaded {len(df):,} rows from {args.predictions}")

    images = df["filename"].unique()
    remaining = [f for f in images if not (parts_dir / f"{Path(f).stem}.parquet").exists()]
    print(
        f"{len(images)} images total | {len(images) - len(remaining)} already done | "
        f"{len(remaining)} remaining"
    )
    print(f"Device: {args.device}  |  Model: {args.model}  |  Batch size: {args.batch_size}")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(prepare_crops, df[df["filename"] == remaining[0]].copy(), image_dir) if remaining else None

        for i, filename in enumerate(tqdm(remaining, desc="Images")):
            group_df = df[df["filename"] == filename].copy()

            try:
                crops = future.result()
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
                if i + 1 < len(remaining):
                    future = executor.submit(prepare_crops, df[df["filename"] == remaining[i + 1]].copy(), image_dir)
                continue

            # Prefetch next image while GPU runs inference on current
            if i + 1 < len(remaining):
                future = executor.submit(prepare_crops, df[df["filename"] == remaining[i + 1]].copy(), image_dir)

            try:
                result_df = process_image_group(group_df, crops, model, args.batch_size)
                result_df["ocr_model"] = args.model
                part_path = parts_dir / f"{Path(filename).stem}.parquet"
                result_df.to_parquet(part_path, index=False)
            except Exception as e:
                print(f"  Error processing {filename}: {e}")

    print(f"\nAll images processed. Merging into {output_path} ...")
    merge_parts(parts_dir, output_path)


if __name__ == "__main__":
    main()

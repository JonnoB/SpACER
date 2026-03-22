"""
Run OCR on bounding box crops using the Docling VLM pipeline.

Reads a predictions CSV with columns [filename, image_path, x, y, width, height, ...],
crops each region from the source image, converts each crop via Docling's DocumentConverter
(which handles prompt formatting and output parsing internally), and writes a parquet file
containing all original columns plus `ocr_text` and `ocr_model`.

Per-image parquet parts are saved incrementally so interrupted runs can resume cleanly.

Usage:
    python scripts/run_docling_ocr.py \\
        --predictions data/results_spiritualist/spiritualist_heron_predictions.csv \\
        --output data/results_spiritualist/spiritualist_heron_docling_ocr.parquet

    # Override image directory (if paths in CSV are not locally accessible):
    python scripts/run_docling_ocr.py \\
        --predictions data/results_spiritualist/spiritualist_heron_predictions.csv \\
        --image-dir /teamspace/lightning_storage/the_spiritualist/spiritualist_images \\
        --output data/results_spiritualist/spiritualist_heron_docling_ocr.parquet

    # Use a different Docling preset:
    python scripts/run_docling_ocr.py \\
        --predictions data/results_spiritualist/spiritualist_heron_predictions.csv \\
        --output data/results_spiritualist/spiritualist_heron_docling_ocr.parquet \\
        --model smoldocling
"""

import argparse
import signal
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

DEFAULT_MODEL = "granite_docling"
CROP_TIMEOUT_SECS = 120


def _timeout_handler(signum, frame):
    raise TimeoutError("crop OCR timed out")


def build_converter(preset: str) -> DocumentConverter:
    vlm_options = VlmConvertOptions.from_preset(preset)
    return DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=VlmPipelineOptions(vlm_options=vlm_options),
            )
        }
    )


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


def merge_parts(parts_dir: Path, output_path: Path) -> None:
    part_files = sorted(parts_dir.glob("*.parquet"))
    if not part_files:
        print("No parts found to merge.")
        return
    df = pd.concat([pd.read_parquet(f) for f in part_files], ignore_index=True)
    df.to_parquet(output_path, index=False)
    print(f"Merged {len(part_files)} parts ({len(df):,} rows) -> {output_path}")


def process_image_group(
    group_df: pd.DataFrame,
    image_dir: Path | None,
    converter: DocumentConverter,
) -> pd.DataFrame:
    img_path = resolve_image_path(str(group_df.iloc[0]["image_path"]), image_dir)
    print(f"Processing {img_path.name} ({len(group_df)} crops)")
    if not img_path.exists():
        print(f"  Warning: image not found, skipping: {img_path}")
        out = group_df.copy()
        out["ocr_text"] = ""
        return out
    image = Image.open(img_path).convert("RGB")

    signal.signal(signal.SIGALRM, _timeout_handler)

    texts = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, row in enumerate(tqdm(group_df.itertuples(index=False), total=len(group_df), desc="  crops")):
            crop = crop_region(image, row.x, row.y, row.width, row.height)
            p = Path(tmp) / f"crop_{i:04d}.png"
            crop.save(p)
            signal.alarm(CROP_TIMEOUT_SECS)
            try:
                result = converter.convert(p)
                texts.append(result.document.export_to_markdown().strip())
            except TimeoutError:
                print(f"  Warning: crop {i} timed out after {CROP_TIMEOUT_SECS}s, skipping")
                texts.append("")
            except Exception as e:
                print(f"  Warning: crop {i} failed ({e}), skipping")
                texts.append("")
            finally:
                signal.alarm(0)

    out = group_df.copy()
    out["ocr_text"] = texts
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run Docling VLM OCR on bounding box crops."
    )
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV")
    parser.add_argument("--output", required=True, help="Path to final merged parquet file")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Docling VLM preset name (e.g. granite_docling, smoldocling)",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Override image directory (uses filename from image_path column)",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir) if args.image_dir else None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts_dir = output_path.parent / output_path.stem / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    print(f"Loaded {len(df):,} rows from {args.predictions}")

    images = df["filename"].unique()
    remaining = [f for f in images if not (parts_dir / f"{Path(f).stem}.parquet").exists()]
    print(
        f"{len(images)} images total | {len(images) - len(remaining)} already done | "
        f"{len(remaining)} remaining"
    )

    if remaining:
        print(f"Model: {args.model}")
        converter = build_converter(args.model)

        for filename in remaining:
            group_df = df[df["filename"] == filename].copy()
            result_df = process_image_group(group_df, image_dir, converter)
            result_df["ocr_model"] = args.model
            part_path = parts_dir / f"{Path(filename).stem}.parquet"
            result_df.to_parquet(part_path, index=False)

    print(f"\nMerging into {output_path} ...")
    merge_parts(parts_dir, output_path)


if __name__ == "__main__":
    main()

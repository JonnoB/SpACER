"""
Run Docling VLM OCR on whole page images (the intended usage).

Reads a predictions CSV to determine which images to process, then converts each
full page image via Docling's DocumentConverter. Output is one row per page with
the full-page markdown text, for comparison against crop-based OCR results.

Per-image parquet parts are saved incrementally so interrupted runs can resume cleanly.

Usage:
    python scripts/run_docling_wholepage_ocr.py \\
        --predictions data/results_spiritualist/spiritualist_heron_predictions.csv \\
        --output data/results_spiritualist/spiritualist_wholepage_docling_ocr.parquet

    # Override image directory (if paths in CSV are not locally accessible):
    python scripts/run_docling_wholepage_ocr.py \\
        --predictions data/results_spiritualist/spiritualist_heron_predictions.csv \\
        --image-dir /teamspace/lightning_storage/the_spiritualist/spiritualist_images \\
        --output data/results_spiritualist/spiritualist_wholepage_docling_ocr.parquet

    # Use a different Docling preset:
    python scripts/run_docling_wholepage_ocr.py \\
        --predictions data/results_spiritualist/spiritualist_heron_predictions.csv \\
        --output data/results_spiritualist/spiritualist_wholepage_docling_ocr.parquet \\
        --model smoldocling
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

DEFAULT_MODEL = "granite_docling"


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


def main():
    parser = argparse.ArgumentParser(
        description="Run Docling VLM OCR on whole page images."
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

    # One row per unique image — take the first occurrence for metadata
    image_meta = (
        df.drop_duplicates(subset="filename")[["filename", "image_path", "image_width", "image_height"]]
        .reset_index(drop=True)
    )
    print(f"Loaded {len(image_meta)} unique images from {args.predictions}")

    remaining = image_meta[
        ~image_meta["filename"].apply(
            lambda f: (parts_dir / f"{Path(f).stem}.parquet").exists()
        )
    ]
    print(
        f"{len(image_meta)} images total | {len(image_meta) - len(remaining)} already done | "
        f"{len(remaining)} remaining"
    )

    if not remaining.empty:
        print(f"Model: {args.model}")
        converter = build_converter(args.model)

        for row in tqdm(remaining.itertuples(index=False), total=len(remaining), desc="Images"):
            img_path = resolve_image_path(str(row.image_path), image_dir)
            if not img_path.exists():
                print(f"  Warning: image not found, skipping: {img_path}")
                continue
            result = converter.convert(str(img_path))
            ocr_text = result.document.export_to_markdown().strip()

            part_df = pd.DataFrame([{
                "filename": row.filename,
                "image_path": row.image_path,
                "image_width": row.image_width,
                "image_height": row.image_height,
                "ocr_text": ocr_text,
                "ocr_model": args.model,
            }])
            part_path = parts_dir / f"{Path(row.filename).stem}.parquet"
            part_df.to_parquet(part_path, index=False)

    print(f"\nMerging into {output_path} ...")
    merge_parts(parts_dir, output_path)


if __name__ == "__main__":
    main()

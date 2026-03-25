"""
Run all OCR models over all bbox prediction CSVs in a directory.

Loops over every combination of (parsing model CSV) x (OCR model), loading
each OCR model once and running it across all CSVs before moving to the next.
Output parquets follow the naming: {csv_stem}_{ocr_model}_ocr.parquet

Usage:
    python scripts/run_all_ocr.py \\
        --bboxes-dir data/results_spiritualist/bboxes \\
        --output-dir data/results_spiritualist/ocr

    # Specific OCR models only:
    python scripts/run_all_ocr.py \\
        --bboxes-dir data/results_spiritualist/bboxes \\
        --output-dir data/results_spiritualist/ocr \\
        --ocr-models tesseract trocr

    # With image path override and GPU:
    python scripts/run_all_ocr.py \\
        --bboxes-dir data/results_spiritualist/bboxes \\
        --output-dir data/results_spiritualist/ocr \\
        --image-dir /path/to/images \\
        --device cuda
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_ocr import load_model, prepare_crops_and_splits, process_image_group, merge_parts

ALL_MODELS = ["tesseract", "trocr", "paddleocr", "easyocr"]


def main():
    parser = argparse.ArgumentParser(
        description="Run all OCR models over all bbox prediction CSVs."
    )
    parser.add_argument(
        "--bboxes-dir",
        required=True,
        help="Directory containing prediction CSVs",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write output parquet files",
    )
    parser.add_argument(
        "--ocr-models",
        nargs="+",
        required=True,
        choices=ALL_MODELS,
        help="OCR models to run. Note: paddleocr is incompatible with pytorch-based models "
             "(trocr, easyocr) at the CUDA level — run them in separate invocations.",
    )
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
    args = parser.parse_args()

    bboxes_dir = Path(args.bboxes_dir)
    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None

    csv_files = sorted(bboxes_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {bboxes_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV(s): {[f.name for f in csv_files]}")
    print(f"OCR models: {args.ocr_models}")
    print(f"Device: {args.device}  |  Batch size: {args.batch_size}\n")

    for ocr_model_name in args.ocr_models:
        print(f"=== Loading OCR model: {ocr_model_name} ===")
        model = load_model(ocr_model_name, args.device, args.batch_size, {})
        model.load()

        for csv_path in csv_files:
            output_path = output_dir / f"{csv_path.stem}_{ocr_model_name}_ocr.parquet"

            if output_path.exists():
                print(f"  Skipping {csv_path.name} -> {output_path.name} (already exists)")
                continue

            print(f"  Processing {csv_path.name} -> {output_path.name}")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            parts_dir = output_path.parent / output_path.stem / "parts"
            parts_dir.mkdir(parents=True, exist_ok=True)

            df = pd.read_csv(csv_path)
            if "source" in df.columns:
                df = df[df["source"] == "pred"].copy()
            print(f"    {len(df):,} prediction rows")

            images = df["filename"].unique()
            remaining = [
                f for f in images
                if not (parts_dir / f"{Path(f).stem}.parquet").exists()
            ]
            print(f"    {len(images)} images | {len(images) - len(remaining)} done | {len(remaining)} remaining")

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(prepare_crops_and_splits, df[df["filename"] == remaining[0]].copy(), image_dir, model) if remaining else None

                for i, filename in enumerate(tqdm(remaining, desc=f"    {ocr_model_name}/{csv_path.stem}")):
                    try:
                        group_df, prepared = future.result()
                    except Exception as e:
                        print(f"    Error loading {filename}: {e}")
                        if i + 1 < len(remaining):
                            future = executor.submit(prepare_crops_and_splits, df[df["filename"] == remaining[i + 1]].copy(), image_dir, model)
                        continue

                    # Prefetch next image (+ splits) while GPU runs inference on current
                    if i + 1 < len(remaining):
                        future = executor.submit(prepare_crops_and_splits, df[df["filename"] == remaining[i + 1]].copy(), image_dir, model)

                    try:
                        result_df = process_image_group(group_df, prepared, model)
                        result_df["ocr_model"] = ocr_model_name
                        part_path = parts_dir / f"{Path(filename).stem}.parquet"
                        result_df.to_parquet(part_path, index=False)
                    except Exception as e:
                        print(f"    Error processing {filename}: {e}")

            merge_parts(parts_dir, output_path)

        print()


if __name__ == "__main__":
    main()

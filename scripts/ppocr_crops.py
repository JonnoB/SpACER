"""
Run PaddleOCR over a flat folder of pre-cropped region images.

Designed for crop folders produced by mask/RLE extraction, where filenames
encode the source page and region: {page_id}_{region_idx:03d}_{label}.{ext}
(region_idx is always 3 digits). No source page image or bbox CSV is needed —
each file is OCR'd as-is.

PaddleOCR-only and intentionally un-abstracted: all the det/rec model choice,
engine backend, and pipeline toggles are exposed directly as CLI flags so you
can pick any PaddleOCR model combo (PP-OCRv5_server, PP-OCRv6_medium, ...)
without going through a generic OCRModel interface. If/when other backends
need to share this input/output contract, that's the point to extract a
small library — not before.

Output is a single parquet, one row per crop:
    [crop_file, page_id, region_idx, label, ocr_text, ocr_model]

Usage:
    # PaddleOCR's installed default (whatever that resolves to)
    python scripts/run_ocr_crops.py \\
        --crops-dir data/nls_directories_crops \\
        --output data/nls_directories_paddleocr_ocr.parquet \\
        --device gpu

    # Explicit PP-OCRv6_medium via the transformers engine
    python scripts/run_ocr_crops.py \\
        --crops-dir data/nls_directories_crops \\
        --output data/nls_directories_paddleocr_v6medium_ocr.parquet \\
        --device gpu \\
        --engine transformers \\
        --det-model-name PP-OCRv6_medium_det \\
        --rec-model-name PP-OCRv6_medium_rec
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

CROP_NAME_RE = re.compile(r"^(?P<page_id>.+)_(?P<region_idx>\d{3})_(?P<label>.+)$")


def parse_crop_filename(path: Path) -> tuple[str, int, str]:
    """Split '{page_id}_{region_idx:03d}_{label}.ext' into its parts.

    Raises ValueError if the filename doesn't match the expected convention.
    """
    m = CROP_NAME_RE.match(path.stem)
    if not m:
        raise ValueError(f"Filename does not match '{{page_id}}_{{idx}}_{{label}}': {path.name}")
    return m["page_id"], int(m["region_idx"]), m["label"]


def build_engine(args):
    from paddleocr import PaddleOCR

    kwargs = {
        "lang": args.lang,
        "device": args.device,
        "use_doc_orientation_classify": args.use_doc_orientation_classify,
        "use_doc_unwarping": args.use_doc_unwarping,
        "use_textline_orientation": args.use_textline_orientation,
    }
    if args.engine is not None:
        kwargs["engine"] = args.engine
    if args.det_model_name is not None:
        kwargs["text_detection_model_name"] = args.det_model_name
    if args.rec_model_name is not None:
        kwargs["text_recognition_model_name"] = args.rec_model_name
    return PaddleOCR(**kwargs)


def ocr_paths(engine, paths: list[str], batch_size: int) -> list[str]:
    """Run the pipeline over a list of crop file paths, one result per path."""
    results = list(engine.predict(paths, batch_size=batch_size))
    if len(results) != len(paths):
        # Fall back to one-by-one if the engine didn't return a 1:1 mapping
        results = []
        for p in paths:
            res = list(engine.predict(p, batch_size=1))
            results.append(res[0] if res else {})
    texts = []
    for res in results:
        rec_texts = (res or {}).get("rec_texts") or []
        texts.append(" ".join(t for t in rec_texts if t).strip())
    return texts


def merge_parts(parts_dir: Path, output_path: Path) -> None:
    part_files = sorted(parts_dir.glob("*.parquet"))
    if not part_files:
        print("No parts found to merge.")
        return
    df = pd.concat([pd.read_parquet(f) for f in part_files], ignore_index=True)
    df.to_parquet(output_path, index=False)
    print(f"Merged {len(part_files)} parts ({len(df):,} rows) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run PaddleOCR on a folder of pre-cropped region images.")
    parser.add_argument("--crops-dir", required=True, help="Folder of crop image files")
    parser.add_argument("--output", required=True, help="Path to final merged parquet file")
    parser.add_argument("--device", default="cpu", help="e.g. cpu, gpu, gpu:0 (default: cpu)")
    parser.add_argument("--lang", default="en", help="PaddleOCR language (default: en)")
    parser.add_argument(
        "--engine", default=None, choices=["paddle", "transformers"],
        help="Inference backend. Default: PaddleOCR's own default (paddle).",
    )
    parser.add_argument(
        "--det-model-name", default=None,
        help="e.g. PP-OCRv5_server_det, PP-OCRv6_medium_det. Default: PaddleOCR's installed default.",
    )
    parser.add_argument(
        "--rec-model-name", default=None,
        help="e.g. PP-OCRv5_server_rec, PP-OCRv6_medium_rec. Default: PaddleOCR's installed default.",
    )
    parser.add_argument("--use-textline-orientation", action="store_true")
    parser.add_argument("--use-doc-orientation-classify", action="store_true")
    parser.add_argument("--use-doc-unwarping", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8, help="Crops per predict() call (default: 8)")
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts_dir = output_path.parent / output_path.stem / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    crop_paths = sorted(p for p in crops_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not crop_paths:
        print(f"No crop images found in {crops_dir}")
        sys.exit(1)

    pages: dict[str, list[tuple[int, str, Path]]] = defaultdict(list)
    for p in crop_paths:
        try:
            page_id, region_idx, label = parse_crop_filename(p)
        except ValueError as e:
            print(f"  Skipping unparsable filename: {e}")
            continue
        pages[page_id].append((region_idx, label, p))
    for page_id in pages:
        pages[page_id].sort(key=lambda t: t[0])

    page_ids = sorted(pages)
    remaining = [pid for pid in page_ids if not (parts_dir / f"{pid}.parquet").exists()]
    print(f"{len(crop_paths):,} crop files | {len(page_ids)} pages | "
          f"{len(page_ids) - len(remaining)} pages already done | {len(remaining)} remaining")

    model_label = "+".join(filter(None, [args.det_model_name, args.rec_model_name])) or "paddleocr_default"
    print(f"Device: {args.device}  |  Engine: {args.engine or 'default'}  |  Model: {model_label}  |  "
          f"Batch size: {args.batch_size}")

    engine = build_engine(args)

    for page_id in tqdm(remaining, desc="Pages"):
        rows = pages[page_id]
        paths = [str(p) for _, _, p in rows]
        try:
            ocr_texts = ocr_paths(engine, paths, args.batch_size)
        except Exception as e:
            print(f"  Error processing {page_id}: {e}")
            continue

        result_df = pd.DataFrame({
            "crop_file": [p.name for _, _, p in rows],
            "page_id": page_id,
            "region_idx": [idx for idx, _, _ in rows],
            "label": [label for _, label, _ in rows],
            "ocr_text": ocr_texts,
        })
        result_df["ocr_model"] = model_label
        result_df.to_parquet(parts_dir / f"{page_id}.parquet", index=False)

    print(f"\nAll pages processed. Merging into {output_path} ...")
    merge_parts(parts_dir, output_path)


if __name__ == "__main__":
    main()

"""
Extract ground-truth text and bounding boxes per SSU (paragraph) region from the
HierText validation annotations, in the same schema as extract_ssu_text.py
produces for the spiritualist dataset.

Reads:
    data/heiertext_validation.jsonl
        HierText GT: {"info": ..., "annotations": [{"image_id", "paragraphs": [
            {"vertices", "legible", "lines": [{"vertices", "text", ...}]}
        ]}]}
    data/hiertext_predictions/hiertext_yolo_predictions.csv
        Used only to look up image_width/image_height/image_path per filename
        (avoids needing local image files to build the GT text CSV). Any of the
        hiertext_*_predictions.csv files would do — they share identical
        filename/image_width/image_height/image_path columns.

Writes:
    data/hiertext/gt_ssu_bboxes.csv
        columns: filename, page_id, image_width, image_height,
                 x, y, width, height, polygon_points, ssu_id, gt_text
    data/hiertext_predictions/hiertext_gt_predictions.csv
        Same rows without gt_text (image_path = filename), dropped alongside
        the other hiertext_*_predictions.csv files so run_all_ocr.py / run_ocr.py
        can crop and OCR them exactly like the spiritualist GT boxes. This file
        has no "source" column, so run_all_ocr.py's `source == "pred"` filter
        (which only applies to CSVs that carry that column) does not touch it.
    data/hiertext/gt_page_texts/<page_id>.txt

SSU/paragraph geometry mirrors cotescore's HierTextDataset._build_annotations
exactly: paragraphs are 1-indexed in document order to form ssu_id, and lines
with fewer than 3 vertices or a degenerate axis-aligned bbox are dropped. This
keeps ssu_id numbering identical to the "source"=="gt" rows already embedded in
data/hiertext_predictions/*.csv, so both can be joined/cross-checked directly.
Text is preserved as-is (each line's `text` field, joined with spaces).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.crop_utils import staircase_polygon

GT_JSON      = Path("data/heiertext_validation.jsonl")
IMAGE_META   = Path("data/hiertext_predictions/hiertext_yolo_predictions.csv")
OUTPUT       = Path("data/hiertext/gt_ssu_bboxes.csv")
BBOXES_OUT   = Path("data/hiertext_predictions/hiertext_gt_predictions.csv")
PAGE_TEXTS   = Path("data/hiertext/gt_page_texts")


def line_bbox(vertices: list) -> tuple | None:
    if len(vertices) < 3:
        return None
    xs = [pt[0] for pt in vertices]
    ys = [pt[1] for pt in vertices]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def extract_page(entry: dict, image_meta: dict) -> list[dict]:
    image_id = entry["image_id"]
    filename = f"{image_id}.jpg"
    if filename not in image_meta:
        return []
    image_width, image_height = image_meta[filename]

    records = []
    for para_idx, paragraph in enumerate(entry.get("paragraphs", []), start=1):
        line_tuples = []
        texts = []
        for line in paragraph.get("lines", []):
            bbox = line_bbox(line.get("vertices", []))
            if bbox is None:
                continue
            line_tuples.append(bbox)
            texts.append(line.get("text", ""))

        if not line_tuples:
            continue

        x1 = min(t[0] for t in line_tuples)
        y1 = min(t[1] for t in line_tuples)
        x2 = max(t[0] + t[2] for t in line_tuples)
        y2 = max(t[1] + t[3] for t in line_tuples)

        records.append({
            "filename":       filename,
            "page_id":        image_id,
            "image_width":    image_width,
            "image_height":   image_height,
            "x":              x1,
            "y":              y1,
            "width":          x2 - x1,
            "height":         y2 - y1,
            "polygon_points": staircase_polygon(line_tuples),
            "ssu_id":         para_idx,
            "gt_text":        " ".join(texts),
        })
    return records


def main() -> None:
    import json

    print(f"Loading {GT_JSON} …")
    gt = json.loads(GT_JSON.read_text())
    entries = gt["annotations"]
    print(f"  {len(entries):,} annotated images")

    meta_df = pd.read_csv(IMAGE_META, usecols=["filename", "image_width", "image_height"]).drop_duplicates("filename")
    image_meta = {
        row.filename: (row.image_width, row.image_height)
        for row in meta_df.itertuples(index=False)
    }
    print(f"  {len(image_meta):,} images with known dimensions from {IMAGE_META}")

    PAGE_TEXTS.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    skipped = 0
    for entry in entries:
        filename = f"{entry['image_id']}.jpg"
        if filename not in image_meta:
            skipped += 1
            continue
        records = extract_page(entry, image_meta)
        all_records.extend(records)

        page_text = "\n\n".join(r["gt_text"] for r in records)
        (PAGE_TEXTS / f"{entry['image_id']}.txt").write_text(page_text, encoding="utf-8")

    if skipped:
        print(f"  Skipped {skipped:,} images not present in {IMAGE_META}")

    df = pd.DataFrame(all_records, columns=[
        "filename", "page_id", "image_width", "image_height",
        "x", "y", "width", "height", "polygon_points", "ssu_id", "gt_text",
    ])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df):,} SSU (paragraph) regions across {df['page_id'].nunique():,} pages → {OUTPUT}")

    # Write the OCR pipeline bboxes file (no gt_text; image_path = filename),
    # dropped alongside hiertext_{model}_predictions.csv for run_all_ocr.py.
    bboxes_df = df[["filename", "image_width", "image_height", "x", "y", "width", "height", "polygon_points", "ssu_id"]].copy()
    bboxes_df.insert(1, "image_path", bboxes_df["filename"])
    BBOXES_OUT.parent.mkdir(parents=True, exist_ok=True)
    bboxes_df.to_csv(BBOXES_OUT, index=False)
    print(f"Saved {len(bboxes_df):,} rows → {BBOXES_OUT}")
    print(f"Page text files → {PAGE_TEXTS}/")


if __name__ == "__main__":
    main()

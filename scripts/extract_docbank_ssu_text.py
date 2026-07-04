"""
Extract ground-truth text and bounding boxes per SSU (structural semantic
unit) region from DocBank's raw per-word annotation files, in the same
schema extract_hiertext_ssu_text.py / extract_ssu_text.py produce for
HierText / spiritualist.

DocBank has no explicit region grouping in its raw annotation — SSU regions
are derived here as maximal runs of consecutive words sharing the same
semantic label (paragraph/title/caption/...), in document order. This
mirrors how DocBank's own MSCOCO conversion merges token spans into region
boxes, but is derived directly from the word-level gt so it doesn't depend
on data/docbank/mscoco_annotations_subset.json for anything but page pixel
dimensions.

Coordinate system: DocBank's raw bbox coordinates are normalized to a
0-1000 scale per page, NOT actual pixel coordinates (verified against
data/docbank/mscoco_annotations_subset.json: raw x values exceed a page's
actual pixel width, while raw y values happen to fit only because every
DocBank page image is rendered at a fixed height of 1000px). Converted here
via px = raw * page_dim_px / 1000 for both axes — see
infer_characters_docbank.py, which relies on the same conversion.

Reads:
    data/docbank/gt_word_annotations/<basename>.txt
        Raw DocBank format: word\\tx0\\ty0\\tx1\\ty1\\tR\\tG\\tB\\tfont\\tlabel
    data/docbank/mscoco_annotations_subset.json
        Used only to look up image_width/image_height per basename.

Writes:
    data/docbank/gt_ssu_bboxes.csv
        columns: filename, page_id, image_width, image_height, x, y, width,
                 height, ssu_id, label, gt_text
    data/docbank/docbank_gt_predictions.csv
        Same rows without gt_text/label (image_path = filename), dropped
        alongside any data/docbank_predictions/docbank_{model}_predictions.csv
        so run_all_ocr.py can crop and OCR them exactly like hiertext's GT boxes.
    data/docbank/gt_page_texts/<basename>.txt
"""

import json
from pathlib import Path

import pandas as pd

GT_DIR      = Path("data/docbank/gt_word_annotations")
MSCOCO_PATH = Path("data/docbank/mscoco_annotations_subset.json")
OUTPUT      = Path("data/docbank/gt_ssu_bboxes.csv")
BBOXES_OUT  = Path("data/docbank/docbank_gt_predictions.csv")
PAGE_TEXTS  = Path("data/docbank/gt_page_texts")


def parse_word_line(line: str) -> dict | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 10:
        return None
    word = parts[0]
    x0, y0, x1, y1 = (int(v) for v in parts[1:5])
    label = parts[9]
    return {"word": word, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "label": label}


def group_into_ssu_runs(words: list[dict]) -> list[list[dict]]:
    """Group words into maximal runs of consecutive identical labels (document order)."""
    runs: list[list[dict]] = []
    for w in words:
        if runs and runs[-1][-1]["label"] == w["label"]:
            runs[-1].append(w)
        else:
            runs.append([w])
    return runs


def to_pixel_bbox(run: list[dict], width_px: int, height_px: int) -> tuple[float, float, float, float]:
    x0 = min(w["x0"] for w in run) * width_px / 1000
    y0 = min(w["y0"] for w in run) * height_px / 1000
    x1 = max(w["x1"] for w in run) * width_px / 1000
    y1 = max(w["y1"] for w in run) * height_px / 1000
    return x0, y0, x1 - x0, y1 - y0


def main() -> None:
    print(f"Loading {MSCOCO_PATH} ...")
    coco = json.loads(MSCOCO_PATH.read_text())
    image_dims = {img["file_name"]: (img["width"], img["height"]) for img in coco["images"]}

    txt_files = sorted(GT_DIR.glob("*.txt"))
    print(f"  {len(txt_files):,} gt word-annotation files in {GT_DIR}")

    PAGE_TEXTS.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    skipped = 0
    for txt_path in txt_files:
        basename = txt_path.stem
        filename = f"{basename}_ori.jpg"
        if filename not in image_dims:
            skipped += 1
            continue
        width_px, height_px = image_dims[filename]

        words = [
            w for line in txt_path.read_text(encoding="utf8").splitlines()
            if (w := parse_word_line(line)) is not None
        ]
        runs = group_into_ssu_runs(words)

        records = []
        for ssu_id, run in enumerate(runs, start=1):
            x, y, w_, h_ = to_pixel_bbox(run, width_px, height_px)
            records.append({
                "filename": filename,
                "page_id": basename,
                "image_width": width_px,
                "image_height": height_px,
                "x": x,
                "y": y,
                "width": w_,
                "height": h_,
                "ssu_id": ssu_id,
                "label": run[0]["label"],
                "gt_text": " ".join(w["word"] for w in run),
            })
        all_records.extend(records)

        page_text = "\n\n".join(r["gt_text"] for r in records)
        (PAGE_TEXTS / f"{basename}.txt").write_text(page_text, encoding="utf-8")

    if skipped:
        print(f"  Skipped {skipped:,} pages not present in {MSCOCO_PATH}")

    df = pd.DataFrame(all_records, columns=[
        "filename", "page_id", "image_width", "image_height",
        "x", "y", "width", "height", "ssu_id", "label", "gt_text",
    ])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df):,} SSU regions across {df['page_id'].nunique():,} pages -> {OUTPUT}")

    bboxes_df = df[["filename", "image_width", "image_height", "x", "y", "width", "height", "ssu_id"]].copy()
    bboxes_df.insert(1, "image_path", bboxes_df["filename"])
    BBOXES_OUT.parent.mkdir(parents=True, exist_ok=True)
    bboxes_df.to_csv(BBOXES_OUT, index=False)
    print(f"Saved {len(bboxes_df):,} rows -> {BBOXES_OUT}")
    print(f"Page text files -> {PAGE_TEXTS}/")


if __name__ == "__main__":
    main()

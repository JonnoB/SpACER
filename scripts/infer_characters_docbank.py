"""
Infer character-level bounding boxes from DocBank word-level ground truth.

For each word, character positions are estimated by uniformly distributing
the word's pixel-space bounding box across its characters (mirrors
scripts/infer_characters.py, the spiritualist/ALTO version). Unlike HierText,
DocBank already gives an exact bbox per word, so no line-level inference
step is needed first.

ssu_id numbering mirrors scripts/extract_docbank_ssu_text.py exactly: words
are grouped into maximal runs of consecutive identical labels, 1-indexed in
document order, so values line up with the ssu_id column in
data/docbank/gt_ssu_bboxes.csv.

Input:  data/docbank/gt_word_annotations/*.txt
        data/docbank/mscoco_annotations_subset.json (page pixel dimensions)
Output: data/docbank/characters_inferred.parquet
Columns: char_id, page_id, char_text, x, y, w, h, ssu_id
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.extract_docbank_ssu_text import GT_DIR, MSCOCO_PATH, group_into_ssu_runs, parse_word_line

OUTPUT_PATH = Path("data/docbank/characters_inferred.parquet")


def infer_characters_from_word(
    word: dict, page_id: str, ssu_id: int, word_idx: int, width_px: int, height_px: int,
) -> list[dict]:
    x0 = word["x0"] * width_px / 1000
    y0 = word["y0"] * height_px / 1000
    x1 = word["x1"] * width_px / 1000
    y1 = word["y1"] * height_px / 1000
    width, height = x1 - x0, y1 - y0

    chars = [c for c in word["word"] if c != " "]
    if not chars:
        return []

    char_w = width / len(chars)
    records = []
    for char_idx, char in enumerate(chars):
        records.append({
            "char_id": f"{page_id}_s{ssu_id}_w{word_idx}_c{char_idx}",
            "page_id": page_id,
            "char_text": char,
            "x": x0 + char_idx * char_w,
            "y": y0,
            "w": char_w,
            "h": height,
            "ssu_id": ssu_id,
        })
    return records


def main() -> None:
    print(f"Loading {MSCOCO_PATH} ...")
    coco = json.loads(MSCOCO_PATH.read_text())
    image_dims = {img["file_name"]: (img["width"], img["height"]) for img in coco["images"]}

    txt_files = sorted(GT_DIR.glob("*.txt"))
    print(f"  {len(txt_files):,} gt word-annotation files in {GT_DIR}")

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

        for ssu_id, run in enumerate(runs, start=1):
            for word_idx, word in enumerate(run):
                all_records.extend(
                    infer_characters_from_word(word, basename, ssu_id, word_idx, width_px, height_px)
                )

    if skipped:
        print(f"  Skipped {skipped:,} pages not present in {MSCOCO_PATH}")

    df = pd.DataFrame(
        all_records,
        columns=["char_id", "page_id", "char_text", "x", "y", "w", "h", "ssu_id"],
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} characters across {df['page_id'].nunique():,} pages -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

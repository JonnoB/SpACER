"""
Infer character-level bounding boxes from HierText word-level ground truth.

For each word (within each line, within each paragraph/SSU), character
positions are estimated by uniformly distributing the word's axis-aligned
bounding box across its characters. Spaces are excluded from the output.
Mirrors scripts/infer_characters.py, which does the same thing from
word-level ALTO boxes for the spiritualist dataset.

Paragraph numbering mirrors scripts/extract_hiertext_ssu_text.py: paragraphs
are 1-indexed in document order to form ssu_id, so values line up with the
ssu_id column in data/hiertext/gt_ssu_bboxes.csv and the region ids in
data/hiertext_predictions/hiertext_gt_predictions.csv. Only images present in
IMAGE_META are processed, matching the page set used everywhere else in the
hiertext pipeline.

Input:  data/heiertext_validation.jsonl
        data/hiertext_predictions/hiertext_yolo_predictions.csv (page filter only)
Output: data/hiertext/characters_inferred.parquet
Columns: char_id, page_id, char_text, x, y, w, h, ssu_id
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.extract_hiertext_ssu_text import line_bbox

GT_JSON = Path("data/heiertext_validation.jsonl")
IMAGE_META = Path("data/hiertext_predictions/hiertext_yolo_predictions.csv")
OUTPUT_PATH = Path("data/hiertext/characters_inferred.parquet")


def infer_characters_from_entry(entry: dict) -> list[dict]:
    page_id = entry["image_id"]
    records = []
    for para_idx, paragraph in enumerate(entry.get("paragraphs", []), start=1):
        for line_idx, line in enumerate(paragraph.get("lines", [])):
            for word_idx, word in enumerate(line.get("words", [])):
                bbox = line_bbox(word.get("vertices", []))
                if bbox is None:
                    continue
                hpos, vpos, width, height = bbox

                chars = [c for c in word.get("text", "") if c != " "]
                if not chars:
                    continue

                char_w = width / len(chars)
                for char_idx, char in enumerate(chars):
                    char_id = f"{page_id}_p{para_idx}_l{line_idx}_w{word_idx}_c{char_idx}"
                    records.append({
                        "char_id": char_id,
                        "page_id": page_id,
                        "char_text": char,
                        "x": hpos + char_idx * char_w,
                        "y": vpos,
                        "w": char_w,
                        "h": height,
                        "ssu_id": para_idx,
                    })
    return records


def main() -> None:
    print(f"Loading {GT_JSON} ...")
    gt = json.loads(GT_JSON.read_text())
    entries = gt["annotations"]
    print(f"  {len(entries):,} annotated images")

    meta_df = pd.read_csv(IMAGE_META, usecols=["filename"]).drop_duplicates("filename")
    known_filenames = set(meta_df["filename"])
    print(f"  {len(known_filenames):,} images with known dimensions from {IMAGE_META}")

    all_records: list[dict] = []
    skipped = 0
    for entry in entries:
        filename = f"{entry['image_id']}.jpg"
        if filename not in known_filenames:
            skipped += 1
            continue
        all_records.extend(infer_characters_from_entry(entry))

    if skipped:
        print(f"  Skipped {skipped:,} images not present in {IMAGE_META}")

    df = pd.DataFrame(
        all_records,
        columns=["char_id", "page_id", "char_text", "x", "y", "w", "h", "ssu_id"],
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} characters across {df['page_id'].nunique():,} pages -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

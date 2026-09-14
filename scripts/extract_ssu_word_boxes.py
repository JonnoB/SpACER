"""
Extract per-word ground-truth bounding boxes per SSU region from labelled ALTO XML files.

Reads data/spiritualist/ocr_gt_labelled/*.xml and writes:
    data/spiritualist/gt_word_boxes.csv
        columns: filename, page_id, image_width, image_height,
                 ssu_id, x, y, width, height, word_text

Companion to extract_ssu_text.py, which unions all TextLines/TextBlocks
sharing an ssu_id into a single region bbox (or staircase polygon). This
script instead keeps every ALTO <String> (word) box as its own row, tagged
with its parent TextBlock's ssu_id -- the union of those boxes is the GT
region shape when rasterized via cotescore's boxes_to_gt_ssu_map, without
needing a hand-built polygon.

Does not replace gt_ssu_bboxes.csv: text/CER-based downstream consumers
(chars_df, box_level_ocr_comparison.parquet, combined_validation.py's
region counts, ...) still read that one-row-per-SSU file with full gt_text.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ALTO_NS   = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
INPUT_DIR = Path("data/spiritualist/ocr_gt_labelled")
OUTPUT    = Path("data/spiritualist/gt_word_boxes.csv")


def extract_page(xml_path: Path) -> list[dict]:
    page_id  = xml_path.stem
    filename = page_id + ".jpg"
    tree     = ET.parse(xml_path)
    root     = tree.getroot()

    page_el      = root.find(".//alto:Page", ALTO_NS)
    image_width  = int(page_el.attrib.get("WIDTH",  0)) if page_el is not None else 0
    image_height = int(page_el.attrib.get("HEIGHT", 0)) if page_el is not None else 0

    records = []
    for block in root.findall(".//alto:TextBlock", ALTO_NS):
        ssu_id = block.attrib.get("SSU_ID", "").strip()
        if not ssu_id:
            continue

        for line in block.findall("alto:TextLine", ALTO_NS):
            for string_el in line:
                if string_el.tag.split("}")[-1] != "String":
                    continue
                width  = int(string_el.attrib.get("WIDTH",  0))
                height = int(string_el.attrib.get("HEIGHT", 0))
                if width <= 0 or height <= 0:
                    continue
                records.append({
                    "filename":     filename,
                    "page_id":      page_id,
                    "image_width":  image_width,
                    "image_height": image_height,
                    "ssu_id":       ssu_id,
                    "x":            int(string_el.attrib.get("HPOS", 0)),
                    "y":            int(string_el.attrib.get("VPOS", 0)),
                    "width":        width,
                    "height":       height,
                    "word_text":    string_el.attrib.get("CONTENT", ""),
                })
    return records


def main() -> None:
    xml_files = sorted(INPUT_DIR.glob("*.xml"))
    print(f"Processing {len(xml_files)} files …")

    all_records: list[dict] = []
    for xml_path in xml_files:
        records = extract_page(xml_path)
        all_records.extend(records)
        print(f"  {xml_path.name}: {len(records)} word boxes")

    df = pd.DataFrame(all_records, columns=[
        "filename", "page_id", "image_width", "image_height",
        "ssu_id", "x", "y", "width", "height", "word_text",
    ])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df):,} rows → {OUTPUT}")


if __name__ == "__main__":
    main()

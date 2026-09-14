"""
Extract ground-truth boxes/text for Structural Semantic Units from labelled
ALTO XML files, at line granularity, with the SSU-level view derived from
that same line data.

Standalone box-based GT pipeline (no polygon, no dependency on crop_utils.py
/ run_ocr.py / the HierText or DocBank scripts) -- a clean break from the
existing staircase-polygon pipeline (extract_ssu_text.py / gt_ssu_bboxes.csv),
which stays untouched and in production use elsewhere in this repo.

Reads data/spiritualist/ocr_gt_labelled/*.xml and writes:
    data/spiritualist/gt_line_boxes.csv     (primary: one row per TextLine)
        columns: filename, page_id, image_width, image_height,
                 ssu_id, x, y, width, height, line_text
    data/spiritualist/gt_ssu_from_lines.csv (derived: one row per SSU region)
        columns: filename, page_id, image_width, image_height,
                 ssu_id, x, y, width, height, gt_text

Each line's box is recomputed as the min/max envelope of its own <String>
children rather than read from the TextLine's own HPOS/VPOS/WIDTH/HEIGHT
attributes -- those are stale relative to the current String content in
this data (mean +12px, max 117px looser on the left edge than the words
they're supposed to bound, likely inherited from an earlier dedupe stage
that dropped/adjusted Strings without recomputing the parent TextLine
extent). The SSU-level file is an aggregate over that same corrected line
data (bbox = envelope of the SSU's lines, gt_text = line_text joined in
reading order) rather than a second independent parse of the XML, so the
two outputs can't drift apart from each other.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ALTO_NS      = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
INPUT_DIR    = Path("data/spiritualist/ocr_gt_labelled")
LINE_OUTPUT  = Path("data/spiritualist/gt_line_boxes.csv")
SSU_OUTPUT   = Path("data/spiritualist/gt_ssu_from_lines.csv")


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
            strings = [e for e in line if e.tag.split("}")[-1] == "String"]
            if not strings:
                continue

            x1 = min(int(s.attrib.get("HPOS", 0)) for s in strings)
            y1 = min(int(s.attrib.get("VPOS", 0)) for s in strings)
            x2 = max(int(s.attrib.get("HPOS", 0)) + int(s.attrib.get("WIDTH",  0)) for s in strings)
            y2 = max(int(s.attrib.get("VPOS", 0)) + int(s.attrib.get("HEIGHT", 0)) for s in strings)
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue

            records.append({
                "filename":     filename,
                "page_id":      page_id,
                "image_width":  image_width,
                "image_height": image_height,
                "ssu_id":       ssu_id,
                "x":            x1,
                "y":            y1,
                "width":        width,
                "height":       height,
                "line_text":    " ".join(s.attrib.get("CONTENT", "") for s in strings),
            })
    return records


def derive_ssu_rows(line_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate line-level rows into one row per (page_id, ssu_id).

    bbox = envelope (min/max) of the SSU's member line boxes.
    gt_text = each line's line_text joined in reading order (sorted by y).
    """
    ssu_records = []
    group_cols = ["filename", "page_id", "image_width", "image_height", "ssu_id"]
    for keys, grp in line_df.groupby(group_cols, sort=False):
        filename, page_id, image_width, image_height, ssu_id = keys
        grp = grp.sort_values("y")
        x1 = int(grp["x"].min())
        y1 = int(grp["y"].min())
        x2 = int((grp["x"] + grp["width"]).max())
        y2 = int((grp["y"] + grp["height"]).max())
        ssu_records.append({
            "filename":     filename,
            "page_id":      page_id,
            "image_width":  image_width,
            "image_height": image_height,
            "ssu_id":       ssu_id,
            "x":            x1,
            "y":            y1,
            "width":        x2 - x1,
            "height":       y2 - y1,
            "gt_text":      " ".join(grp["line_text"]),
        })
    return pd.DataFrame(ssu_records, columns=[
        "filename", "page_id", "image_width", "image_height",
        "ssu_id", "x", "y", "width", "height", "gt_text",
    ])


def main() -> None:
    xml_files = sorted(INPUT_DIR.glob("*.xml"))
    print(f"Processing {len(xml_files)} files …")

    all_records: list[dict] = []
    for xml_path in xml_files:
        records = extract_page(xml_path)
        all_records.extend(records)
        print(f"  {xml_path.name}: {len(records)} line boxes")

    line_df = pd.DataFrame(all_records, columns=[
        "filename", "page_id", "image_width", "image_height",
        "ssu_id", "x", "y", "width", "height", "line_text",
    ])

    LINE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    line_df.to_csv(LINE_OUTPUT, index=False)
    print(f"\nSaved {len(line_df):,} rows → {LINE_OUTPUT}")

    ssu_df = derive_ssu_rows(line_df)
    ssu_df.to_csv(SSU_OUTPUT, index=False)
    print(f"Saved {len(ssu_df):,} rows → {SSU_OUTPUT}")


if __name__ == "__main__":
    main()

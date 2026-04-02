"""
Extract ground-truth text and bounding boxes per SSU region from labelled ALTO XML files.

Reads data/spiritualist/ocr_gt_labelled/*.xml and writes:
    data/spiritualist/gt_ssu_bboxes.csv
        columns: filename, page_id, image_width, image_height,
                 x, y, width, height, ssu_id, gt_text

The CSV is the single source of truth for SSU boundaries and GT text used by
all downstream scripts.  It replaces the former gt_ssu_text.parquet.

Text is preserved as-is (spaces between words within a line).
Bounding boxes are the union of all TextBlocks sharing the same ssu_id.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.crop_utils import staircase_polygon

ALTO_NS        = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
INPUT_DIR      = Path("data/spiritualist/ocr_gt_labelled")
OUTPUT         = Path("data/spiritualist/gt_ssu_bboxes.csv")
BBOXES_OUT     = Path("data/results_spiritualist/bboxes/spiritualist_gt_predictions.csv")
PAGE_TEXTS     = Path("data/spiritualist/gt_page_texts")


def extract_page(xml_path: Path) -> list[dict]:
    page_id  = xml_path.stem
    filename = page_id + ".jpg"
    tree     = ET.parse(xml_path)
    root     = tree.getroot()

    # Image dimensions from the Page element.
    page_el      = root.find(".//alto:Page", ALTO_NS)
    image_width  = int(page_el.attrib.get("WIDTH",  0)) if page_el is not None else 0
    image_height = int(page_el.attrib.get("HEIGHT", 0)) if page_el is not None else 0

    # One pass: collect text lines and bbox extents per ssu_id, preserving order.
    order:       list[str]              = []
    texts:       dict[str, list[str]]   = {}
    extents:     dict[str, dict]        = {}
    line_tuples: dict[str, list]        = {}

    for block in root.findall(".//alto:TextBlock", ALTO_NS):
        ssu_id = block.attrib.get("SSU_ID", "").strip()
        if not ssu_id:
            continue

        # Bbox: union across all blocks sharing this ssu_id.
        hpos   = int(block.attrib.get("HPOS",   0))
        vpos   = int(block.attrib.get("VPOS",   0))
        width  = int(block.attrib.get("WIDTH",  0))
        height = int(block.attrib.get("HEIGHT", 0))
        x2, y2 = hpos + width, vpos + height

        if ssu_id not in extents:
            extents[ssu_id]     = {"x1": hpos, "y1": vpos, "x2": x2, "y2": y2}
            texts[ssu_id]       = []
            line_tuples[ssu_id] = []
            order.append(ssu_id)
        else:
            b = extents[ssu_id]
            b["x1"] = min(b["x1"], hpos)
            b["y1"] = min(b["y1"], vpos)
            b["x2"] = max(b["x2"], x2)
            b["y2"] = max(b["y2"], y2)

        # Text and polygon: process every TextLine in this block.
        for line in block.findall("alto:TextLine", ALTO_NS):
            words = " ".join(
                e.attrib.get("CONTENT", "")
                for e in line
                if e.tag.split("}")[-1] == "String"
            )
            texts[ssu_id].append(words)
            line_tuples[ssu_id].append((
                int(line.attrib.get("HPOS",   0)),
                int(line.attrib.get("VPOS",   0)),
                int(line.attrib.get("WIDTH",  0)),
                int(line.attrib.get("HEIGHT", 0)),
            ))

    records = []
    for ssu_id in order:
        b = extents[ssu_id]
        records.append({
            "filename":       filename,
            "page_id":        page_id,
            "image_width":    image_width,
            "image_height":   image_height,
            "x":              b["x1"],
            "y":              b["y1"],
            "width":          b["x2"] - b["x1"],
            "height":         b["y2"] - b["y1"],
            "polygon_points": staircase_polygon(line_tuples[ssu_id]) if line_tuples.get(ssu_id) else "",
            "ssu_id":         ssu_id,
            "gt_text":        " ".join(texts[ssu_id]),
        })
    return records


def main() -> None:
    xml_files = sorted(INPUT_DIR.glob("*.xml"))
    print(f"Processing {len(xml_files)} files …")

    PAGE_TEXTS.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for xml_path in xml_files:
        records = extract_page(xml_path)
        all_records.extend(records)
        print(f"  {xml_path.name}: {len(records)} SSU regions")

        page_text = "\n\n".join(r["gt_text"] for r in records)
        (PAGE_TEXTS / f"{xml_path.stem}.txt").write_text(page_text, encoding="utf-8")

    df = pd.DataFrame(all_records, columns=[
        "filename", "page_id", "image_width", "image_height",
        "x", "y", "width", "height", "polygon_points", "ssu_id", "gt_text",
    ])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df):,} rows → {OUTPUT}")

    # Write the OCR pipeline bboxes file (no gt_text; image_path = filename).
    # run_all_ocr.py picks up every CSV from the bboxes dir and names output
    # parquets as {stem}_{ocr_model}_ocr.parquet, so this must be kept in sync
    # with the GT SSU regions whenever extract_ssu_text.py is re-run.
    bboxes_df = df[["filename", "image_width", "image_height", "x", "y", "width", "height", "polygon_points", "ssu_id"]].copy()
    bboxes_df.insert(1, "image_path", bboxes_df["filename"])
    BBOXES_OUT.parent.mkdir(parents=True, exist_ok=True)
    bboxes_df.to_csv(BBOXES_OUT, index=False)
    print(f"Saved {len(bboxes_df):,} rows → {BBOXES_OUT}")
    print(f"Page text files → {PAGE_TEXTS}/")

if __name__ == "__main__":
    main()

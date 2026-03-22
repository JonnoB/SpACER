"""
Extract GT bounding boxes at SSU level from ALTO XML files using word (String)
geometry instead of TextLine geometry.

For each SSU, the bounding box is the union of all <String> (word) bounding
boxes belonging to TextLines in that SSU, rather than the union of the
TextLine boxes themselves.  TextLine boxes in Transkribus are wide horizontal
bands that span the full column width; String boxes hug the actual ink,
producing much tighter SSU crops for OCR evaluation.

If a TextLine has no String children with valid geometry, the TextLine box
is used as a fallback so no SSU is silently dropped.

Output format is identical to extract_gt_ssu_boxes.py and is directly usable
as input to run_docling_ocr.py / run_bbox_ocr.py.

Usage:
    python scripts/extract_gt_ssu_boxes_word.py \\
        --gt-dir data/spiritualist/ocr_gt_with_ssu \\
        --output data/results_spiritualist/spiritualist_gt_ssu_boxes_word.csv

    # Override image directory for use with run_docling_ocr.py:
    python scripts/extract_gt_ssu_boxes_word.py \\
        --gt-dir data/spiritualist/ocr_gt_with_ssu \\
        --image-dir /teamspace/lightning_storage/the_spiritualist/spiritualist_images \\
        --output data/results_spiritualist/spiritualist_gt_ssu_boxes_word.csv
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}

# Local tag names (without namespace prefix) that carry word geometry.
# SP = space, HYP = hyphen — these are layout artefacts, not real words.
_WORD_TAGS = {"String"}


def _local_tag(element: ET.Element) -> str:
    """Return the tag name without its namespace URI."""
    tag = element.tag
    return tag[tag.index("}") + 1:] if "}" in tag else tag


def _expand_ssu_box(ssu_boxes: dict, ssu: str, x1: float, y1: float, x2: float, y2: float) -> None:
    if ssu not in ssu_boxes:
        ssu_boxes[ssu] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    else:
        b = ssu_boxes[ssu]
        b["x1"] = min(b["x1"], x1)
        b["y1"] = min(b["y1"], y1)
        b["x2"] = max(b["x2"], x2)
        b["y2"] = max(b["y2"], y2)


def extract_ssu_boxes_word(xml_path: Path, image_dir: Path | None) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    page = root.find(".//alto:Page", ALTO_NS)
    image_width = int(page.attrib["WIDTH"])
    image_height = int(page.attrib["HEIGHT"])

    filename = xml_path.stem + ".jpg"
    image_path = str(image_dir / filename) if image_dir is not None else filename

    ssu_boxes: dict[str, dict] = {}

    for tl in root.findall(".//alto:TextLine[@SSU]", ALTO_NS):
        ssu = tl.attrib["SSU"]
        word_count = 0

        for child in tl:
            if _local_tag(child) not in _WORD_TAGS:
                continue
            try:
                hpos = float(child.attrib["HPOS"])
                vpos = float(child.attrib["VPOS"])
                w = float(child.attrib["WIDTH"])
                h = float(child.attrib["HEIGHT"])
            except (KeyError, ValueError):
                continue
            if w <= 0 or h <= 0:
                continue
            _expand_ssu_box(ssu_boxes, ssu, hpos, vpos, hpos + w, vpos + h)
            word_count += 1

        # Fallback: use the TextLine box if no valid String children found
        if word_count == 0:
            try:
                hpos = float(tl.attrib["HPOS"])
                vpos = float(tl.attrib["VPOS"])
                w = float(tl.attrib["WIDTH"])
                h = float(tl.attrib["HEIGHT"])
                _expand_ssu_box(ssu_boxes, ssu, hpos, vpos, hpos + w, vpos + h)
            except (KeyError, ValueError):
                pass

    rows = []
    for ssu, b in ssu_boxes.items():
        rows.append({
            "filename": filename,
            "image_path": image_path,
            "image_width": image_width,
            "image_height": image_height,
            "source": "gt",
            "x": b["x1"],
            "y": b["y1"],
            "width": b["x2"] - b["x1"],
            "height": b["y2"] - b["y1"],
            "class": "text",
            "confidence": 1.0,
            "ssu_id": ssu,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract SSU-level GT bounding boxes from ALTO XML files using word geometry."
    )
    parser.add_argument("--gt-dir", required=True, help="Directory containing ALTO XML files with SSU attributes")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Image directory to use for image_path column (default: filename only)",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(gt_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files in {gt_dir}")

    all_rows = []
    for xml_path in xml_files:
        rows = extract_ssu_boxes_word(xml_path, image_dir)
        all_rows.extend(rows)
        print(f"  {xml_path.name}: {len(rows)} SSUs")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\nWrote {len(df):,} rows to {output_path}")


if __name__ == "__main__":
    main()

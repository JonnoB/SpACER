"""
Extract GT bounding boxes at SSU level from ALTO XML files.

For each SSU, the bounding box is the maximum extent (union) of all TextLines
belonging to that SSU. Output matches the predictions CSV format used by
run_docling_ocr.py and related scripts.

Usage:
    python scripts/extract_gt_ssu_boxes.py \\
        --gt-dir data/spiritualist/ocr_gt_with_ssu \\
        --output data/results_spiritualist/spiritualist_gt_ssu_boxes.csv

    # Override image directory for use with run_docling_ocr.py:
    python scripts/extract_gt_ssu_boxes.py \\
        --gt-dir data/spiritualist/ocr_gt_with_ssu \\
        --image-dir /teamspace/lightning_storage/the_spiritualist/spiritualist_images \\
        --output data/results_spiritualist/spiritualist_gt_ssu_boxes.csv
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}


def extract_ssu_boxes(xml_path: Path, image_dir: Path | None) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    page = root.find(".//alto:Page", ALTO_NS)
    image_width = int(page.attrib["WIDTH"])
    image_height = int(page.attrib["HEIGHT"])

    filename = xml_path.stem + ".jpg"
    if image_dir is not None:
        image_path = str(image_dir / filename)
    else:
        image_path = filename

    # Group TextLines by SSU attribute
    ssu_boxes: dict[str, dict] = {}
    for tl in root.findall(".//alto:TextLine[@SSU]", ALTO_NS):
        ssu = tl.attrib["SSU"]
        hpos = float(tl.attrib["HPOS"])
        vpos = float(tl.attrib["VPOS"])
        w = float(tl.attrib["WIDTH"])
        h = float(tl.attrib["HEIGHT"])
        right = hpos + w
        bottom = vpos + h

        if ssu not in ssu_boxes:
            ssu_boxes[ssu] = {"x1": hpos, "y1": vpos, "x2": right, "y2": bottom}
        else:
            b = ssu_boxes[ssu]
            b["x1"] = min(b["x1"], hpos)
            b["y1"] = min(b["y1"], vpos)
            b["x2"] = max(b["x2"], right)
            b["y2"] = max(b["y2"], bottom)

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
        description="Extract SSU-level GT bounding boxes from ALTO XML files."
    )
    parser.add_argument("--gt-dir", required=True, help="Directory containing ALTO XML files")
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
        rows = extract_ssu_boxes(xml_path, image_dir)
        all_rows.extend(rows)
        print(f"  {xml_path.name}: {len(rows)} SSUs")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\nWrote {len(df):,} rows to {output_path}")


if __name__ == "__main__":
    main()

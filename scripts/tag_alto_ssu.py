#!/usr/bin/env python
"""Tag Spiritualist ALTO XML files with SSU identifiers and extract SSU bounding boxes.

Tags every ALTO XML in INPUT_DIR, writes tagged copies to OUTPUT_DIR, and writes
a single CSV of SSU bounding boxes (one row per SSU per page) to CSV_PATH.  The
CSV is the authoritative source for SSU boundaries used by the analysis notebook;
generating it here ensures it is always in sync with the tagged XML.

Usage:
    python scripts/tag_alto_ssu.py
"""

import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "cotescore/src"))

from cotescore import assign_alto_ssu

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

INPUT_DIR = repo_root / "data/spiritualist/ocr_gt"
OUTPUT_DIR = repo_root / "data/spiritualist/ocr_gt_with_ssu"
IMAGE_DIR = repo_root / "data/spiritualist/spiritualist_images"
CSV_PATH = repo_root / "data/results_spiritualist/spiritualist_gt_ssu_boxes.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}


def _page_dimensions(xml_path: Path) -> tuple[int, int]:
    """Return (width, height) from the ALTO Page element."""
    root = ET.parse(xml_path).getroot()
    page = root.find(".//alto:Page", ALTO_NS)
    if page is None:
        return 0, 0
    return int(page.attrib.get("WIDTH", 0)), int(page.attrib.get("HEIGHT", 0))


def _ssu_boxes_from_result(result: dict, xml_path: Path) -> list[dict]:
    """Compute per-SSU bounding boxes (union of member TextLine bboxes)."""
    image_width, image_height = _page_dimensions(xml_path)
    filename = xml_path.stem + ".jpg"
    image_path = str(IMAGE_DIR / filename)

    ssu_extents: dict[str, dict] = {}
    for line_id, ssu_id in result["line_to_ssu"].items():
        hpos, vpos, w, h = result["line_metadata"][line_id]["bbox"]
        x2, y2 = hpos + w, vpos + h
        if ssu_id not in ssu_extents:
            ssu_extents[ssu_id] = {"x1": hpos, "y1": vpos, "x2": x2, "y2": y2}
        else:
            b = ssu_extents[ssu_id]
            b["x1"] = min(b["x1"], hpos)
            b["y1"] = min(b["y1"], vpos)
            b["x2"] = max(b["x2"], x2)
            b["y2"] = max(b["y2"], y2)

    rows = []
    for ssu_id, b in ssu_extents.items():
        rows.append({
            "filename": filename,
            "image_path": image_path,
            "image_width": image_width,
            "image_height": image_height,
            "x": b["x1"],
            "y": b["y1"],
            "width": b["x2"] - b["x1"],
            "height": b["y2"] - b["y1"],
            "ssu_id": ssu_id,
        })
    return rows


all_rows: list[dict] = []
errors = 0

for xml_file in sorted(INPUT_DIR.glob("*.xml")):
    output_path = OUTPUT_DIR / xml_file.name
    try:
        result = assign_alto_ssu(str(xml_file), output_path=str(output_path))
        rows = _ssu_boxes_from_result(result, output_path)
        all_rows.extend(rows)
        logging.info("%-30s  %d SSUs", xml_file.name, len(result["ssu_to_lines"]))
    except Exception as exc:
        logging.error("%-30s  FAILED: %s", xml_file.name, exc)
        errors += 1

df = pd.DataFrame(all_rows)
df.to_csv(CSV_PATH, index=False)
logging.info("Wrote %d SSU boxes to %s", len(df), CSV_PATH)

sys.exit(1 if errors else 0)

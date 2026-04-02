#!/usr/bin/env python
"""Process ALTO XML files using image-region labels to assign SSU tags.

Labels are loaded from a Labelbox NDJSON export containing bounding-box
annotations with five classes: masthead, header, text, advert, other.

For each labelled page the script:
  1. Classifies every TextLine by which label region its centroid falls in.
  2. Splits mixed-class TextBlocks into pure-class sub-blocks.
  3. Detects columns from text/advert line midpoints (gap-based).
  4. Assigns columns in two passes (direct midpoint, then extent fallback).
  5. Determines reading order (top-to-bottom within columns, left-to-right).
  6. Assigns semantic units (headers start units; adverts are own unit).
  7. Computes SSU IDs as (semantic_unit, column) intersections.
  8. Rewrites the ALTO XML with restructured TextBlocks and new attributes.

Pages without label data are skipped.

Usage:
    uv run scripts/process_alto_labelled.py \\
        --labels data/spiritualist/spiritualist_class_labels.ndjson \\
        --xml-dir data/spiritualist/ocr_gt_dedupe \\
        --output-dir output/labelled/
"""

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Optional

from lxml import etree

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.crop_utils import staircase_polygon

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
NS = {"a": ALTO_NS}

# Gap between sorted TextLine midpoints that signals a column boundary,
# expressed as a fraction of page width.
COLUMN_GAP_FRACTION = 0.08


# ---------------------------------------------------------------------------
# Stage 0 — load labels
# ---------------------------------------------------------------------------

def load_labels(ndjson_path: Path) -> dict[str, dict]:
    """Parse Labelbox NDJSON and return a mapping from page stem to label data.

    Returns:
        {page_stem: {"img_width": int, "img_height": int, "regions": [...]}}
        where each region is {"class_name": str, "top": float, "left": float,
                               "height": float, "width": float}
    """
    result: dict[str, dict] = {}
    with ndjson_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            external_id: str = record["data_row"]["external_id"]
            page_stem = Path(external_id).stem  # e.g. "0001_p001"

            img_w = record["media_attributes"]["width"]
            img_h = record["media_attributes"]["height"]

            projects = record.get("projects", {})
            regions = []
            for proj in projects.values():
                labels = proj.get("labels", [])
                if not labels:
                    continue
                for obj in labels[0]["annotations"]["objects"]:
                    bb = obj["bounding_box"]
                    regions.append({
                        "class_name": obj["name"],
                        "top": bb["top"],
                        "left": bb["left"],
                        "height": bb["height"],
                        "width": bb["width"],
                    })

            result[page_stem] = {
                "img_width": img_w,
                "img_height": img_h,
                "regions": regions,
            }
    log.info("Loaded labels for %d pages", len(result))
    return result


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _intersection_area(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> float:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _classify_line(
    hpos: int, vpos: int, width: int, height: int,
    scaled_regions: list[dict],
) -> str:
    """Return the class name of the best-matching label region for a TextLine.

    Uses the centroid for containment; breaks ties by overlap area.
    Returns "unknown" if no region contains the centroid.
    """
    cx = hpos + width / 2.0
    cy = vpos + height / 2.0

    best_class = "unknown"
    best_area = -1.0

    lx1, ly1, lx2, ly2 = hpos, vpos, hpos + width, vpos + height

    for r in scaled_regions:
        rx1 = r["left"]
        ry1 = r["top"]
        rx2 = rx1 + r["width"]
        ry2 = ry1 + r["height"]

        # Check centroid containment
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            area = _intersection_area(lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2)
            if area > best_area:
                best_area = area
                best_class = r["class_name"]

    return best_class


def _block_bbox_from_lines(lines: list[dict]) -> tuple[int, int, int, int]:
    """Return (hpos, vpos, width, height) enclosing all lines."""
    x1 = min(l["hpos"] for l in lines)
    y1 = min(l["vpos"] for l in lines)
    x2 = max(l["hpos"] + l["width"] for l in lines)
    y2 = max(l["vpos"] + l["height"] for l in lines)
    return x1, y1, x2 - x1, y2 - y1


# ---------------------------------------------------------------------------
# Stage 1 — parse ALTO and classify TextLines
# ---------------------------------------------------------------------------

def parse_and_classify(
    xml_path: Path,
    label_data: dict,
) -> tuple[int, int, list[dict], list[dict]]:
    """Parse the ALTO XML, classify every TextLine, return blocks and raw lines.

    Returns:
        (page_width, page_height, blocks, all_lines)

        Each block dict:
            {block_id, hpos, vpos, width, height, element, lines: [line_dict]}

        Each line dict:
            {line_id, hpos, vpos, width, height, class_name,
             orig_block_id, element}
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    page_el = root.xpath("//a:Page", namespaces=NS)[0]
    page_w = int(page_el.get("WIDTH", 0))
    page_h = int(page_el.get("HEIGHT", 0))

    img_w = label_data["img_width"]
    img_h = label_data["img_height"]
    scale_x = page_w / img_w
    scale_y = page_h / img_h

    # Scale label regions to ALTO coordinate space
    scaled_regions = []
    for r in label_data["regions"]:
        scaled_regions.append({
            "class_name": r["class_name"],
            "left":   r["left"]   * scale_x,
            "top":    r["top"]    * scale_y,
            "width":  r["width"]  * scale_x,
            "height": r["height"] * scale_y,
        })

    blocks = []
    all_lines = []

    for block_el in root.xpath("//a:TextBlock", namespaces=NS):
        bid = block_el.get("ID")
        b_hpos   = int(block_el.get("HPOS",   0))
        b_vpos   = int(block_el.get("VPOS",   0))
        b_width  = int(block_el.get("WIDTH",  0))
        b_height = int(block_el.get("HEIGHT", 0))

        lines = []
        for line_el in block_el.xpath("a:TextLine", namespaces=NS):
            lid = line_el.get("ID")
            l_hpos   = int(line_el.get("HPOS",   0))
            l_vpos   = int(line_el.get("VPOS",   0))
            l_width  = int(line_el.get("WIDTH",  0))
            l_height = int(line_el.get("HEIGHT", 0))

            class_name = _classify_line(
                l_hpos, l_vpos, l_width, l_height, scaled_regions
            )

            line_info = {
                "line_id":       lid,
                "hpos":          l_hpos,
                "vpos":          l_vpos,
                "width":         l_width,
                "height":        l_height,
                "class_name":    class_name,
                "orig_block_id": bid,
                "element":       line_el,
            }
            lines.append(line_info)
            all_lines.append(line_info)

        blocks.append({
            "block_id":  bid,
            "hpos":      b_hpos,
            "vpos":      b_vpos,
            "width":     b_width,
            "height":    b_height,
            "element":   block_el,
            "lines":     lines,
        })

    return page_w, page_h, blocks, all_lines, scaled_regions, tree


# ---------------------------------------------------------------------------
# Stage 2 — restructure TextBlocks into pure-class sub-blocks
# ---------------------------------------------------------------------------

def restructure_blocks(original_blocks: list[dict]) -> list[dict]:
    """Split any TextBlock that crosses a class boundary into per-run sub-blocks.

    Lines are processed in VPOS order and split into contiguous runs of the
    same class (run-length encoding). A header line therefore acts as a natural
    semantic barrier: text lines on either side of a header run become separate
    sub-blocks and will be assigned to different SSUs downstream.

    This correctly handles:
      - Homogeneous blocks → single sub-block, original ID preserved.
      - Adjacent same-class regions with no class change → kept together.
      - Blocks spanning a header separator → split at the header boundary.

    Returns a flat list of new block dicts:
        {block_id, orig_block_id, class_name, lines,
         hpos, vpos, width, height,
         column_id, reading_order, semantic_id, ssu_id}
    """
    new_blocks = []

    for block in original_blocks:
        if not block["lines"]:
            continue

        # Sort lines by VPOS to establish document order, then build
        # contiguous runs of the same class.
        sorted_lines = sorted(block["lines"], key=lambda l: l["vpos"])

        runs: list[tuple[str, list[dict]]] = []
        for line in sorted_lines:
            cls = line["class_name"]
            if not runs or runs[-1][0] != cls:
                runs.append((cls, []))
            runs[-1][1].append(line)

        if len(runs) == 1:
            # Homogeneous and unsplit — keep original block ID
            cls, lines = runs[0]
            hpos, vpos, width, height = _block_bbox_from_lines(lines)
            new_blocks.append({
                "block_id":      block["block_id"],
                "orig_block_id": block["block_id"],
                "class_name":    cls,
                "lines":         lines,
                "hpos":          hpos,
                "vpos":          vpos,
                "width":         width,
                "height":        height,
                "column_id":     None,
                "reading_order": -1,
                "semantic_id":   -1,
                "ssu_id":        "",
            })
        else:
            # Multiple runs — suffix each sub-block ID with class and run index
            for i, (cls, lines) in enumerate(runs):
                hpos, vpos, width, height = _block_bbox_from_lines(lines)
                new_blocks.append({
                    "block_id":      f"{block['block_id']}__{cls}_{i}",
                    "orig_block_id": block["block_id"],
                    "class_name":    cls,
                    "lines":         lines,
                    "hpos":          hpos,
                    "vpos":          vpos,
                    "width":         width,
                    "height":        height,
                    "column_id":     None,
                    "reading_order": -1,
                    "semantic_id":   -1,
                    "ssu_id":        "",
                })

    return new_blocks


# ---------------------------------------------------------------------------
# Stage 3 — gap-based column detection
# ---------------------------------------------------------------------------

def detect_columns(scaled_regions: list[dict], page_width: int) -> list[dict]:
    """Infer columns from the x-centres of text and advert label regions.

    Using label regions rather than TextLine midpoints avoids noise from
    wide Transkribus TextBlocks whose lines span column boundaries.

    Returns list of column dicts sorted left-to-right:
        {column_id, x_center, x_left, x_right}
    """
    content_classes = {"text", "advert"}
    region_extents = []  # (cx, left, right)
    for r in scaled_regions:
        if r["class_name"] not in content_classes:
            continue
        left  = r["left"]
        right = r["left"] + r["width"]
        cx    = (left + right) / 2.0
        region_extents.append((cx, left, right))

    if not region_extents:
        log.warning("No text/advert label regions found for column detection")
        return [{"column_id": 1, "x_center": page_width / 2.0,
                 "x_left": 0.0, "x_right": float(page_width)}]

    sorted_by_cx = sorted(region_extents, key=lambda t: t[0])
    gap_threshold = COLUMN_GAP_FRACTION * page_width

    # Split into clusters at large horizontal gaps between region centres
    clusters: list[list[tuple]] = [[sorted_by_cx[0]]]
    for prev, curr in zip(sorted_by_cx, sorted_by_cx[1:]):
        if curr[0] - prev[0] > gap_threshold:
            clusters.append([])
        clusters[-1].append(curr)

    columns = []
    for i, cluster in enumerate(clusters, start=1):
        cxs    = [t[0] for t in cluster]
        lefts  = [t[1] for t in cluster]
        rights = [t[2] for t in cluster]
        columns.append({
            "column_id": i,
            "x_center":  statistics.median(cxs),
            "x_left":    statistics.median(lefts),
            "x_right":   statistics.median(rights),
        })

    log.info("Detected %d columns", len(columns))
    return columns


# ---------------------------------------------------------------------------
# Stage 4 — two-pass column assignment
# ---------------------------------------------------------------------------

def _nearest_column(cx: float, columns: list[dict]) -> int:
    return min(columns, key=lambda c: abs(cx - c["x_center"]))["column_id"]


def assign_columns(new_blocks: list[dict], columns: list[dict]) -> None:
    """Assign column_id to each block in-place.

    Pass 1: text and advert blocks — assign by midpoint to nearest column.
    Pass 2: header and unknown blocks — assign by column extent, fallback nearest.
    Masthead and other blocks are left with column_id = None.
    """
    skip_classes = {"masthead", "other"}

    # Pass 1: text + advert
    for block in new_blocks:
        if block["class_name"] in skip_classes:
            continue
        if block["class_name"] not in {"text", "advert"}:
            continue
        cx = block["hpos"] + block["width"] / 2.0
        block["column_id"] = _nearest_column(cx, columns)

    # Pass 2: header + unknown (+ any text/advert still unassigned)
    for block in new_blocks:
        if block["class_name"] in skip_classes:
            continue
        if block["column_id"] is not None:
            continue
        cx = block["hpos"] + block["width"] / 2.0
        # Try extent match first
        matched = None
        for col in columns:
            if col["x_left"] <= cx <= col["x_right"]:
                matched = col["column_id"]
                break
        block["column_id"] = matched if matched is not None else _nearest_column(cx, columns)


# ---------------------------------------------------------------------------
# Stage 5 — reading order
# ---------------------------------------------------------------------------

def assign_reading_order(new_blocks: list[dict]) -> None:
    """Sort column-assigned blocks by (column_id, min_vpos) and set reading_order."""
    content = [b for b in new_blocks if b["column_id"] is not None]
    content.sort(key=lambda b: (b["column_id"], b["vpos"]))
    for i, block in enumerate(content):
        block["reading_order"] = i
    # masthead/other stay at -1


# ---------------------------------------------------------------------------
# Stage 6 — semantic unit assignment
# ---------------------------------------------------------------------------

def assign_semantic_units(new_blocks: list[dict]) -> None:
    """Walk blocks in reading order and assign semantic_id in-place.

    Rules:
      - masthead blocks: semantic_id = 0
      - other/unknown blocks: semantic_id keyed to orig_block_id (negative int)
      - header: starts a new SU (unless previous content block was also a header)
      - advert: always its own SU
      - text/unknown-in-column: continues current SU
    """
    # Sort by reading_order; blocks with -1 (masthead/other) handled separately
    ordered = sorted(
        [b for b in new_blocks if b["reading_order"] >= 0],
        key=lambda b: b["reading_order"],
    )

    # Masthead
    for b in new_blocks:
        if b["class_name"] == "masthead":
            b["semantic_id"] = 0

    # Other — allocate unique negative IDs keyed by orig_block_id so that lines
    # from the same original TextBlock share an SSU
    other_sem: dict[str, int] = {}
    other_counter = -1
    for b in new_blocks:
        if b["class_name"] == "other":
            key = b["orig_block_id"]
            if key not in other_sem:
                other_sem[key] = other_counter
                other_counter -= 1
            b["semantic_id"] = other_sem[key]

    # Column content — state machine
    sem_id = 0
    prev_class: Optional[str] = None

    for block in ordered:
        cls = block["class_name"]

        if cls == "header":
            if prev_class != "header":
                sem_id += 1
            block["semantic_id"] = sem_id

        elif cls == "advert":
            sem_id += 1
            block["semantic_id"] = sem_id

        elif cls in ("text", "unknown"):
            if sem_id == 0:
                sem_id = 1
            block["semantic_id"] = sem_id

        # masthead and other already handled above; skip
        prev_class = cls if cls not in ("masthead", "other") else prev_class


# ---------------------------------------------------------------------------
# Stage 7 — SSU assignment
# ---------------------------------------------------------------------------

def assign_ssus(new_blocks: list[dict]) -> None:
    """Compute ssu_id for every block in-place."""
    for block in new_blocks:
        cls = block["class_name"]
        sem = block["semantic_id"]
        col = block["column_id"]

        if cls == "masthead":
            block["ssu_id"] = "ssu_masthead"
        elif cls == "other":
            block["ssu_id"] = f"ssu_other_{block['orig_block_id']}"
        elif sem < 0:
            # Shouldn't happen for column content, but be defensive
            block["ssu_id"] = f"ssu_unknown_{block['block_id']}"
        else:
            block["ssu_id"] = f"ssu_{sem}_col_{col}"


# ---------------------------------------------------------------------------
# Stage 8 — rewrite ALTO XML
# ---------------------------------------------------------------------------

def _make_textblock_element(block: dict) -> etree._Element:
    """Create a new <TextBlock> lxml element from a new_block dict."""
    tb = etree.Element(f"{{{ALTO_NS}}}TextBlock")
    tb.set("ID",     block["block_id"])
    tb.set("HPOS",   str(block["hpos"]))
    tb.set("VPOS",   str(block["vpos"]))
    tb.set("WIDTH",  str(block["width"]))
    tb.set("HEIGHT", str(block["height"]))
    tb.set("BLOCK_TYPE",    block["class_name"].upper())
    tb.set("COLUMN_ID",     str(block["column_id"]) if block["column_id"] is not None else "")
    tb.set("READING_ORDER", str(block["reading_order"]))
    tb.set("SEMANTIC_ID",   str(block["semantic_id"]))
    tb.set("SSU_ID",        block["ssu_id"])

    # Shape polygon — staircase from constituent TextLine bboxes
    shape = etree.SubElement(tb, f"{{{ALTO_NS}}}Shape")
    poly  = etree.SubElement(shape, f"{{{ALTO_NS}}}Polygon")
    line_tuples = [(l["hpos"], l["vpos"], l["width"], l["height"]) for l in block["lines"]]
    poly.set("POINTS", staircase_polygon(line_tuples))

    # Re-attach TextLine elements (already parsed; detach from their current parent)
    for line_info in block["lines"]:
        line_el = line_info["element"]
        # Remove from current parent if attached
        parent = line_el.getparent()
        if parent is not None:
            parent.remove(line_el)
        tb.append(line_el)

    return tb


def rewrite_xml(
    tree: etree._ElementTree,
    new_blocks: list[dict],
    output_path: Path,
) -> None:
    """Remove all original TextBlocks and insert restructured ones."""
    root = tree.getroot()
    print_space = root.xpath("//a:PrintSpace", namespaces=NS)[0]

    # Remove existing TextBlock children
    for tb in print_space.xpath("a:TextBlock", namespaces=NS):
        print_space.remove(tb)

    # Insertion order: masthead → column content (by reading_order) → other/unknown
    def sort_key(b: dict) -> tuple:
        cls = b["class_name"]
        if cls == "masthead":
            return (0, 0, b["vpos"])
        if b["reading_order"] >= 0:
            return (1, b["reading_order"], 0)
        return (2, 0, b["vpos"])

    for block in sorted(new_blocks, key=sort_key):
        tb_el = _make_textblock_element(block)
        print_space.append(tb_el)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        etree.tostring(
            root,
            pretty_print=True,
            xml_declaration=True,
            encoding="UTF-8",
        )
    )
    log.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_page(
    xml_path: Path,
    label_data: dict,
    output_dir: Path,
) -> None:
    page_stem = xml_path.stem
    log.info("=== Processing %s ===", xml_path.name)

    page_w, page_h, orig_blocks, all_lines, scaled_regions, tree = parse_and_classify(
        xml_path, label_data
    )

    class_counts: dict[str, int] = {}
    for l in all_lines:
        class_counts[l["class_name"]] = class_counts.get(l["class_name"], 0) + 1
    log.info("  Line classes: %s", class_counts)

    new_blocks = restructure_blocks(orig_blocks)
    log.info("  Blocks after restructure: %d (was %d)", len(new_blocks), len(orig_blocks))

    columns = detect_columns(scaled_regions, page_w)
    assign_columns(new_blocks, columns)
    assign_reading_order(new_blocks)
    assign_semantic_units(new_blocks)
    assign_ssus(new_blocks)

    out_path = output_dir  / f"{page_stem}.xml"
    rewrite_xml(tree, new_blocks, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process labelled ALTO XML files with label-driven SSU tagging"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/spiritualist/spiritualist_class_labels.ndjson"),
        help="Labelbox NDJSON export",
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=Path("data/spiritualist/ocr_gt_dedupe"),
        help="Directory containing ALTO XML files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/spiritualist/ocr_gt_labelled"),
        help="Output directory",
    )
    args = parser.parse_args()

    all_labels = load_labels(args.labels)
    xml_files = sorted(args.xml_dir.glob("*.xml"))

    errors = 0
    processed = 0
    skipped = 0
    for xml_path in xml_files:
        page_stem = xml_path.stem
        if page_stem not in all_labels:
            log.info("Skipping %s (no label data)", xml_path.name)
            skipped += 1
            continue
        try:
            process_page(xml_path, all_labels[page_stem], args.output_dir)
            processed += 1
        except Exception as exc:
            log.error("FAILED %s: %s", xml_path.name, exc, exc_info=True)
            errors += 1

    log.info("Done: %d processed, %d skipped, %d errors", processed, skipped, errors)
    if errors:
        raise SystemExit(f"{errors} page(s) failed")


if __name__ == "__main__":
    main()

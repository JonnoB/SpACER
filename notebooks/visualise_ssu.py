import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import io
    import sys
    import xml.etree.ElementTree as ET
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont

    # Reuse the exact polygon logic from the extract pipeline.
    sys.path.insert(0, str(Path.cwd()))
    from scripts.crop_utils import column_polygon

    ENRICHED_DIR = Path("data/spiritualist/ocr_gt_labelled")
    IMAGE_DIR = Path("data/spiritualist/spiritualist_images")

    pages = sorted(p.stem for p in ENRICHED_DIR.glob("*.xml"))
    return (
        ENRICHED_DIR,
        ET,
        IMAGE_DIR,
        Image,
        ImageDraw,
        ImageFont,
        column_polygon,
        io,
        mo,
        pages,
        pd,
    )


@app.cell
def _(ENRICHED_DIR, ET, IMAGE_DIR, Image, ImageDraw, ImageFont, column_polygon, io, pd):
    """Helper functions — no output."""

    def _envelope(rects):
        """Min/max (x1, y1, x2, y2) envelope of a list of (x, y, w, h) rects."""
        x1 = min(r[0] for r in rects)
        y1 = min(r[1] for r in rects)
        x2 = max(r[0] + r[2] for r in rects)
        y2 = max(r[1] + r[3] for r in rects)
        return x1, y1, x2, y2

    def parse_page(page: str):
        """Return (ssus, ssu_ids) from the enriched XML.

        Descends TextBlock -> TextLine -> String so each SSU carries its true
        word-level shape, not just the coarse block-union bbox. Each line box is
        recomputed as the envelope of its own String children (the parent
        TextLine HPOS/VPOS is stale/loose in this data), matching the
        extract_ssu_* pipeline. Falls back to TextLine attrs if a line has no
        String children.
        """
        xml_path = ENRICHED_DIR / f"{page}.xml"
        root = ET.parse(xml_path).getroot()
        ns_uri = root.tag[1:root.tag.index("}")] if root.tag.startswith("{") else ""
        ns = {"a": ns_uri} if ns_uri else {}
        q = lambda t: f"a:{t}" if ns else t

        ssus: dict = {}
        for tb in root.findall(f".//{q('TextBlock')}", ns):
            sid = tb.get("SSU_ID")
            if sid is None:
                continue
            entry = ssus.setdefault(sid, {
                "words": [], "lines": [],
                "meta": {
                    "block_type": tb.get("BLOCK_TYPE", ""),
                    "semantic_id": tb.get("SEMANTIC_ID", ""),
                    "column_id": tb.get("COLUMN_ID", ""),
                },
            })
            for line in tb.findall(q("TextLine"), ns):
                line_words = []
                for s in line.findall(q("String"), ns):
                    wx, wy = int(s.get("HPOS", 0)), int(s.get("VPOS", 0))
                    ww, wh = int(s.get("WIDTH", 0)), int(s.get("HEIGHT", 0))
                    line_words.append((wx, wy, ww, wh))
                if line_words:
                    entry["words"].extend(line_words)
                    lx1, ly1, lx2, ly2 = _envelope(line_words)
                    entry["lines"].append((lx1, ly1, lx2 - lx1, ly2 - ly1))
                else:  # fall back to (loose) TextLine attrs
                    lx, ly = int(line.get("HPOS", 0)), int(line.get("VPOS", 0))
                    lw, lh = int(line.get("WIDTH", 0)), int(line.get("HEIGHT", 0))
                    entry["lines"].append((lx, ly, lw, lh))

        # Precompute per SSU: block-union bbox, simple column polygon, and a
        # flag for the GT representation this SSU should use. The masthead's
        # polygon is simple but ~63% whitespace (scattered full-width
        # fragments), so it is represented by boxes instead; every other block
        # type gets the tight, simple column_polygon.
        for sid, entry in ssus.items():
            rects = entry["lines"] or entry["words"]
            entry["bbox"] = _envelope(rects) if rects else (0, 0, 0, 0)
            entry["use_boxes"] = entry["meta"]["block_type"].upper() == "MASTHEAD"
            if entry["lines"]:
                poly_str = column_polygon(entry["lines"])
                entry["polygon"] = [
                    tuple(map(int, p.split(","))) for p in poly_str.split()
                ]
            else:
                entry["polygon"] = []

        return ssus, sorted(ssus.keys())

    def ssu_color(sid: str):
        h = abs(hash(sid))
        return ((h >> 0) % 180 + 60, (h >> 8) % 180 + 60, (h >> 16) % 180 + 60)

    def render(page: str, ssus: dict, ssu_ids: list, selected: set, mode: str):
        """Return (image_bytes, table_df).

        mode:
          'gt'      the GT representation each SSU will actually use — masthead
                    as word boxes, every other SSU as its column_polygon
          'polygon' force the column_polygon for every SSU (incl. the loose
                    masthead), to inspect the polygon itself
          'words'   per-word boxes for every SSU
          'block'   coarse block-union bbox for every SSU
        """
        img = Image.open(IMAGE_DIR / f"{page}.jpg").convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except OSError:
            font = ImageFont.load_default()

        for sid in ssu_ids:
            if sid not in selected:
                continue
            entry, color = ssus[sid], ssu_color(sid)
            draw_words = mode == "words" or (mode == "gt" and entry["use_boxes"])
            draw_poly = len(entry["polygon"]) >= 3 and (
                mode == "polygon" or (mode == "gt" and not entry["use_boxes"])
            )
            if draw_words:
                for wx, wy, ww, wh in entry["words"]:
                    draw.rectangle([wx, wy, wx + ww, wy + wh], outline=color, width=2)
            elif draw_poly:
                draw.polygon(entry["polygon"], outline=color, width=4)
            else:  # 'block' (or polygon-less SSU)
                x1, y1, x2, y2 = entry["bbox"]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
            bx1, by1, _, _ = entry["bbox"]
            draw.text((bx1 + 6, by1 + 6), f"SSU {sid}", fill=color, font=font)

        factor = min(1.0, 1600 / max(img.width, img.height))
        disp = img.resize((int(img.width * factor), int(img.height * factor)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        disp.save(buf, format="PNG")

        table_df = pd.DataFrame([
            {"SSU": sid, "type": ssus[sid]["meta"]["block_type"],
             "gt": "boxes" if ssus[sid]["use_boxes"] else "polygon",
             "semantic": ssus[sid]["meta"]["semantic_id"],
             "column": ssus[sid]["meta"]["column_id"],
             "words": len(ssus[sid]["words"]), "lines": len(ssus[sid]["lines"]),
             "w": ssus[sid]["bbox"][2] - ssus[sid]["bbox"][0],
             "h": ssus[sid]["bbox"][3] - ssus[sid]["bbox"][1]}
            for sid in ssu_ids
        ])
        return buf.getvalue(), table_df
    return parse_page, render


@app.cell
def _(mo, pages):
    page_picker = mo.ui.dropdown(options=pages, value=pages[0], label="Page")
    page_picker
    return (page_picker,)


@app.cell
def _(ENRICHED_DIR, ET, page_picker, parse_page):
    # silence the unused import warning — parse_page uses ENRICHED_DIR/ET via closure
    _ = ENRICHED_DIR, ET
    ssus, ssu_ids = parse_page(page_picker.value)
    return ssus, ssu_ids


@app.cell
def _(mo, ssu_ids):
    ssu_picker = mo.ui.multiselect(options=ssu_ids, value=ssu_ids, label="Highlight SSUs")
    mode_picker = mo.ui.radio(
        options=["gt", "polygon", "words", "block"], value="gt", label="SSU shape", inline=True
    )
    mo.vstack([mode_picker, ssu_picker])
    return mode_picker, ssu_picker


@app.cell
def _(mo, mode_picker, page_picker, render, ssu_ids, ssu_picker, ssus):
    img_bytes, table_df = render(
        page_picker.value, ssus, ssu_ids, set(ssu_picker.value), mode_picker.value
    )
    mo.hstack([
        mo.image(img_bytes),
        mo.ui.table(table_df),
    ], align="start", gap=2)
    return


if __name__ == "__main__":
    app.run()

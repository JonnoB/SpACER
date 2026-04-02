import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import io
    import xml.etree.ElementTree as ET
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont

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
        io,
        mo,
        pages,
        pd,
    )


@app.cell
def _(ENRICHED_DIR, ET, IMAGE_DIR, Image, ImageDraw, ImageFont, io, pd):
    """Helper functions — no output."""

    def parse_page(page: str):
        """Return (ssu_boxes, ssu_ids, ssu_meta) from the enriched XML."""
        xml_path = ENRICHED_DIR / f"{page}.xml"
        root = ET.parse(xml_path).getroot()
        ns_uri = root.tag[1:root.tag.index("}")] if root.tag.startswith("{") else ""
        ns = {"a": ns_uri} if ns_uri else {}
        q = lambda t: f"a:{t}" if ns else t

        boxes: dict = {}
        meta: dict = {}
        for tb in root.findall(f".//{q('TextBlock')}", ns):
            sid = tb.get("SSU_ID")
            if sid is None:
                continue
            x, y = int(tb.get("HPOS", 0)), int(tb.get("VPOS", 0))
            w, h = int(tb.get("WIDTH", 0)), int(tb.get("HEIGHT", 0))
            if sid not in boxes:
                boxes[sid] = {"x1": x, "y1": y, "x2": x + w, "y2": y + h}
                meta[sid] = {
                    "block_type": tb.get("BLOCK_TYPE", ""),
                    "semantic_id": tb.get("SEMANTIC_ID", ""),
                    "column_id": tb.get("COLUMN_ID", ""),
                }
            else:
                b = boxes[sid]
                b["x1"], b["y1"] = min(b["x1"], x), min(b["y1"], y)
                b["x2"], b["y2"] = max(b["x2"], x + w), max(b["y2"], y + h)

        ssu_ids = sorted(boxes.keys())
        return boxes, ssu_ids, meta

    def ssu_color(sid: str):
        h = abs(hash(sid))
        return ((h >> 0) % 180 + 60, (h >> 8) % 180 + 60, (h >> 16) % 180 + 60)

    def render(page: str, boxes: dict, ssu_ids: list, meta: dict, selected: set):
        """Return (image_bytes, table_df)."""
        img = Image.open(IMAGE_DIR / f"{page}.jpg").convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except OSError:
            font = ImageFont.load_default()

        for sid in ssu_ids:
            if sid not in selected:
                continue
            b, color = boxes[sid], ssu_color(sid)
            draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=color, width=5)
            draw.text((b["x1"] + 6, b["y1"] + 6), f"SSU {sid}", fill=color, font=font)

        factor = min(1.0, 1600 / max(img.width, img.height))
        disp = img.resize((int(img.width * factor), int(img.height * factor)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        disp.save(buf, format="PNG")

        table_df = pd.DataFrame([
            {"SSU": sid, "type": meta[sid]["block_type"],
             "semantic": meta[sid]["semantic_id"], "column": meta[sid]["column_id"],
             "w": boxes[sid]["x2"] - boxes[sid]["x1"], "h": boxes[sid]["y2"] - boxes[sid]["y1"]}
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
    ssu_boxes, ssu_ids, ssu_meta = parse_page(page_picker.value)
    return ssu_boxes, ssu_ids, ssu_meta


@app.cell
def _(mo, ssu_ids):
    ssu_picker = mo.ui.multiselect(options=ssu_ids, value=ssu_ids, label="Highlight SSUs")
    ssu_picker
    return (ssu_picker,)


@app.cell
def _(mo, page_picker, render, ssu_boxes, ssu_ids, ssu_meta, ssu_picker):
    img_bytes, table_df = render(page_picker.value, ssu_boxes, ssu_ids, ssu_meta, set(ssu_picker.value))
    mo.hstack([
        mo.image(img_bytes),
        mo.ui.table(table_df),
    ], align="start", gap=2)
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import io
    import sys
    import xml.etree.ElementTree as ET
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, str(Path("..").resolve() / "cotescore/src"))
    return ET, Image, ImageDraw, ImageFont, Path, io, mo, pd


@app.cell
def _(Path, mo, pd):
    ssu_csv = Path("data/results_spiritualist/spiritualist_gt_ssu_boxes.csv")
    ocr_parquet_dir = Path("data/results_spiritualist/docling_ocr_gt_s_star")
    image_dir = Path("data/spiritualist/spiritualist_images")
    alto_dir = Path("data/spiritualist/ocr_gt_with_ssu")

    _ssu_all = pd.read_csv(ssu_csv)
    pages = sorted(_ssu_all["filename"].str.replace(r"\.\w+$", "", regex=True).unique())

    page_picker = mo.ui.dropdown(options=pages, value=pages[0], label="Page")
    show_lines = mo.ui.checkbox(label="Show line-level boxes", value=True)
    show_word_ssus = mo.ui.checkbox(label="Show word-unioned SSU boxes", value=False)
    mo.vstack([page_picker, show_lines, show_word_ssus])
    return alto_dir, image_dir, ocr_parquet_dir, page_picker, pages, show_lines, show_word_ssus, ssu_csv


@app.cell
def _(
    ET,
    Image,
    ImageDraw,
    ImageFont,
    alto_dir,
    image_dir,
    io,
    ocr_parquet_dir,
    page_picker,
    pd,
    show_lines,
    show_word_ssus,
    ssu_csv,
):
    """Load page data and render SSU boxes on a scaled copy of the image."""
    page = page_picker.value

    # SSU boundaries always come from the CSV (in sync with tagged XML)
    _all = pd.read_csv(ssu_csv)
    df = _all[_all["filename"] == f"{page}.jpg"].copy().reset_index(drop=True)

    # Optionally join OCR text from parquet when available
    _parquet_path = ocr_parquet_dir / f"{page}.parquet"
    if _parquet_path.exists():
        _ocr = pd.read_parquet(_parquet_path)[["ssu_id", "ocr_text"]].drop_duplicates("ssu_id")
        df = df.merge(_ocr, on="ssu_id", how="left")
        df["ocr_text"] = df["ocr_text"].fillna("")
    else:
        df["ocr_text"] = ""

    img = Image.open(image_dir / f"{page}.jpg").convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    def _color_for_ssu(ssu_id):
        h = abs(hash(ssu_id))
        return ((h >> 0) % 180 + 50, (h >> 8) % 180 + 50, (h >> 16) % 180 + 50)

    def _pale(color):
        """Lighten a colour for thin line boxes so they don't compete with SSU outlines."""
        return tuple(min(255, int(c + (255 - c) * 0.55)) for c in color)

    def _mid(color):
        """Mid-tone variant for word-unioned SSU boxes (between pale and full)."""
        return tuple(min(255, int(c + (255 - c) * 0.25)) for c in color)

    # --- line-level boxes from ALTO XML (thin, pale, same SSU colour) ---
    if show_lines.value:
        alto_path = alto_dir / f"{page}.xml"
        if alto_path.exists():
            _tree = ET.parse(alto_path)
            _root = _tree.getroot()
            _ns_uri = _root.tag[1:_root.tag.index("}")] if _root.tag.startswith("{") else ""
            _ns = {"alto": _ns_uri} if _ns_uri else {}
            _q = lambda t: f"alto:{t}" if _ns else t
            for _tl in _root.findall(f".//{_q('TextLine')}", _ns):
                _ssu = _tl.get("SSU")
                if _ssu is None:
                    continue
                try:
                    lx = int(_tl.get("HPOS", 0))
                    ly = int(_tl.get("VPOS", 0))
                    lw = int(_tl.get("WIDTH", 0))
                    lh = int(_tl.get("HEIGHT", 0))
                except (ValueError, TypeError):
                    continue
                draw.rectangle(
                    [lx, ly, lx + lw, ly + lh],
                    outline=_pale(_color_for_ssu(_ssu)),
                    width=1,
                )

    # --- word-unioned SSU boxes (medium weight, mid-tone, same colour scheme) ---
    if show_word_ssus.value:
        _alto_path2 = alto_dir / f"{page}.xml"
        if _alto_path2.exists():
            _tree2 = ET.parse(_alto_path2)
            _root2 = _tree2.getroot()
            _ns_uri2 = _root2.tag[1:_root2.tag.index("}")] if _root2.tag.startswith("{") else ""
            _ns2 = {"alto": _ns_uri2} if _ns_uri2 else {}
            _q2 = lambda t: f"alto:{t}" if _ns2 else t

            _word_ssu: dict = {}
            for _tl in _root2.findall(f".//{_q2('TextLine')}", _ns2):
                _ssu = _tl.get("SSU")
                if _ssu is None:
                    continue
                _word_count = 0
                for _ch in _tl:
                    _ltag = _ch.tag[_ch.tag.index("}") + 1:] if "}" in _ch.tag else _ch.tag
                    if _ltag != "String":
                        continue
                    try:
                        _wx = float(_ch.attrib["HPOS"])
                        _wy = float(_ch.attrib["VPOS"])
                        _ww = float(_ch.attrib["WIDTH"])
                        _wh = float(_ch.attrib["HEIGHT"])
                    except (KeyError, ValueError):
                        continue
                    if _ww <= 0 or _wh <= 0:
                        continue
                    if _ssu not in _word_ssu:
                        _word_ssu[_ssu] = {"x1": _wx, "y1": _wy, "x2": _wx + _ww, "y2": _wy + _wh}
                    else:
                        _b = _word_ssu[_ssu]
                        _b["x1"] = min(_b["x1"], _wx)
                        _b["y1"] = min(_b["y1"], _wy)
                        _b["x2"] = max(_b["x2"], _wx + _ww)
                        _b["y2"] = max(_b["y2"], _wy + _wh)
                    _word_count += 1
                # fallback to TextLine if no String children
                if _word_count == 0:
                    try:
                        _wx = float(_tl.attrib["HPOS"])
                        _wy = float(_tl.attrib["VPOS"])
                        _ww = float(_tl.attrib["WIDTH"])
                        _wh = float(_tl.attrib["HEIGHT"])
                        if _ssu not in _word_ssu:
                            _word_ssu[_ssu] = {"x1": _wx, "y1": _wy, "x2": _wx + _ww, "y2": _wy + _wh}
                        else:
                            _b = _word_ssu[_ssu]
                            _b["x1"] = min(_b["x1"], _wx)
                            _b["y1"] = min(_b["y1"], _wy)
                            _b["x2"] = max(_b["x2"], _wx + _ww)
                            _b["y2"] = max(_b["y2"], _wy + _wh)
                    except (KeyError, ValueError):
                        pass

            for _ssu, _b in _word_ssu.items():
                draw.rectangle(
                    [int(_b["x1"]), int(_b["y1"]), int(_b["x2"]), int(_b["y2"])],
                    outline=_mid(_color_for_ssu(_ssu)),
                    width=2,
                )

    # --- SSU-level boxes from parquet (thick outline + label) ---
    for _, row in df.iterrows():
        x, y, w, h = int(row.x), int(row.y), int(row.width), int(row.height)
        color = _color_for_ssu(row.ssu_id)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=4)
        label = row.ssu_id.replace("ssu_", "").replace("_col_", "/c").replace("_span_", "/s")
        draw.text((x + 4, y + 4), label, fill=color, font=font)

    max_dim = 1400
    factor = min(1.0, max_dim / max(img.width, img.height))
    disp = img.resize((int(img.width * factor), int(img.height * factor)), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    disp.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    return df, img_bytes, page


@app.cell
def _(df, mo):
    """Build OCR summary table and SSU picker."""
    summary = df[["ssu_id", "x", "y", "width", "height", "ocr_text"]].copy()
    summary["ocr_preview"] = summary["ocr_text"].str[:120].str.replace("\n", " ") + "…"

    ssu_options = summary["ssu_id"].tolist()
    ssu_picker = mo.ui.dropdown(options=ssu_options, value=ssu_options[0], label="Inspect SSU")
    return ssu_picker, summary


@app.cell
def _(Image, ImageDraw, image_dir, io, page, ssu_picker, summary):
    """Crop the selected SSU region from the original image."""
    sel = summary[summary["ssu_id"] == ssu_picker.value].iloc[0]

    orig = Image.open(image_dir / f"{page}.jpg").convert("RGB")
    pad = 20
    cx1 = max(0, int(sel.x) - pad)
    cy1 = max(0, int(sel.y) - pad)
    cx2 = min(orig.width, int(sel.x) + int(sel.width) + pad)
    cy2 = min(orig.height, int(sel.y) + int(sel.height) + pad)
    crop = orig.crop((cx1, cy1, cx2, cy2))

    crop_draw = ImageDraw.Draw(crop)
    crop_draw.rectangle(
        [pad, pad, pad + int(sel.width), pad + int(sel.height)],
        outline=(220, 50, 50),
        width=3,
    )

    crop_buf = io.BytesIO()
    crop.save(crop_buf, format="PNG")
    crop_bytes = crop_buf.getvalue()

    full_ocr = sel["ocr_text"]
    return crop_bytes, full_ocr, sel


@app.cell
def _(crop_bytes, full_ocr, img_bytes, mo, sel, ssu_picker, summary):
    """Layout: annotated page image on the left, OCR table + SSU inspector on the right."""
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("### SSU boxes on page"),
                    mo.image(img_bytes),
                ],
                align="start",
            ),
            mo.vstack(
                [
                    mo.md("### OCR results by SSU"),
                    mo.ui.table(
                        summary[["ssu_id", "width", "height", "ocr_preview"]].rename(
                            columns={
                                "ssu_id": "SSU",
                                "width": "W",
                                "height": "H",
                                "ocr_preview": "OCR preview (first 120 chars)",
                            }
                        )
                    ),
                    mo.md("---"),
                    mo.md("### Inspect individual SSU"),
                    ssu_picker,
                    mo.hstack(
                        [
                            mo.vstack([mo.md("**Crop**"), mo.image(crop_bytes, width=400)]),
                            mo.vstack(
                                [
                                    mo.md(
                                        f"**Box:** x={int(sel.x)}, y={int(sel.y)}, "
                                        f"w={int(sel.width)}, h={int(sel.height)}"
                                    ),
                                    mo.md("**Full OCR text:**"),
                                    mo.Html(
                                        "<div style='"
                                        "max-height:400px;overflow-y:auto;overflow-x:hidden;"
                                        "white-space:pre-wrap;word-break:break-word;"
                                        "font-family:monospace;font-size:0.85em;"
                                        "border:1px solid #ccc;border-radius:4px;padding:8px;"
                                        "width:420px'>"
                                        + full_ocr.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                                        + "</div>"
                                    ),
                                ]
                            ),
                        ],
                        align="start",
                    ),
                ],
                align="start",
            ),
        ],
        align="start",
        gap=2,
    )
    return


@app.cell
def _(ET, Image, ImageDraw, alto_dir, image_dir, io, mo, page, pd):
    """
    Duplicate word detection.

    Extracts all <String> elements from the ALTO XML for the current page and
    checks for pairs whose bounding boxes overlap substantially (IoU >= threshold).
    Any such pair means the same physical region contributes to two different SSUs,
    causing double-counting in aggregate metrics.
    """
    import numpy as np
    from difflib import SequenceMatcher

    _iou_threshold = 0.5   # flag pairs with spatial IoU >= this value
    _sim_threshold = 0.8   # only keep pairs where content is also highly similar

    _alto_path = alto_dir / f"{page}.xml"
    _tree3 = ET.parse(_alto_path)
    _root3 = _tree3.getroot()
    _ns_uri3 = _root3.tag[1:_root3.tag.index("}")] if _root3.tag.startswith("{") else ""
    _ns3 = {"alto": _ns_uri3} if _ns_uri3 else {}
    _q3 = lambda t: f"alto:{t}" if _ns3 else t

    _words = []
    for _tl3 in _root3.findall(f".//{_q3('TextLine')}", _ns3):
        _ssu3 = _tl3.get("SSU", "")
        _lid3 = _tl3.get("ID", "")
        for _s3 in _tl3:
            _ltag3 = _s3.tag[_s3.tag.index("}") + 1:] if "}" in _s3.tag else _s3.tag
            if _ltag3 != "String":
                continue
            try:
                _wx3 = float(_s3.attrib["HPOS"])
                _wy3 = float(_s3.attrib["VPOS"])
                _ww3 = float(_s3.attrib["WIDTH"])
                _wh3 = float(_s3.attrib["HEIGHT"])
            except (KeyError, ValueError):
                continue
            if _ww3 <= 0 or _wh3 <= 0:
                continue
            _words.append({
                "content": _s3.get("CONTENT", ""),
                "x1": _wx3, "y1": _wy3,
                "x2": _wx3 + _ww3, "y2": _wy3 + _wh3,
                "ssu": _ssu3, "line_id": _lid3,
            })

    # Vectorised pairwise IoU
    _n = len(_words)
    _duplicates = []
    _flagged = set()

    if _n > 0:
        _arr = np.array([[_w["x1"], _w["y1"], _w["x2"], _w["y2"]] for _w in _words], dtype=np.float32)
        _ix1 = np.maximum(_arr[:, 0:1], _arr[None, :, 0])
        _iy1 = np.maximum(_arr[:, 1:2], _arr[None, :, 1])
        _ix2 = np.minimum(_arr[:, 2:3], _arr[None, :, 2])
        _iy2 = np.minimum(_arr[:, 3:4], _arr[None, :, 3])
        _inter = np.maximum(0, _ix2 - _ix1) * np.maximum(0, _iy2 - _iy1)
        _areas = (_arr[:, 2] - _arr[:, 0]) * (_arr[:, 3] - _arr[:, 1])
        _union = _areas[:, None] + _areas[None, :] - _inter
        _iou_mat = np.where(_union > 0, _inter / _union, 0.0)
        _pr, _pc = np.where(np.triu(_iou_mat >= _iou_threshold, k=1))
        for _ri, _ci in zip(_pr.tolist(), _pc.tolist()):
            _wa, _wb = _words[_ri], _words[_ci]
            _sim = round(SequenceMatcher(None, _wa["content"], _wb["content"]).ratio(), 3)
            if _sim < _sim_threshold:
                continue
            _duplicates.append({
                "word_A": _wa["content"], "ssu_A": _wa["ssu"],
                "word_B": _wb["content"], "ssu_B": _wb["ssu"],
                "IoU": round(float(_iou_mat[_ri, _ci]), 3),
                "content_sim": _sim,
                "x1_A": int(_wa["x1"]), "y1_A": int(_wa["y1"]),
                "x1_B": int(_wb["x1"]), "y1_B": int(_wb["y1"]),
            })
            _flagged.add(_ri)
            _flagged.add(_ci)

    dup_df = pd.DataFrame(_duplicates) if _duplicates else pd.DataFrame(
        columns=["word_A", "ssu_A", "word_B", "ssu_B", "IoU", "content_sim", "x1_A", "y1_A", "x1_B", "y1_B"]
    )
    n_words = len(_words)
    n_dups = len(_duplicates)

    # Render flagged boxes on image
    _orig3 = Image.open(image_dir / f"{page}.jpg").convert("RGB")
    _draw3 = ImageDraw.Draw(_orig3)
    for _idx in _flagged:
        _fw = _words[_idx]
        _draw3.rectangle(
            [int(_fw["x1"]), int(_fw["y1"]), int(_fw["x2"]), int(_fw["y2"])],
            outline=(220, 30, 30), width=3,
        )
    _maxd = 1400
    _fd = min(1.0, _maxd / max(_orig3.width, _orig3.height))
    _dispd = _orig3.resize((int(_orig3.width * _fd), int(_orig3.height * _fd)), Image.Resampling.LANCZOS)
    _bufd = io.BytesIO()
    _dispd.save(_bufd, format="PNG")
    dup_img_bytes = _bufd.getvalue()

    return dup_df, dup_img_bytes, n_dups, n_words


@app.cell
def _(dup_df, dup_img_bytes, mo, n_dups, n_words):
    """Display duplicate word analysis."""
    status = (
        mo.callout(mo.md(f"**No duplicate word positions found** across {n_words} words."), kind="success")
        if n_dups == 0
        else mo.callout(
            mo.md(
                f"**{n_dups} overlapping word pairs** found across {n_words} words "
                f"(IoU ≥ 0.5). These words are double-counted across SSU regions."
            ),
            kind="warn",
        )
    )
    mo.vstack([
        mo.md("---\n### Duplicate / overlapping word positions"),
        status,
        mo.hstack([
            mo.vstack([mo.md("**Flagged positions (red)**"), mo.image(dup_img_bytes)]),
            mo.vstack([
                mo.md("**Duplicate pairs**"),
                mo.ui.table(dup_df[["word_A", "ssu_A", "word_B", "ssu_B", "IoU", "content_sim"]]) if n_dups > 0
                else mo.md("_None_"),
            ]),
        ], align="start"),
    ])


if __name__ == "__main__":
    app.run()

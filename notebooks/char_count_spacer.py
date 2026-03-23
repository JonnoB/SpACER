import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import xml.etree.ElementTree as ET
    from collections import Counter
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    from cotescore.ocr import spacer, text_to_counter, spacer_micro
    return Counter, ET, Path, mo, pd, spacer, text_to_counter


@app.cell
def _(ET, Path, pd, text_to_counter):
    """Build or load GT character count parquet (Q)."""

    _ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
    _GT_DIR = Path("data/spiritualist/ocr_gt_with_ssu")
    _GT_PARQUET = Path("data/results_spiritualist/char_counts_gt.parquet")

    def _extract_gt_text(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        tokens = [s.attrib["CONTENT"] for s in root.findall(".//alto:String", _ALTO_NS)]
        return " ".join(tokens)

    def _build_gt_parquet(gt_dir, out_path):
        rows = []
        for xml_path in sorted(gt_dir.glob("*.xml")):
            page_id = xml_path.stem
            text = _extract_gt_text(xml_path)
            counter = text_to_counter(text, mode="char")
            for char, count in counter.items():
                rows.append({"page_id": page_id, "char": char, "count": count})
        df = pd.DataFrame(rows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return df

    if _GT_PARQUET.exists():
        gt_df = pd.read_parquet(_GT_PARQUET)
        print(f"Loaded GT parquet: {len(gt_df):,} rows, {gt_df['page_id'].nunique()} pages")
    else:
        gt_df = _build_gt_parquet(_GT_DIR, _GT_PARQUET)
        print(f"Built GT parquet: {len(gt_df):,} rows, {gt_df['page_id'].nunique()} pages")

    gt_df
    return (gt_df,)


@app.cell
def _(Path, pd, text_to_counter):
    """Build or load predicted character count parquet."""

    _PRED_DIR = Path("data/results_spiritualist/results_dotsmocr_300dpi_wholepage")
    _PRED_PARQUET = Path("data/results_spiritualist/char_counts_pred_dotsmocr_300dpi.parquet")

    def _build_pred_parquet(pred_dir, out_path):
        rows = []
        for txt_path in sorted(pred_dir.glob("*.txt")):
            page_id = txt_path.stem
            text = txt_path.read_text(encoding="utf-8")
            counter = text_to_counter(text, mode="char")
            for char, count in counter.items():
                rows.append({"page_id": page_id, "char": char, "count": count})
        df = pd.DataFrame(rows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return df

    if _PRED_PARQUET.exists():
        pred_df = pd.read_parquet(_PRED_PARQUET)
        print(f"Loaded pred parquet: {len(pred_df):,} rows, {pred_df['page_id'].nunique()} pages")
    else:
        pred_df = _build_pred_parquet(_PRED_DIR, _PRED_PARQUET)
        print(f"Built pred parquet: {len(pred_df):,} rows, {pred_df['page_id'].nunique()} pages")

    pred_df
    return (pred_df,)


@app.cell
def _(Counter, gt_df, pd, pred_df, spacer):
    """Compute per-page SpACER scores."""

    def _df_to_counters(df):
        return {
            page_id: Counter(dict(zip(grp["char"], grp["count"])))
            for page_id, grp in df.groupby("page_id")
        }

    _gt_counters = _df_to_counters(gt_df)
    _pred_counters = _df_to_counters(pred_df)

    _rows = []
    for _page_id in sorted(set(_gt_counters) & set(_pred_counters)):
        _score = spacer(_gt_counters[_page_id], _pred_counters[_page_id])
        _rows.append({
            "page_id": _page_id,
            "spacer": _score,
            "gt_chars": sum(_gt_counters[_page_id].values()),
            "pred_chars": sum(_pred_counters[_page_id].values()),
        })

    spacer_df = pd.DataFrame(_rows)
    spacer_df
    return (spacer_df,)


@app.cell
def _(spacer_df):
    spacer_df
    return


@app.cell
def _(mo, spacer_df):
    """Summary statistics."""

    _summary_md = (
        f"**Mean SpACER:** {spacer_df['spacer'].mean():.4f}  |  "
        f"**Median:** {spacer_df['spacer'].median():.4f}  |  "
        f"**Min:** {spacer_df['spacer'].min():.4f}  |  "
        f"**Max:** {spacer_df['spacer'].max():.4f}"
    )
    mo.vstack([
        mo.md(f"### SpACER results — {len(spacer_df)} pages"),
        mo.md(_summary_md),
        mo.ui.table(spacer_df.sort_values("spacer", ascending=False).reset_index(drop=True)),
    ])
    return


@app.cell
def _(spacer_df):
    spacer_df.describe()
    return


if __name__ == "__main__":
    app.run()

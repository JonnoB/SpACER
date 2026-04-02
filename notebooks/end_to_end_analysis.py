import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import json
    from pathlib import Path
    from collections import Counter
    from cotescore import spacer, cdd_decomp
    import plotnine as p9
    import re
    import unicodedata
    return Counter, Path, cdd_decomp, json, mo, p9, pd, re, spacer, unicodedata


@app.cell
def _(re, unicodedata):
    def _normalize_quotes(text):
        text = re.sub(r'[\u2018\u2019\u201a\u201b\u2039\u203a`]', "'", text)
        text = re.sub(r'[\u201c\u201d\u201e\u201f\u00ab\u00bb]', '"', text)
        return text

    def _normalize_dashes(text):
        text = re.sub(r'[\u2013\u2014\u2015\u2012]', '-', text)
        return text

    def normalize_for_cer(text):
        text = text.lower()
        text = unicodedata.normalize('NFKC', text)
        text = _normalize_quotes(text)
        text = _normalize_dashes(text)
        text = text.replace('\xa0', ' ')
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
    return (normalize_for_cer,)


@app.cell
def _(Path):
    _REPO = Path(__file__).resolve().parent.parent
    _END_TO_END = _REPO / "data/results_spiritualist/end_to_end"
    _CROPS_DIR = _END_TO_END / "crops"
    _SSU_BBOXES = _REPO / "data/spiritualist/gt_ssu_bboxes.csv"
    repo_root = _REPO
    end_to_end_dir = _END_TO_END
    crops_dir = _CROPS_DIR
    ssu_bboxes_path = _SSU_BBOXES
    return crops_dir, end_to_end_dir, ssu_bboxes_path


@app.cell
def _(pd, ssu_bboxes_path):
    def _ssu_n(ssu_id):
        try:
            return int(ssu_id.split("_")[1])
        except (IndexError, ValueError):
            return -1

    _ssu = pd.read_csv(ssu_bboxes_path)
    if "page_id" not in _ssu.columns:
        _ssu["page_id"] = _ssu["filename"].str.removesuffix(".jpg")
    _ssu["ssu_n"] = _ssu["ssu_id"].map(_ssu_n)
    _ssu = _ssu.sort_values(["page_id", "ssu_n", "y", "x"])

    gt_ssu_df = _ssu[["page_id", "ssu_id", "ssu_n", "x", "y", "gt_text"]]

    gt_page_text = (
        _ssu.groupby("page_id")["gt_text"]
        .apply(lambda texts: " ".join(t for t in texts if isinstance(t, str) and t.strip()))
        .to_dict()
    )
    return (gt_page_text,)


@app.cell
def _(crops_dir, pd):
    _files = {
        "mocr": crops_dir / "dotsmocr_crop.parquet",
        "olmocr": crops_dir / "olmocr_crop_results.parquet",
        "granite": crops_dir / "granite_docling_crop.parquet",
    }
    _parts = [pd.read_parquet(p).assign(model=name) for name, p in _files.items() if p.exists()]
    crop_df = pd.concat(_parts, ignore_index=True) if _parts else pd.DataFrame()
    crop_df
    return (crop_df,)


@app.cell
def _(end_to_end_dir, json, pd):
    _records = []

    # olmocr page
    _olmocr = pd.read_parquet(end_to_end_dir / "olmocr_page_results.parquet")
    for _, _row in _olmocr.iterrows():
        _records.append({"model": "olmocr", "filename": _row["filename"], "ocr_text": _row["ocr_text"]})

    # DOTS 300dpi text files
    for _p in sorted((end_to_end_dir / "results_dotsmocr_300dpi_wholepage").glob("*.txt")):
        _records.append({
            "model": "mocr",
            "filename": _p.stem + ".jpg",
            "ocr_text": _p.read_text(encoding="utf-8"),
        })

    # Docling pages JSON
    for _p in sorted((end_to_end_dir / "results_granite_docling_pages").glob("*.json")):
        _data = json.loads(_p.read_text(encoding="utf-8"))
        _text = " ".join(el["text"] for el in _data.get("elements", []) if el.get("text"))
        _records.append({
            "model": "granite",
            "filename": _p.stem + ".jpg",
            "ocr_text": _text,
        })

    page_df = pd.DataFrame(_records)
    page_df
    return (page_df,)


@app.cell
def _(
    Counter,
    cdd_decomp,
    crop_df,
    gt_page_text,
    normalize_for_cer,
    pd,
    spacer,
):
    def _strip(text):
        return normalize_for_cer(text).replace(" ", "")

    _records = []
    # extract reading-order index directly from ssu_id (e.g. "ssu_3_col_1" → 3)
    def _ssu_n(ssu_id):
        try:
            return int(ssu_id.split("_")[1])
        except (IndexError, ValueError):
            return -1

    _crop = crop_df.copy()
    _crop["ssu_n"] = _crop["ssu_id"].map(_ssu_n)

    for (_model, _filename), _grp in _crop.groupby(["model", "filename"]):
        _page_id = _filename.removesuffix(".jpg")
        if _page_id not in gt_page_text:
            continue

        _grp = _grp.sort_values(["ssu_n", "y", "x"])

        _pred_text = " ".join(
            t for t in _grp["ocr_text"] if isinstance(t, str) and t.strip()
        )
        _gt_text = gt_page_text[_page_id]

        _gt_stripped = _strip(_gt_text)
        _pred_stripped = _strip(_pred_text)

        if not _gt_stripped:
            continue

        _spacer_val = spacer(Counter(_gt_stripped), Counter(_pred_stripped))
        _cdd_val = cdd_decomp({"gt": _gt_stripped, "ocr": _pred_stripped}).d_ocr
        _char_coverage = len(_pred_stripped) / len(_gt_stripped) * 100

        _records.append({
            "model": _model,
            "filename": _filename,
            "metric_type": "d_ocr",
            "spacer": _spacer_val,
            "cdd": _cdd_val,
            "char_coverage_pct": _char_coverage,
        })

    crop_results_df = pd.DataFrame(_records)
    crop_results_df
    return (crop_results_df,)


@app.cell
def _(
    Counter,
    cdd_decomp,
    gt_page_text,
    normalize_for_cer,
    page_df,
    pd,
    spacer,
):
    def _strip(text):
        return normalize_for_cer(text).replace(" ", "")

    _records = []
    for _, _row in page_df.iterrows():
        _page_id = _row["filename"].removesuffix(".jpg")
        if _page_id not in gt_page_text:
            continue

        _gt_text = gt_page_text[_page_id]
        _pred_text = _row["ocr_text"] if isinstance(_row["ocr_text"], str) else ""

        _gt_stripped = _strip(_gt_text)
        _pred_stripped = _strip(_pred_text)

        if not _gt_stripped:
            continue

        _spacer_val = spacer(Counter(_gt_stripped), Counter(_pred_stripped))
        _cdd_val = cdd_decomp({"gt": _gt_stripped, "total": _pred_stripped}).d_total
        _char_coverage = len(_pred_stripped) / len(_gt_stripped) * 100

        _records.append({
            "model": _row["model"],
            "filename": _row["filename"],
            "metric_type": "d_total",
            "spacer": _spacer_val,
            "cdd": _cdd_val,
            "char_coverage_pct": _char_coverage,
        })

    page_results_df = pd.DataFrame(_records)
    page_results_df
    return (page_results_df,)


@app.cell
def _(crop_results_df, page_results_df, pd):
    results_df = pd.concat([crop_results_df, page_results_df], ignore_index=True)
    results_df['d_diff'] = results_df['spacer'] - results_df['cdd']

    results_df
    return (results_df,)


@app.cell
def _(results_df):
    _df = results_df.loc[results_df['metric_type']=='d_total', ['model', 'spacer', 'cdd']].groupby('model').median().round(3)

    latex_str = _df.to_latex(
        index=True,
        caption="Median spacer and CDD by model (d\\_total)",
        label="tab:model_median",
        position="h",
        column_format="l" + "r" * len(_df.columns),  # left for index, right for numeric cols
    )

    print(latex_str)
    return


@app.cell
def _(mo, results_df):
    _summary = (
        results_df
        .groupby(["model", "metric_type"])[["spacer", "cdd", "char_coverage_pct"]]
        .mean()
        .round(4)
        .reset_index()
    )
    mo.ui.table(_summary)
    return


@app.cell
def _(p9, results_df):
    p9.ggplot(results_df.loc[results_df['metric_type'] == 'd_total'], p9.aes(x='spacer', y='cdd', colour='model')) + p9.geom_point()
    return


@app.cell
def _(p9, results_df):
    p9.ggplot(results_df.loc[(results_df['metric_type'] == 'd_total') & ~results_df['model'].isin(['docling_pages'])], p9.aes(x='spacer', y='cdd', colour='model')) + p9.geom_point() + p9.ylim(0,0.1)  + p9.xlim(0,0.2)
    return


@app.cell
def _(results_df):
    results_df.loc[(results_df['metric_type'] == 'd_total') & ~results_df['model'].isin(['docling_pages'])]
    return


@app.cell
def _(p9, results_df):
    _plot_df = (
        results_df.melt(
            id_vars=["model", "metric_type"],
            value_vars=["spacer", "cdd"],
            var_name="metric",
            value_name="score",
        )
        .groupby(["model", "metric_type", "metric"], as_index=False)["score"]
        .median()
    )
    (
        p9.ggplot(_plot_df, p9.aes(x="model", y="score", fill="metric_type"))
        + p9.geom_col(position="dodge")
        + p9.facet_wrap("~metric", scales="free_y")
        + p9.theme(axis_text_x=p9.element_text(angle=30, hjust=1), figure_size=(12, 6))
        + p9.labs(title="SpACER and CDD scores by model", x="Model", y="Score", fill="Metric type")
    )
    return


if __name__ == "__main__":
    app.run()

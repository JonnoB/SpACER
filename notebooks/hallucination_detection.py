import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import numpy as np
    from collections import Counter
    from cotescore import spacer, cdd_decomp
    from jiwer import cer as jiwer_cer
    import plotnine as p9
    import unicodedata
    import re

    def normalize_quotes(text):
        # Single quotes / apostrophes
        text = re.sub(r'[\u2018\u2019\u201a\u201b\u2039\u203a`]', "'", text)
        # Double quotes
        text = re.sub(r'[\u201c\u201d\u201e\u201f\u00ab\u00bb]', '"', text)
        return text

    def normalize_dashes(text):
        text = re.sub(r'[\u2013\u2014\u2015\u2012]', '-', text)  # en/em dashes
        return text

    def normalize_for_cer(text):
        text = text.lower()
        text = unicodedata.normalize('NFKC', text)
        text = normalize_quotes(text)
        text = normalize_dashes(text)
        text = text.replace('\xa0', ' ')
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        return text
    return (
        Counter,
        Path,
        cdd_decomp,
        jiwer_cer,
        mo,
        normalize_for_cer,
        p9,
        pd,
        spacer,
    )


@app.cell
def _(Path):
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    OCR_DIR = _REPO_ROOT / "data/results_spiritualist/ocr"
    SSU_TEXT_PATH = _REPO_ROOT / "data/spiritualist/gt_ssu_bboxes.csv"
    return OCR_DIR, SSU_TEXT_PATH


@app.cell
def _(OCR_DIR, SSU_TEXT_PATH, normalize_for_cer, pd):
    """Load all spiritualist_gt_predictions_* parquets and join with GT text."""
    _parts = []
    for _f in sorted(OCR_DIR.glob("spiritualist_gt_predictions_*.parquet")):
        _parts.append(pd.read_parquet(_f))
    _raw = pd.concat(_parts, ignore_index=True)

    _gt = pd.read_csv(SSU_TEXT_PATH)  # page_id, ssu_id, gt_text (+ bbox cols)
    _raw["page_id"] = _raw["filename"].str.removesuffix(".jpg")

    bb_df = _raw.merge(_gt[["page_id", "ssu_id", "gt_text"]], on=["page_id", "ssu_id"], how="left")

    bb_df['gt_text'] = bb_df['gt_text'].apply(normalize_for_cer)
    bb_df['ocr_text'] = bb_df['ocr_text'].apply(normalize_for_cer)


    #bb_df['gt_text'] = bb_df['gt_text'].str.lower()
    #bb_df['ocr_text'] = bb_df['ocr_text'].str.lower()

    #bb_df['gt_text'] = bb_df['gt_text'].str.replace(r'(?<!\n)\n(?!\n)', ' ', regex=True)
    #bb_df['ocr_text'] = bb_df['ocr_text'].str.replace(r'(?<!\n)\n(?!\n)', ' ', regex=True)

    bb_df
    return (bb_df,)


@app.cell
def _(Counter, bb_df, cdd_decomp, jiwer_cer, pd, spacer):
    """Compute CER, SpACER, and CDD (d_ocr) for each bounding box row."""

    def _compute_row(ocr_text, gt_text):
        pred = ocr_text if isinstance(ocr_text, str) else ""
        gt = gt_text if isinstance(gt_text, str) else ""
        if not gt.strip():
            return None, None, None, None
        cer_val = jiwer_cer(gt, pred)
        spacer_val = spacer(Counter(gt), Counter(pred))
        cdd_val = cdd_decomp({"gt": gt, "ocr": pred}).d_ocr
        return cer_val, spacer_val, cdd_val, len(gt)

    _records = [
        _compute_row(row["ocr_text"], row.get("gt_text"))
        for _, row in bb_df.iterrows()
    ]

    metrics_df = bb_df.copy()
    metrics_df[["cer", "spacer", "cdd", "gt_len"]] = pd.DataFrame(
        _records, columns=["cer", "spacer", "cdd", "gt_len"], index=metrics_df.index
    )
    metrics_df = metrics_df.dropna(subset=["cer", "spacer", "cdd"]).reset_index(drop=True)
    metrics_df
    return (metrics_df,)


@app.cell
def _(metrics_df):
    metrics_df.loc[(metrics_df['ssu_id']=="ssu_1_col_1") & (metrics_df['page_id']=="0001_p001") & (metrics_df['ocr_model']=="easyocr")]
    return


@app.cell
def _(metrics_df):
    """Add global and per-model ranks for CER, SpACER, CDD; rank 1 = best (lowest error).

    rank_diff = cdd_rank - spacer_rank:
      positive → CDD penalises this row more than SpACER (relative to the rest)
      negative → SpACER penalises this row more than CDD
    Large absolute rank differences are candidates for hallucination or other
    systematic errors worth inspecting.
    """
    ranked_df = metrics_df.copy()

    for _col in ["cer", "spacer", "cdd"]:
        ranked_df[f"{_col}_rank_global"] = (
            ranked_df[_col].rank(method="average", ascending=True)
        )
        ranked_df[f"{_col}_rank_model"] = (
            ranked_df.groupby("ocr_model")[_col]
            .rank(method="average", ascending=True)
        )

    ranked_df["rank_diff_global"] = (
        ranked_df["cdd_rank_global"] - ranked_df["spacer_rank_global"]
    )
    ranked_df["rank_diff_model"] = (
        ranked_df["cdd_rank_model"] - ranked_df["spacer_rank_model"]
    )

    ranked_df
    return (ranked_df,)


@app.cell
def _(mo, ranked_df):
    """Summary statistics of rank differences by model."""
    _summary = (
        ranked_df.groupby("ocr_model")[["cer", "spacer", "cdd", "rank_diff_global", "rank_diff_model"]]
        .agg(["mean", "std"])
        .round(4)
    )
    mo.ui.table(_summary.reset_index())
    return


@app.cell
def _(ranked_df):
    ranked_df
    return


@app.cell
def _(ranked_df):
    ranked_df.loc[ (ranked_df['gt_len']>500)].shape[0]/ranked_df.shape[0]
    return


@app.cell
def _(p9, ranked_df):
    """SpACER vs CDD scatter, coloured by model — divergence from the diagonal signals rank disagreement."""
    (
        p9.ggplot(ranked_df.loc[ (ranked_df['gt_len']>500)], p9.aes(x="spacer", y="cdd", colour="ocr_model"))
        + p9.geom_point(alpha=0.5, size=1.5)
        #+ p9.geom_abline(slope=1, intercept=0, linetype="dashed", colour="grey")
        + p9.labs(
            title="SpACER vs CDD (d_ocr) per bounding box",
            x="SpACER",
            y="CDD",
            colour="OCR model",
        )
        + p9.theme(figure_size=(8, 6)) + p9.ylim(0,0.2)  + p9.xlim(0,0.1)
    )
    return


@app.cell
def _(ranked_df):
    """Rows with the largest positive rank difference (CDD much worse rank than SpACER).

    These are candidates where character-distribution looks relatively OK to SpACER
    but CDD picks up something different — worth checking for hallucination.

    Use .loc to filter further, e.g.:
        top_pos.loc[top_pos["ocr_model"] == "tesseract"]
    """
    top_pos = (
        ranked_df
        .sort_values("rank_diff_global", ascending=False)
        [["ocr_model", "filename", "ssu_id", "gt_text", "ocr_text",
          "cer", "spacer", "cdd",
          "cer_rank_global", "spacer_rank_global", "cdd_rank_global",
          "rank_diff_global",
          "cer_rank_model", "spacer_rank_model", "cdd_rank_model",
          "rank_diff_model"]]
        .reset_index(drop=True)
    )
    top_pos
    return


@app.cell
def _(ranked_df):
    """Rows with the largest negative rank difference (SpACER much worse rank than CDD).

    These are candidates where SpACER sees poor character-distribution but CDD is
    relatively lenient — another class of potential anomaly.
    """
    top_neg = (
        ranked_df.loc[(ranked_df['ocr_model']!='trocr') & (ranked_df['gt_len']>500) & (ranked_df['spacer']<0.25)]
        .sort_values("rank_diff_global", ascending=True)
        [["ocr_model", "filename", "ssu_id", "gt_text", "ocr_text",
          "cer", "spacer", "cdd",
          "cer_rank_global", "spacer_rank_global", "cdd_rank_global",
          "rank_diff_global",
          "cer_rank_model", "spacer_rank_model", "cdd_rank_model",
          "rank_diff_model"]]
        .reset_index(drop=True)
    )
    top_neg
    return


if __name__ == "__main__":
    app.run()

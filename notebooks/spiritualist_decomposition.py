import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    # Perform the full decomposition for the model pipelines

    This notebook
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    from collections import Counter
    from cotescore._distributions import build_R_spatial
    from cotescore import RegionChars, cdd_decomp, cdd_decomp_spatial, spacer_decomp_spatial
    from jiwer import cer as jiwer_cer
    from cotescore import spacer
    import plotnine as p9
    from scipy.stats import spearmanr

    from spacer_analysis.correlation import dpars_metric_spearman
    from spacer_analysis.cote import COTE_LABELS, compute_cote_df, mean_cote_table
    from spacer_analysis.data import load_dataset
    from spacer_analysis.detection import detection_metrics, page_detection_metrics
    from spacer_analysis.names import display_name
    from spacer_analysis.tables import bold_best_cols, bold_best_pivot, latex_table
    from spacer_analysis.text import normalize_for_cer

    return (
        COTE_LABELS,
        Counter,
        Path,
        RegionChars,
        bold_best_cols,
        bold_best_pivot,
        build_R_spatial,
        cdd_decomp,
        cdd_decomp_spatial,
        compute_cote_df,
        detection_metrics,
        display_name,
        dpars_metric_spearman,
        jiwer_cer,
        latex_table,
        load_dataset,
        mean_cote_table,
        mo,
        normalize_for_cer,
        np,
        p9,
        page_detection_metrics,
        pd,
        spacer,
        spacer_decomp_spatial,
        spearmanr,
    )


@app.cell
def _(load_dataset):
    """Inferred characters, OCR outputs and predicted boxes (see spacer_analysis.data / spacer_analysis.paths)."""
    _ds = load_dataset("spiritualist")
    chars_df = _ds.chars_df
    gt_ocr_df = _ds.gt_ocr_df
    pred_ocr_df = _ds.pred_ocr_df
    bbox_df = _ds.bbox_df
    parsing_models = _ds.parsing_models
    ocr_models = _ds.ocr_models
    pages = _ds.pages
    return (
        bbox_df,
        chars_df,
        gt_ocr_df,
        ocr_models,
        pages,
        parsing_models,
        pred_ocr_df,
    )


@app.cell
def _(
    Counter,
    Path,
    RegionChars,
    bbox_df,
    build_R_spatial,
    cdd_decomp_spatial,
    chars_df,
    gt_ocr_df,
    normalize_for_cer,
    np,
    ocr_models,
    pages,
    parsing_models,
    pd,
    pred_ocr_df,
    spacer_decomp_spatial,
):
    """Precompute CDD and SpACER decompositions for all pages and model combinations.

    Cached to disk since this loop is expensive (nested over pages × parsing
    models × OCR models). Delete the cache file to force a recompute after
    adding new OCR results or parsing models.

    Data assumptions:
      chars_df columns: char_text, cx, cy, ssu_id, page_id
      gt_ocr_df columns: ocr_model, filename, ssu_id, ocr_text
      pred_ocr_df columns: parsing_model, ocr_model, filename, x, y, width, height, ocr_text
      bbox_df columns: parsing_model, filename, x, y, width, height
    """

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _RESULTS_CACHE = _REPO_ROOT / "data/spiritualist/decomposition_results.parquet"

    if _RESULTS_CACHE.exists():
        results_df = pd.read_parquet(_RESULTS_CACHE)
        print(f"Loaded cached decomposition results: {len(results_df):,} rows from {_RESULTS_CACHE}")
    else:
        def _join_ocr_text(texts):
            return normalize_for_cer(" ".join(texts).replace(" ", ""))

        def _bbox_key(x, y, w, h):
            return (int(x), int(y), int(w), int(h))

        from tqdm import tqdm

        _records = []
        for _page in tqdm(pages, desc="pages"):
            _chars_page = chars_df[chars_df["page_id"] == _page]

            _ssu_codes, _ssu_uniques = pd.factorize(_chars_page["ssu_id"])
            _ssu_to_int = {s: i for i, s in enumerate(_ssu_uniques)}

            _gt_chars = RegionChars(
                tokens=np.array([normalize_for_cer(c) for c in _chars_page["char_text"]], dtype=object),
                xs=_chars_page["cx"].to_numpy(dtype=np.intp),
                ys=_chars_page["cy"].to_numpy(dtype=np.intp),
                region_ids=_ssu_codes.astype(np.intp),
            )

            for _pm in parsing_models:
                _bbox_page = bbox_df.loc[
                    (bbox_df["parsing_model"] == _pm) &
                    (bbox_df["filename"] == f"{_page}.jpg")
                ].reset_index(drop=True)

                _bbox_arr = _bbox_page[["x", "y", "width", "height"]].to_numpy(dtype=float)

                _bbox_key_to_id = {
                    _bbox_key(r["x"], r["y"], r["width"], r["height"]): i
                    for i, r in enumerate(_bbox_page.to_dict("records"))
                }

                for _om in ocr_models:
                    # GT OCR: {ssu_int -> ocr_text} from OCR on GT regions
                    _gt_page_ocr = gt_ocr_df.loc[
                        (gt_ocr_df["ocr_model"] == _om) &
                        (gt_ocr_df["filename"] == f"{_page}.jpg")
                    ]
                    if not _gt_page_ocr.empty:
                        _pred_gt_ocr = {
                            _ssu_to_int[row["ssu_id"]]: _join_ocr_text(row["ocr_text"].split())
                            for _, row in _gt_page_ocr.iterrows()
                            if row["ssu_id"] in _ssu_to_int
                        }
                    else:
                        _pred_gt_ocr = {}

                    # Prediction OCR: {bbox_id -> ocr_text} matched by integer coordinates.
                    # For "gt" parsing, regions are identical to the GT SSU boxes so there
                    # are no predicted-region OCR files; _pred_parse_ocr = {} correctly
                    # yields d_pars ≈ 0, d_int ≈ 0, d_total ≈ d_ocr (perfect-parsing baseline).
                    if _pm == "gt":
                        _pred_parse_ocr = {}
                    else:
                        _pred_page_ocr = pred_ocr_df.loc[
                            (pred_ocr_df["parsing_model"] == _pm) &
                            (pred_ocr_df["ocr_model"] == _om) &
                            (pred_ocr_df["filename"] == f"{_page}.jpg")
                        ]
                        if not _pred_page_ocr.empty:
                            _pred_parse_ocr = {
                                _bbox_key_to_id[_k]: _join_ocr_text(row["ocr_text"].split())
                                for _, row in _pred_page_ocr.iterrows()
                                if (_k := _bbox_key(row["x"], row["y"], row["width"], row["height"])) in _bbox_key_to_id
                            }
                        else:
                            _pred_parse_ocr = {}

                    _cdd = cdd_decomp_spatial(_gt_chars, _bbox_arr, _pred_gt_ocr, _pred_parse_ocr)
                    _sp = spacer_decomp_spatial(_gt_chars, _bbox_arr, _pred_gt_ocr, _pred_parse_ocr)

                    _Q = Counter(_gt_chars.tokens.tolist())
                    _R_agg, _ = build_R_spatial(_gt_chars, _bbox_arr)

                    _records.append({
                        "page": _page,
                        "parsing_model": _pm,
                        "ocr_model": _om,
                        "n_gt_chars": sum(_Q.values()),
                        "n_captured_chars": sum(_R_agg.values()),
                        "n_predicted_boxes": len(_bbox_arr),
                        # CDD (sqrt-JSD based)
                        "d_pars_cdd": _cdd.d_pars,
                        "d_ocr_cdd": _cdd.d_ocr,
                        "d_int_cdd": _cdd.d_int,
                        "d_total_cdd": _cdd.d_total,
                        # SpACER macro (dominant metric)
                        "d_pars_spacer_macro": _sp.d_pars_macro,
                        "d_ocr_spacer_macro": _sp.d_ocr_macro,
                        "d_int_spacer_macro": _sp.d_int_macro,
                        "d_total_spacer_macro": _sp.d_total_macro,
                        # SpACER micro (supporting metric; d_pars_micro is always None with spatial API)
                        "d_ocr_spacer_micro": _sp.d_ocr_micro,
                        "d_int_spacer_micro": _sp.d_int_micro,
                        "d_total_spacer_micro": _sp.d_total_micro,
                    })

        results_df = pd.DataFrame(_records)
        _RESULTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(_RESULTS_CACHE, index=False)
        print(f"Computed and cached decomposition results: {len(results_df):,} rows -> {_RESULTS_CACHE}")

    results_df['pars_int'] = results_df['d_int_spacer_macro'] + results_df['d_pars_spacer_macro']
    results_df['pars_int_over_ocr'] = results_df['pars_int'] / results_df['d_ocr_spacer_macro']
    results_df['total_over_two_ocr'] =  results_df['d_total_spacer_macro'] /(2*results_df['d_ocr_spacer_macro'])
    results_df['ocr_over_total'] =  results_df['d_ocr_spacer_macro'] / results_df['d_total_spacer_macro']
    return (results_df,)


@app.cell
def _(results_df):
    results_df.loc[(results_df['page']=='0001_p001') & (results_df['parsing_model']=='ppdoc_s')]
    return


@app.cell
def _(bold_best_cols, box_df, display_name, latex_table, mo, results_df):
    """d_ocr — mean per OCR model (independent of parsing model).

    d_ocr is identical across all parsing models, so we deduplicate by
    taking the first parsing model's rows per (page, ocr_model).
    """
    _first_pm = results_df["parsing_model"].iloc[0]
    _ocr_rows = results_df[results_df["parsing_model"] == _first_pm]

    _cer_by_model = (
        box_df.groupby("ocr_model")["cer"]
        .median()
        .rename("CER")
    )

    d_ocr_table = (
        _ocr_rows.groupby("ocr_model")
        .median(numeric_only=True)[["d_ocr_spacer_macro", "d_ocr_cdd"]]
        .rename(columns={
            "d_ocr_spacer_macro": "SpACER macro",
            "d_ocr_cdd": "CDD",
        })
        .join(_cer_by_model)
        .rename(index=display_name)
        .round(4)
    )
    d_ocr_table.index.name = "OCR Model"

    latex_table(
        bold_best_cols(
            d_ocr_table,
            lower_cols=["SpACER macro", "CDD", "CER"],
        ),
        caption=r"OCR error ($d_\text{ocr}$) by OCR model, averaged over pages using GT regions. "
                r"SpACER macro is the primary metric; lower is better. "
                r"CER is computed at GT bounding-box level.",
        label="tab:d_ocr",
    )

    mo.vstack([mo.md("### $d_\\text{ocr}$ — mean by OCR model"), mo.ui.table(d_ocr_table, selection=None)])
    return


@app.cell
def _(bold_best_cols, cote_table, display_name, latex_table, mo, results_df):
    """d_pars — median per parsing model (independent of OCR model).

    Also joins COTe total score.
    """
    d_pars_table = (
        results_df[results_df["parsing_model"] != "gt"]
        .groupby("parsing_model")
        .median(numeric_only=True)[["d_pars_spacer_macro", "d_pars_cdd"]]
        .rename(columns={
            "d_pars_spacer_macro": "SpACER macro",
            "d_pars_cdd": "CDD",
        })
        # Join on raw parsing_model names before renaming index
        .join(cote_table[["cote"]].rename(columns={"cote": "COTe"}))
        .rename(index=display_name)
        .round(4)
    )
    d_pars_table.index.name = "Parsing Model"

    latex_table(
        bold_best_cols(
            d_pars_table,
            lower_cols=["SpACER macro", "CDD"],
            higher_cols=["COTe"],
        ),
        caption=r"Parsing error ($d_\text{pars}$) by parsing model with COTe. "
                r"SpACER macro is the primary metric; lower is better for SpACER/CDD, higher for COTe.",
        label="tab:d_pars",
    )

    mo.vstack([mo.md("### $d_\\text{pars}$ — median by parsing model"), mo.ui.table(d_pars_table, selection=None)])
    return


@app.cell
def _(bold_best_pivot, display_name, latex_table, mo, results_df):
    """d_int — median grouped by (parsing_model × ocr_model)."""
    _df = results_df[results_df["parsing_model"] != "gt"]

    def _pivot(col, agg="median"):
        return (
            _df.groupby(["parsing_model", "ocr_model"])[col]
            .agg(agg)
            .unstack("ocr_model")
            .rename(index=display_name, columns=display_name)
            .round(4)
        )

    _d_int_spacer_macro = _pivot("d_int_spacer_macro")
    _d_int_spacer_micro = _pivot("d_int_spacer_micro")
    _d_int_cdd = _pivot("d_int_cdd")
    _d_int_spacer_macro.index.name = "Parsing Model"

    latex_table(
        bold_best_pivot(_d_int_spacer_macro, lower_is_better=True),
        caption=r"Interaction error ($d_\text{int}$, SpACER macro) by parsing model (rows) "
                r"and OCR model (columns). \textbf{Bold}: column best; \textbf{bold}$^*$: overall best.",
        label="tab:d_int",
    )

    mo.vstack([
        mo.md(r"### $d_\text{int}$ — median by parsing × OCR model"),
        mo.md("**SpACER macro**"), mo.ui.table(_d_int_spacer_macro.reset_index(), selection=None),
        mo.md("**SpACER micro**"), mo.ui.table(_d_int_spacer_micro.reset_index(), selection=None),
        mo.md("**CDD**"), mo.ui.table(_d_int_cdd.reset_index(), selection=None),
    ])
    return


@app.cell
def _(bold_best_pivot, display_name, latex_table, mo, results_df):
    """d_total — mean grouped by (parsing_model × ocr_model)."""
    _df = results_df[results_df["parsing_model"] != "gt"]

    def _pivot(col, agg="median"):
        return (
            _df.groupby(["parsing_model", "ocr_model"])[col]
            .agg(agg)
            .unstack("ocr_model")
            .rename(index=display_name, columns=display_name)
            .round(4)
        )

    _d_total_spacer_macro = _pivot("d_total_spacer_macro")
    _d_total_spacer_micro = _pivot("d_total_spacer_micro")
    _d_total_cdd = _pivot("d_total_cdd")
    _d_total_spacer_macro.index.name = "Parsing Model"

    latex_table(
        bold_best_pivot(_d_total_spacer_macro, lower_is_better=True),
        caption=r"Total error ($d_\text{total}$, SpACER macro) by parsing model (rows) "
                r"and OCR model (columns). \textbf{Bold}: column best; \textbf{bold}$^*$: overall best.",
        label="tab:d_total",
    )

    mo.vstack([
        mo.md(r"### $d_\text{total}$ — mean by parsing × OCR model"),
        mo.md("**SpACER macro**"), mo.ui.table(_d_total_spacer_macro.reset_index(), selection=None),
        mo.md("**SpACER micro**"), mo.ui.table(_d_total_spacer_micro.reset_index(), selection=None),
        mo.md("**CDD**"), mo.ui.table(_d_total_cdd.reset_index(), selection=None),
    ])
    return


@app.cell
def _(mo, ocr_models, parsing_models):
    """Dropdown selector for the (parsing_model × ocr_model) pair used in the scatter plot below."""
    _pairs = [f"{pm} × {om}" for pm in parsing_models for om in ocr_models]
    model_pair_dropdown = mo.ui.dropdown(_pairs, value=_pairs[0], label="Parsing × OCR model")
    model_pair_dropdown
    return


@app.cell
def _(mo, ocr_models, parsing_models, pd, results_df):
    """P/R/F1: rule `d_total < 2*d_ocr` as predictor of `d_pars < d_ocr` (CDD), per model pair."""
    _records = []
    for _pm in parsing_models:
        for _om in ocr_models:
            _df = results_df.loc[
                (results_df["parsing_model"] == _pm) & (results_df["ocr_model"] == _om)
            ]
            if _df.empty:
                continue
            _BAND = 0.05
            _threshold_pred = 2 * _df["d_ocr_cdd"]
            _pred_similar = (
                (_df["d_total_cdd"] >= _threshold_pred * (1 - _BAND)) &
                (_df["d_total_cdd"] <= _threshold_pred * (1 + _BAND))
            )
            _actual_similar = (
                ((_df["d_pars_cdd"] + _df["d_int_cdd"]) >= _df["d_ocr_cdd"] * (1 - _BAND)) &
                ((_df["d_pars_cdd"] + _df["d_int_cdd"]) <= _df["d_ocr_cdd"] * (1 + _BAND))
            )
            _df_excl = _df[~_pred_similar & ~_actual_similar]
            _n_similar = int((_pred_similar | _actual_similar).sum())
            _pred = _df_excl["d_total_cdd"] < 2 * _df_excl["d_ocr_cdd"]
            _actual = (_df_excl["d_pars_cdd"] + _df_excl["d_int_cdd"]) < _df_excl["d_ocr_cdd"]
            _tp = int((_pred & _actual).sum())
            _fp = int((_pred & ~_actual).sum())
            _fn = int((~_pred & _actual).sum())
            _tn = int((~_pred & ~_actual).sum())
            _n = _tp + _fp + _fn + _tn
            _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0
            _rec = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0
            _f1 = 2 * _prec * _rec / (_prec + _rec) if (_prec + _rec) > 0 else float("nan")
            _acc = (_tp + _tn) / _n if _n > 0 else float("nan")
            _records.append({
                "parsing_model": _pm,
                "ocr_model": _om,
                "n_true": _tp + _fp,
                "n_similar": _n_similar,
                "n_false": _fn + _tn,
                "precision": round(_prec, 3),
                "recall": round(_rec, 3),
                "F1": round(_f1, 3),
                "accuracy": round(_acc, 3),
            })

    prf_table = pd.DataFrame(_records)
    mo.vstack([
        mo.md("### P/R/F1: rule `d_total < 2·d_ocr` predicts `d_pars < d_ocr` (CDD)"),
        mo.ui.table(prf_table, selection=None),
    ])
    return


@app.cell
def _(bbox_df, detection_metrics, mo, page_detection_metrics, parsing_models):
    """Single-class detection metrics vs the GT SSU boxes: COCO mAP, mAP@0.5,
    F1@0.5 and mean matched IoU (see spacer_analysis.detection). page_det_df
    holds the per-page F1 / IoU used for the correlation tables."""
    page_det_df = page_detection_metrics("spiritualist", bbox_df, parsing_models)
    det_df = detection_metrics("spiritualist", bbox_df, parsing_models, per_page=page_det_df)
    mo.vstack([
        mo.md("### Detection metrics (mAP, F1, IoU) by parsing model"),
        mo.ui.table(det_df, selection=None),
    ])
    return det_df, page_det_df


@app.cell
def _(
    COTE_LABELS,
    bbox_df,
    bold_best_cols,
    compute_cote_df,
    det_df,
    display_name,
    latex_table,
    mean_cote_table,
    mo,
    parsing_models,
):
    """COTe score — Coverage, Overlap, Trespass, Excess per parsing model.

    Per-page scores come from spacer_analysis.cote.compute_cote_df (cotescore's exact
    bounding-box path, cached to the dataset's cote_score_cache.parquet so the
    cross-dataset notebooks see the same numbers).
    """
    cote_df = compute_cote_df("spiritualist", bbox_df, parsing_models)
    # Keep raw index so d_pars cell can join on parsing_model names directly.
    cote_table = mean_cote_table(cote_df)

    _cote_display = (
        cote_table.join(det_df.drop(columns="mAP@0.5")).rename(index=display_name).rename(columns=COTE_LABELS)
    )
    latex_table(
        bold_best_cols(
            _cote_display,
            higher_cols=["COTe", "Coverage", "mAP", "mAP@0.5", "F1@0.5", "IoU"],
            lower_cols=["Overlap", "Trespass", "Excess"],
        ),
        caption=r"COTe score and components by parsing model, with single-class box-detection "
                r"metrics against the GT SSUs: COCO mAP (IoU 0.50:0.05:0.95), mAP@0.5, "
                r"page-macro F1@0.5, and mean IoU of matched boxes. "
                r"Higher is better except for Overlap, Trespass and Excess.",
        label="tab:cote",
    )

    mo.vstack([
        mo.md("### COTe score — mean by parsing model"),
        mo.ui.table(_cote_display, selection=None),
    ])
    return cote_df, cote_table


@app.cell
def _(cote_df, mo, p9, results_df):
    """Scatter plot: d_pars vs COTe score per page, faceted by parsing model."""

    _dpars = (
        results_df.groupby(["page", "parsing_model"])[["d_pars_cdd"]]
        .mean()
        .reset_index()
    )
    plot_df = _dpars.merge(
        cote_df[["page", "parsing_model", "cote"]],
        on=["page", "parsing_model"],
    )

    _plt = (
        p9.ggplot(plot_df, p9.aes(x="cote", y="d_pars_cdd"))
        + p9.geom_point(alpha=0.7, size=2)
        + p9.geom_smooth(method="lm", se=False, color="firebrick", size=0.8)
        + p9.facet_wrap("~parsing_model", nrow=1)
        + p9.labs(
            title="d_pars vs COTe score — one point per page",
            x="COTe score",
            y="d_pars (CDD)",
        )
        + p9.theme(figure_size=(12, 4)) + p9.ylim(0, 0.04)
    )

    mo.plain(_plt)
    return


@app.cell
def _(
    bold_best_cols,
    cote_df,
    display_name,
    dpars_metric_spearman,
    latex_table,
    mo,
    page_det_df,
    results_df,
):
    """Spearman correlation between per-page d_pars (SpACER, then CDD) and each
    per-page layout metric: COTe, F1@0.5, IoU. mAP is a corpus-level ranking
    metric with no per-page value, so it is not included (see
    spacer_analysis.correlation)."""
    _corr = dpars_metric_spearman(results_df, cote_df, page_det_df)
    _tables = {}
    for _err, _err_label in [("spacer", "SpACER"), ("cdd", "CDD")]:
        _tbl = _corr[_err].rename(index=display_name)
        _tables[_err_label] = _tbl
        latex_table(
            bold_best_cols(_tbl, lower_cols=list(_tbl.columns)),
            caption=rf"The Spiritualist spearman correlation ($\rho$) between per-page $d_\text{{pars}}$ ({_err_label}) "
                    r"and per-page layout metrics (COTe, F1@0.5, mean matched IoU) by parsing model. "
                    r"Negative $\rho$ indicates that a better layout score corresponds to lower "
                    r"parsing error, as expected. ``--'': undefined (metric constant across pages).",
            label=f"tab:dpars_metrics_spearman_{_err}",
        )

    mo.vstack([
        mo.md("### Spearman $\\rho$: $d_\\text{pars}$ vs COTe / F1@0.5 / IoU"),
        *[mo.vstack([mo.md(f"**{_k}**"), mo.ui.table(_v, selection=None)]) for _k, _v in _tables.items()],
    ])
    return


@app.cell
def _(
    Counter,
    Path,
    cdd_decomp,
    chars_df,
    gt_ocr_df,
    jiwer_cer,
    mo,
    normalize_for_cer,
    p9,
    pd,
    spacer,
):
    """Per-box CER vs d_ocr SpACER/CDD — merge-based, no loops.

    Cached to disk: one row per (page, ssu_id, ocr_model), with a row-wise
    apply of jiwer_cer/spacer/cdd_decomp per OCR-model group — the combined
    cross-dataset validation notebook reads this cache directly rather than
    recomputing it. Delete the cache file to force a recompute after adding
    new OCR results.
    """
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _BOX_CACHE = _REPO_ROOT / "data/spiritualist/box_level_ocr_comparison.parquet"

    if _BOX_CACHE.exists():
        box_df = pd.read_parquet(_BOX_CACHE)
        print(f"Loaded cached box-level comparison: {len(box_df):,} rows from {_BOX_CACHE}")
    else:
        # GT text per SSU box: concatenate char_text within each (page_id, ssu_id)
        _gt_text_df = (
            chars_df.groupby(["page_id", "ssu_id"])["char_text"]
            .apply("".join)
            .reset_index()
            .rename(columns={"char_text": "gt_text", "page_id": "page"})
        )
        _gt_text_df = _gt_text_df[_gt_text_df["gt_text"] != ""]

        # Extract page_id from filename and clean OCR text
        _ocr = gt_ocr_df.copy()
        _ocr["page"] = _ocr["filename"].str.removesuffix(".jpg")
        _ocr["ocr_text"] = _ocr["ocr_text"].str.split().str.join("")

        # Merge GT text with OCR text on (page, ssu_id)
        box_df = _gt_text_df.merge(
            _ocr[["page", "ssu_id", "ocr_model", "ocr_text"]],
            on=["page", "ssu_id"],
        )
        box_df["gt_len"] = box_df["gt_text"].str.len()

        # Compute per-box metrics with apply (jiwer_cer list form returns aggregate, not per-row)
        box_df["cer"] = box_df.apply(
            lambda r: jiwer_cer(normalize_for_cer(r["gt_text"]), normalize_for_cer(r["ocr_text"])), axis=1
        )
        box_df["d_ocr_spacer"] = box_df.apply(
            lambda r: spacer(Counter(normalize_for_cer(r["gt_text"])), Counter(normalize_for_cer(r["ocr_text"]))), axis=1
        )
        box_df["d_ocr_cdd"] = box_df.apply(
            lambda r: cdd_decomp({"gt": normalize_for_cer(r["gt_text"]), "ocr": normalize_for_cer(r["ocr_text"])}).d_ocr,
            axis=1,
        )

        _BOX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        box_df.to_parquet(_BOX_CACHE, index=False)
        print(f"Computed and cached box-level comparison: {len(box_df):,} rows -> {_BOX_CACHE}")

    _plt2 = (
        p9.ggplot(box_df, p9.aes(x="cer", y="d_ocr_spacer"))
        + p9.geom_point(alpha=0.2, size=1)
        + p9.geom_smooth(method="lm", se=False, color="firebrick", size=0.8)
        + p9.facet_wrap("~ocr_model", nrow=2)
        + p9.labs(
            title="Relationship between SpACER and CER given Ground Truth regions",
            x="CER",
            y="d_ocr SpACER",
        )
        + p9.theme(figure_size=(12, 4)) + p9.xlim(0, 1) + p9.ylim(0, 1)
    )

    mo.plain(_plt2)
    return (box_df,)


@app.cell
def _(box_df):
    (box_df["cer"] < 1).sum() / box_df.shape[0]
    return


@app.cell
def _(bold_best_cols, box_df, display_name, latex_table, mo, pd, spearmanr):
    """Spearman correlation between CER and per-box d_ocr metrics."""


    _records = []
    for _om, _grp in box_df.groupby("ocr_model"):
        _r_sp, _p_sp = spearmanr(_grp["cer"], _grp["d_ocr_spacer"])
        _r_cdd, _p_cdd = spearmanr(_grp["cer"], _grp["d_ocr_cdd"])
        _records.append({
            "ocr_model": _om,
            "SpACER $\\rho$": round(_r_sp, 3),
            "CDD $\\rho$": round(_r_cdd, 3),
        })

    _corr_df = (
        pd.DataFrame(_records)
        .set_index("ocr_model")
        .rename(index=display_name)
    )
    _corr_df.index.name = "OCR Model"

    latex_table(
        bold_best_cols(_corr_df, higher_cols=["SpACER $\\rho$", "CDD $\\rho$"]),
        caption=r"Spearman correlation ($\rho$) between per-box CER and $d_\text{ocr}$ "
                r"for SpACER (primary) and CDD, computed over GT regions. "
                r"All correlations significant at $p < 0.001$.",
        label="tab:cer_spearman",
    )

    _lines = ["**Spearman correlation: CER vs per-box d_ocr metrics**\n"]
    for _om, _grp in box_df.groupby("ocr_model"):
        _r_sp, _p_sp = spearmanr(_grp["cer"], _grp["d_ocr_spacer"])
        _r_cdd, _p_cdd = spearmanr(_grp["cer"], _grp["d_ocr_cdd"])
        _lines.append(
            f"- **{_om}**: SpACER ρ = {_r_sp:.3f} (p = {_p_sp:.3f}), "
            f"CDD ρ = {_r_cdd:.3f} (p = {_p_cdd:.3f})"
        )

    mo.vstack([
        mo.md("\n".join(_lines)),
        mo.ui.table(_corr_df, selection=None),
    ])
    return


@app.cell
def _(results_df):
    results_df['parsing_model'].unique()
    return


@app.cell
def _(mo, np, p9, results_df):
    _temp = results_df[~results_df["parsing_model"].isin(["gt", "ppdoc_s", "ppdoc_m"])].copy()

    _temp['truth_triage_pars'] = np.where(_temp['d_pars_spacer_macro']> _temp['d_ocr_spacer_macro'], 'pars', 'ocr')

    #Add in that when it is ppdoc-s or m automatically pars
    _temp['truth_triage_pars_force'] = np.where(_temp['parsing_model'].isin(['ppdoc_m', 'ppdoc_s']), 'pars', _temp['truth_triage_pars']  )


    mod_df = _temp

    _plt =p9.ggplot(_temp, p9.aes(x = 'ocr_over_total', y ='d_pars_spacer_macro', colour = 'truth_triage_pars')) + p9.geom_point( ) +p9.xlim(0,5) +p9.ylim(0,0.1)

    mo.plain(_plt)
    return (mod_df,)


@app.cell
def _():
    return


@app.cell
def _(mod_df):
    mod_df.groupby('truth_triage_pars').size()
    return


@app.cell
def _(mod_df, np, pd):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.metrics import roc_auc_score


    _temp = mod_df.copy()
    _features = [ 'ocr_over_total']
    _target = 'truth_triage_pars'

    # Drop NaNs
    _clean_df = _temp.dropna(subset=_features + [_target])

    _X = _clean_df[_features]
    _y = _clean_df[_target]

    # 1. Stratified Split: ensures 'ocr' is represented in both train and test
    _X_train, _X_test, _y_train, _y_test = train_test_split(
        _X, _y, test_size=0.2, #random_state=42, 
        stratify=_y
    )

    _log_reg = LogisticRegression(class_weight='balanced')
    _log_reg.fit(_X_train, _y_train)


    # Predict
    _y_pred = _log_reg.predict(_X_test)

    # Performance
    print(f"Number OCR dominant {((_y_train!='pars')).sum()}: Number of parsing dominant {((_y_train=='pars')).sum()}")
    print("--- Stratified & Balanced Results ---")
    print("Accuracy:", accuracy_score(_y_test, _y_pred))
    print("\nClassification Report:")
    print(classification_report(_y_test, _y_pred))

    print("\nConfusion Matrix:")
    _cm = confusion_matrix(_y_test, _y_pred)
    print(_cm)

    print("\nAdjusted Coefficients:")
    for _name, _coef in zip(_X.columns, _log_reg.coef_[0]):
        print(f"  {_name}: {_coef:.4f}")

    # 1. Extract the intercept and coefficient
    _intercept = _log_reg.intercept_[0]
    _coef = _log_reg.coef_[0][0]

    print(f"--- Logic for total_over_two_ocr ---")
    print(f"Intercept: {_intercept:.4f}")
    print(f"Coefficient: {_coef:.4f}")

    # You MUST use probabilities for AUC, not class predictions
    _y_probs = _log_reg.predict_proba(_X_test)[:, 1] 

    roc_auc = roc_auc_score(_y_test, _y_probs)
    print(f"ROC-AUC Score: {roc_auc:.4f}")



    import matplotlib.pyplot as plt
    from sklearn.utils import resample

    n_iterations = 1000
    stats = []

    for i in range(n_iterations):
        # 1. Resample the data
        X_res, y_res = resample(_X, _y, stratify=_y)

        # 2. Fit the model
        boot_model = LogisticRegression(class_weight='balanced').fit(X_res, y_res)

        # 3. Get probabilities for the "pars" class
        # Note: ensure [:, 1] corresponds to the correct class index
        y_probs = boot_model.predict_proba(X_res)[:, 1]

        # 4. Calculate AUC for this bootstrap sample
        current_auc = roc_auc_score(y_res, y_probs)

        stats.append({
            'intercept': boot_model.intercept_[0],
            'coefficient': boot_model.coef_[0][0],
            'auc': current_auc,
        })

    boot_results = pd.DataFrame(stats)

    # Calculate 95% Confidence Intervals
    ci_intercept = np.percentile(boot_results['intercept'], [2.5, 97.5])
    ci_coef = np.percentile(boot_results['coefficient'], [2.5, 97.5])

    print(f"Intercept 95% CI: {ci_intercept}")
    print(f"Coefficient 95% CI: {ci_coef}")



    ci_auc = np.percentile(boot_results['auc'], [2.5, 97.5])

    print(f"Mean Bootstrap AUC: {boot_results['auc'].mean():.4f}")
    print(f"AUC 95% CI: {ci_auc}")
    return ci_coef, ci_intercept


@app.cell
def _(ci_coef, ci_intercept, np):
    print(f"Intercept 95% CI: {np.exp(ci_intercept)}")
    print(f"Coefficient 95% CI: {np.exp(ci_coef)}")
    return


@app.cell
def _(mo, np, p9, pd, results_df):
    _temp = results_df[~results_df["parsing_model"].isin(["gt", "ppdoc_s", "ppdoc_m"])].copy()

    _temp['pars_int_cdd'] = _temp['d_int_cdd'] + _temp['d_pars_cdd']
    _temp['pars_int_overesults_df[~results_df["parsing_model"].isin(["gt", "ppdoc_s", "ppdoc_m"])]r_ocr_cdd'] = _temp['pars_int_cdd'] / _temp['d_ocr_cdd']
    _temp['total_over_two_ocr_cdd'] = _temp['d_total_cdd'] /(2*_temp['d_ocr_cdd'] )

    _temp['truth_triage_pars_int_cdd'] = np.where(_temp['pars_int_cdd']> _temp['d_ocr_cdd'], 'pars int_cdd', 'ocr')
    _temp['truth_triage_pars_cdd'] = np.where(_temp['d_pars_cdd']> _temp['d_ocr_cdd'], 'pars', 'ocr')
    _temp['pred_two_cer_cdd'] = np.where(_temp['total_over_two_ocr_cdd']>1, 'pred pars', 'pred ocr')


    print(pd.crosstab(_temp['truth_triage_pars_cdd'] , _temp['pred_two_cer_cdd']))


    _plt =p9.ggplot(_temp, p9.aes(x = 'd_ocr_cdd', y = 'd_pars_cdd', colour = 'pred_two_cer_cdd', shape = 'truth_triage_pars_cdd')) + p9.geom_point( ) +p9.xlim(0,0.1) +p9.ylim(0,0.1)

    mo.plain(_plt)
    return


@app.cell
def _(results_df):
    results_df[~results_df["parsing_model"].isin(["gt"])].shape
    return


@app.cell
def _(cote_df, mo, np, p9, pd, results_df):
    """Heatmap: F1 of (COTe >= x_thresh AND d_ocr/d_total >= y_thresh) predicting d_ocr >= d_pars.

    Both features are observable without character-level positions:
      COTe          — geometric parsing quality vs GT boxes (higher = better geometry)
      d_ocr/d_total — share of the total pipeline error attributable to the OCR step alone;
                      high → OCR is the bottleneck, low → imperfect parsing degrades the result further

    Positive class: d_ocr >= d_pars (SpACER macro).
    Classifier rule: COTe >= x_thresh AND d_ocr/d_total >= y_thresh.
    F1 = 2*TP / (2*TP + FP + FN) evaluated over the full (non-gt) dataset.
    """
    _x_thresh = np.linspace(0.0, 0.9, 10)   # COTe
    _y_thresh = np.linspace(0.0, 0.9, 10)   # d_ocr / d_total
    _x_step = float(_x_thresh[1] - _x_thresh[0])
    _y_step = float(_y_thresh[1] - _y_thresh[0])

    _df = results_df[~results_df["parsing_model"].isin(["gt"])].merge(cote_df[["page", "parsing_model", "cote"]], on=["page", "parsing_model"], how="left")

    _df['ocr_over_total'] = _df["d_ocr_spacer_macro"] / _df["d_total_spacer_macro"]
    _df['ocr_dominant'] = _df["d_ocr_spacer_macro"] >= _df["d_pars_spacer_macro"]



    _df.loc[_df["parsing_model"].isin(["ppdoc_s", "ppdoc_m"]), "ocr_dominant"] = False

    _x_vals = _df["cote"].to_numpy()
    _y_vals = _df["ocr_over_total"].to_numpy()
    _label = _df["ocr_dominant"].to_numpy()

    mod_prep = _df

    def _f1_for_thresh(cote_thresh, ocr_over_total_thresh):
        _pred = (_x_vals >= cote_thresh) & (_y_vals >= ocr_over_total_thresh)
        _tp = (_pred & _label).sum()
        _fp = (_pred & ~_label).sum()
        _fn = (~_pred & _label).sum()
        _denom = 2 * _tp + _fp + _fn
        return pd.Series({
            "n": int(_pred.sum()),
            "f1": float(2 * _tp / _denom) if _denom > 0 else np.nan,
        })

    _thresh_df = pd.DataFrame(
        [(xt, yt) for xt in _x_thresh for yt in _y_thresh],
        columns=["cote_thresh", "ocr_over_total_thresh"],
    )
    _heat_df = _thresh_df.join(
        _thresh_df.apply(lambda r: _f1_for_thresh(r["cote_thresh"], r["ocr_over_total_thresh"]), axis=1)
    )
    _heat_df["label"] = (
        _heat_df["f1"].map(lambda v: f"{v:.2f}")).where(_heat_df["f1"].notna(), "")

    test_mod = _heat_df

    _plt = (
        p9.ggplot(_heat_df, p9.aes(x="cote_thresh", y="ocr_over_total_thresh", fill="f1"))
        + p9.geom_tile(width=_x_step, height=_y_step)
        + p9.geom_text(p9.aes(label="label"), size=7)
        + p9.scale_x_continuous(breaks=_x_thresh, labels=[f"{v:.2f}" for v in _x_thresh])
        + p9.scale_y_continuous(breaks=_y_thresh, labels=[f"{v:.2f}" for v in _y_thresh])
        + p9.labs(
            x="COTe score minimum threshold",
            y="SpACER d_ocr / d_total minimum threshold",
            fill="F1",
            title="Is OCR dominant error source? F1 of SpACER d_ocr/d_total and COTe thresholds", 
        )
        + p9.theme(figure_size=(8, 5), 
                  axis_text_y=p9.element_text(size=12),
                  axis_text_x=p9.element_text(rotation=45, hjust=1, size=12),
                  strip_text=p9.element_text(size=14, weight='bold'))
    )
    _plt.save(filename='data/figures/cutoff_thresholds.pdf', dpi = 300)
    mo.plain(_plt)
    return (mod_prep,)


@app.cell
def _(mod_prep):
    mod_prep
    return


@app.cell
def _(cote_df, mo, np, p9, pd, results_df):
    """Marginal F1 per axis: single-feature threshold classifiers for d_ocr >= d_pars."""

    _df = results_df[~results_df["parsing_model"].isin(["gt", "ppdoc_s", "ppdoc_m"])].merge(cote_df[["page", "parsing_model", "cote"]], on=["page", "parsing_model"], how="left")

    _df['ocr_over_total'] = _df["d_ocr_spacer_macro"] / _df["d_total_spacer_macro"]
    _df['ocr_dominant'] = _df["d_ocr_spacer_macro"] >= _df["d_pars_spacer_macro"]

    _df.loc[_df["parsing_model"].isin(["ppdoc_s", "ppdoc_m"]), "ocr_dominant"] = False

    _x_vals = _df["cote"].to_numpy()
    _y_vals = _df["ocr_over_total"].to_numpy()
    _label = _df["ocr_dominant"].to_numpy()

    _x_thresh = np.linspace(-5, 1, 10)
    _y_thresh = np.linspace(0.0, 1.5, 10)

    def _f1(pred):
        _tp = (pred & _label).sum()
        _fp = (pred & ~_label).sum()
        _fn = (~pred & _label).sum()
        _denom = 2 * _tp + _fp + _fn
        return float(2 * _tp / _denom) if _denom > 0 else np.nan

    marginal_df = pd.concat([
        pd.DataFrame({
            "threshold": _x_thresh,
            "f1": [_f1(_x_vals >= t) for t in _x_thresh],
            "n": [int((_x_vals >= t).sum()) for t in _x_thresh],
            "feature": "COTe",
        }),
        pd.DataFrame({
            "threshold": _y_thresh,
            "f1": [_f1(_y_vals >= t) for t in _y_thresh],
            "n": [int((_y_vals >= t).sum()) for t in _y_thresh],
            "feature": "d_ocr / d_total",
        }),
    ], ignore_index=True)

    _plt = (
        p9.ggplot(marginal_df, p9.aes(x="threshold", y="f1"))
        + p9.geom_line()
        + p9.geom_point()
        + p9.facet_wrap("~feature", scales="free_x", nrow=1)
        + p9.labs(
            x="Minimum threshold (≥)",
            y="F1",

            title="Marginal F1 per axis — single-feature threshold classifier for d_ocr ≥ d_pars",
        )
        + p9.theme(figure_size=(12, 5))
    )

    mo.plain(_plt)
    return (marginal_df,)


@app.cell
def _(marginal_df):
    marginal_df
    return


@app.cell
def _(cote_df, results_df):
    results_df[~results_df["parsing_model"].isin(["gt", "ppdoc_s", "ppdoc_m"])].merge(cote_df[["page", "parsing_model", "cote"]], on=["page", "parsing_model"], how="left")
    return


if __name__ == "__main__":
    app.run()

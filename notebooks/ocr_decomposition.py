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
    import sys
    from pathlib import Path

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_REPO_ROOT / "cotescore/src"))

    import marimo as mo
    import numpy as np
    import pandas as pd
    from collections import Counter
    from cotescore._distributions import build_R_spatial
    from cotescore import RegionChars, cdd_decomp, cdd_decomp_spatial, spacer_decomp_spatial, cote_score

    from cotescore.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks, eval_shape
    from jiwer import cer as jiwer_cer
    from cotescore import spacer
    import plotnine as p9
    import re
    import unicodedata

    return (
        Counter,
        Path,
        RegionChars,
        boxes_to_gt_ssu_map,
        boxes_to_pred_masks,
        build_R_spatial,
        cdd_decomp,
        cdd_decomp_spatial,
        cote_score,
        eval_shape,
        jiwer_cer,
        mo,
        np,
        p9,
        pd,
        re,
        spacer,
        spacer_decomp_spatial,
        unicodedata,
    )


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
def _(Path, pd):
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _CHARS_PATH = _REPO_ROOT / "data/spiritualist/characters_inferred.parquet"
    _OCR_DIR = _REPO_ROOT / "data/results_spiritualist/ocr"
    _BBOX_DIR = _REPO_ROOT / "data/results_spiritualist/bboxes"

    chars_df = pd.read_parquet(_CHARS_PATH)
    chars_df = chars_df[chars_df["char_text"] != " "].reset_index(drop=True)
    chars_df["cx"] = (chars_df["x"] + chars_df["w"] / 2).astype(int)
    chars_df["cy"] = (chars_df["y"] + chars_df["h"] / 2).astype(int)

    # Load GT OCR parquets (parsing_model == "gt") into a single DataFrame.
    # Filename pattern: spiritualist_gt_predictions_{ocr_model}_ocr.parquet
    # Schema: filename, ssu_id, ocr_text
    _gt_parts = []
    for _f in sorted(_OCR_DIR.glob("spiritualist_gt_predictions_*_ocr.parquet")):
        _om = _f.stem.removeprefix("spiritualist_gt_predictions_").removesuffix("_ocr")
        _part = pd.read_parquet(_f)
        _part["ocr_model"] = _om
        _gt_parts.append(_part)
    gt_ocr_df = pd.concat(_gt_parts, ignore_index=True) if _gt_parts else pd.DataFrame()

    # Load prediction OCR parquets (non-GT parsing models) into a single DataFrame.
    # Filename pattern: spiritualist_{parsing_model}_predictions_{ocr_model}_ocr.parquet
    # Schema: filename, x, y, width, height, ocr_text
    _pred_parts = []
    for _f in sorted(_OCR_DIR.glob("*_ocr.parquet")):
        _inner = _f.stem.removeprefix("spiritualist_").removesuffix("_ocr")
        _sep = _inner.index("_predictions_")
        _pm = _inner[:_sep]
        if _pm == "gt":
            continue
        _om = _inner[_sep + len("_predictions_"):]
        _part = pd.read_parquet(_f)
        _part["parsing_model"] = _pm
        _part["ocr_model"] = _om
        _pred_parts.append(_part)
    pred_ocr_df = pd.concat(_pred_parts, ignore_index=True) if _pred_parts else pd.DataFrame()

    # Load all bbox CSVs into a single DataFrame.
    # Filename pattern: spiritualist_{parsing_model}_predictions.csv
    # Schema: filename, x, y, width, height
    _bbox_parts = []
    for _f in sorted(_BBOX_DIR.glob("*.csv")):
        _pm = _f.stem.removeprefix("spiritualist_").removesuffix("_predictions")
        _part = pd.read_csv(_f)
        _part["parsing_model"] = _pm
        _bbox_parts.append(_part)
    bbox_df = pd.concat(_bbox_parts, ignore_index=True)

    parsing_models = sorted(bbox_df["parsing_model"].unique())
    ocr_models = sorted(pred_ocr_df["ocr_model"].unique()) if not pred_ocr_df.empty else []
    pages = sorted(chars_df["page_id"].unique())

    # Model display names: lowercase + strip underscores → display label.
    _MODEL_DISPLAY_NAMES = {
        # OCR models
        "trocr":      "TrOCR",
        "paddleocr":  "PaddleOCR",
        "paddleocr":  "PaddleOCR",
        "tesseract":  "Tesseract",
        "craft":      "CRAFT",
        # Parsing models
        "heron":      "Heron",
        "ppdocl":     "PPDoc-L",
        "ppdocm":     "PPDoc-M",
        "ppdocs":     "PPDoc-S",
        "yolo":       "YOLO",
    }

    def display_name(name: str) -> str:
        lower = name.lower().replace("_", "")
        return _MODEL_DISPLAY_NAMES.get(lower, name.replace("_", "-").title())

    return (
        bbox_df,
        chars_df,
        display_name,
        gt_ocr_df,
        ocr_models,
        pages,
        parsing_models,
        pred_ocr_df,
    )


@app.cell
def _(
    Counter,
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

    Data assumptions:
      chars_df columns: char_text, cx, cy, ssu_id, page_id
      gt_ocr_df columns: ocr_model, filename, ssu_id, ocr_text
      pred_ocr_df columns: parsing_model, ocr_model, filename, x, y, width, height, ocr_text
      bbox_df columns: parsing_model, filename, x, y, width, height
    """

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
def _():
    """LaTeX formatting helpers for ML-paper tables."""

    def bold_best_cols(df, lower_cols=None, higher_cols=None):
        """Bold the best value per column. Returns a string-valued DataFrame for escape=False output.

        lower_cols: column names where lower is better.
        higher_cols: column names where higher is better.
        """
        lower_cols = lower_cols or []
        higher_cols = higher_cols or []
        result = df.copy().astype(object)
        for col in df.columns:
            best = df[col].min() if col in lower_cols else df[col].max()
            for idx in df.index:
                val = df.loc[idx, col]
                s = f"{val:.3f}"
                result.loc[idx, col] = f"\\textbf{{{s}}}" if val == best else s
        return result

    def bold_best_pivot(df, lower_is_better=True):
        """Bold column-best values; bold + $^*$ for the overall table best.

        Intended for pivot tables (parsing_model rows × ocr_model columns).
        """
        fn = "min" if lower_is_better else "max"
        col_best = getattr(df, fn)(axis=0)
        table_best = float(getattr(df.values, fn)())
        result = df.copy().astype(object)
        for col in df.columns:
            for idx in df.index:
                val = df.loc[idx, col]
                s = f"{val:.3f}"
                if val == table_best:
                    s = f"\\textbf{{{s}}}$^{{*}}$"
                elif val == col_best[col]:
                    s = f"\\textbf{{{s}}}"
                result.loc[idx, col] = s
        return result

    def latex_table(df, caption, label, col_fmt=None):
        """Print a booktabs LaTeX table (escape=False, position=t)."""
        kwargs = dict(
            caption=caption,
            label=label,
            escape=False,
            position="t",
            float_format="%.3f",
        )
        if col_fmt:
            kwargs["column_format"] = col_fmt
        # Replace \hline with booktabs rules (\toprule, \midrule, \bottomrule)
        _hline_count = 0
        _lines = []
        for _line in df.to_latex(**kwargs).split("\n"):
            if _line.strip() == r"\hline":
                _hline_count += 1
                _lines.append(
                    r"\toprule" if _hline_count == 1
                    else r"\midrule" if _hline_count == 2
                    else r"\bottomrule"
                )
            else:
                _lines.append(_line)
        print("\n".join(_lines))

    return bold_best_cols, bold_best_pivot, latex_table


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
def _(
    bold_best_cols,
    cote_table,
    display_name,
    latex_table,
    map_df,
    mo,
    results_df,
):
    """d_pars — median per parsing model (independent of OCR model).

    Also joins COTe total score and mAP@0.5.
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
        .join(map_df[["mAP@0.5"]])
        .rename(index=display_name)
        .round(4)
    )

    latex_table(
        bold_best_cols(
            d_pars_table,
            lower_cols=["SpACER macro", "CDD"],
            higher_cols=["COTe", "mAP@0.5"],
        ),
        caption=r"Parsing error ($d_\text{pars}$) by parsing model with COTe and mAP@0.5. "
                r"SpACER macro is the primary metric; lower is better for SpACER/CDD, higher for COTe/mAP.",
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
def _(
    Path,
    bbox_df,
    bold_best_cols,
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    cote_score,
    display_name,
    eval_shape,
    latex_table,
    mo,
    parsing_models,
    pd,
):
    """COTe score — Coverage, Overlap, Trespass, Excess per parsing model."""

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _GT_BBOXES_PATH = _REPO_ROOT / "data/spiritualist/gt_ssu_bboxes.csv"

    _gt_ssu_df = pd.read_csv(_GT_BBOXES_PATH)
    _ssu_codes, _ = pd.factorize(_gt_ssu_df["ssu_id"])
    _gt_ssu_df = _gt_ssu_df.copy()
    _gt_ssu_df["ssu_int"] = _ssu_codes + 1

    _MAX_DIM = 500

    _cote_records = []
    for _filename, _gt_page in _gt_ssu_df.groupby("filename"):
        _page_id = Path(_filename).stem
        _orig_w = int(_gt_page["image_width"].iloc[0])
        _orig_h = int(_gt_page["image_height"].iloc[0])
        _eval_w, _eval_h, _scale = eval_shape(_orig_w, _orig_h, max_dim=_MAX_DIM)

        _ssu_map = boxes_to_gt_ssu_map(
            _gt_page.to_dict("records"), _eval_w, _eval_h,
            scale=_scale, ssu_id_key="ssu_int",
        )

        for _pm in parsing_models:
            _pred_page = bbox_df.loc[
                (bbox_df["parsing_model"] == _pm) &
                (bbox_df["filename"] == _filename)
            ]
            _preds = boxes_to_pred_masks(
                _pred_page.to_dict("records"), _eval_w, _eval_h, scale=_scale,
            )
            _cote, _C, _O, _T, _E = cote_score(_ssu_map, _preds)
            _cote_records.append({
                "page": _page_id,
                "parsing_model": _pm,
                "cote": _cote,
                "coverage": _C,
                "overlap": _O,
                "trespass": _T,
                "excess": _E,
            })

    cote_df = pd.DataFrame(_cote_records)
    # Keep raw index so d_pars cell can join on parsing_model names directly.
    cote_table = (
        cote_df[cote_df["parsing_model"] != "gt"]
        .groupby("parsing_model")[["cote", "coverage", "overlap", "trespass", "excess"]]
        .mean()
        .round(2)
    )

    _cote_display = cote_table.rename(index=display_name)
    latex_table(
        bold_best_cols(
            _cote_display,
            higher_cols=["cote", "coverage"],
            lower_cols=["overlap", "trespass", "excess"],
        ),
        caption=r"COTe score and components by parsing model. "
                r"Higher is better for COTe and Coverage; lower is better for Overlap, Trespass, Excess.",
        label="tab:cote",
    )

    mo.vstack([
        mo.md("### COTe score — mean by parsing model"),
        mo.ui.table(_cote_display, selection=None),
    ])
    return cote_df, cote_table


@app.cell
def _(Path, bbox_df, mo, np, parsing_models, pd):
    """mAP@0.5 — single-class object detection mAP per parsing model."""

    def _iou_matrix(pred_boxes, gt_boxes):
        def _to_xyxy(b):
            return np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]])
        p = _to_xyxy(pred_boxes)
        g = _to_xyxy(gt_boxes)
        inter_x1 = np.maximum(p[:, None, 0], g[None, :, 0])
        inter_y1 = np.maximum(p[:, None, 1], g[None, :, 1])
        inter_x2 = np.minimum(p[:, None, 2], g[None, :, 2])
        inter_y2 = np.minimum(p[:, None, 3], g[None, :, 3])
        inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
        area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        union = area_p[:, None] + area_g[None, :] - inter
        return np.where(union > 0, inter / union, 0.0)

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _gt_ssu_df = pd.read_csv(_REPO_ROOT / "data/spiritualist/gt_ssu_bboxes.csv")

    _map_records = []
    for _pm in [pm for pm in parsing_models if pm != "gt"]:
        _ap_per_page = []
        for _filename, _gt_page in _gt_ssu_df.groupby("filename"):
            _gt_boxes = _gt_page[["x", "y", "width", "height"]].values.astype(float)
            _pred_page = bbox_df.loc[
                (bbox_df["parsing_model"] == _pm) &
                (bbox_df["filename"] == _filename)
            ]
            if len(_pred_page) == 0:
                _ap_per_page.append(0.0)
                continue
            _pred_boxes = _pred_page[["x", "y", "width", "height"]].values.astype(float)
            _iou = _iou_matrix(_pred_boxes, _gt_boxes)
            _matched_gt = set()
            _tp = 0
            for _pi in np.argsort(-_iou.max(axis=1)):
                _gi = int(_iou[_pi].argmax())
                if _iou[_pi, _gi] >= 0.5 and _gi not in _matched_gt:
                    _tp += 1
                    _matched_gt.add(_gi)
            _fp = len(_pred_boxes) - _tp
            _fn = len(_gt_boxes) - _tp
            _denom = _tp + _fp + _fn
            _ap_per_page.append(_tp / _denom if _denom > 0 else 0.0)
        _map_records.append({"parsing_model": _pm, "mAP@0.5": round(float(np.mean(_ap_per_page)), 4)})

    map_df = pd.DataFrame(_map_records).set_index("parsing_model")
    mo.vstack([mo.md("### mAP@0.5 by parsing model"), mo.ui.table(map_df, selection=None)])
    return (map_df,)


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
def _(bold_best_cols, cote_df, display_name, latex_table, mo, pd, results_df):
    """Spearman correlation between per-page d_pars (SpACER & CDD) and COTe score."""
    from scipy.stats import spearmanr

    # Per-page d_pars for both metrics; d_pars is ocr_model-independent so average across it
    _dpars = (
        results_df[results_df["parsing_model"] != "gt"]
        .groupby(["page", "parsing_model"])[["d_pars_spacer_macro", "d_pars_cdd"]]
        .mean()
        .reset_index()
        .merge(cote_df[["page", "parsing_model", "cote"]], on=["page", "parsing_model"])
    )

    _records = []
    for _pm, _grp in _dpars.groupby("parsing_model"):
        _r_sp, _p_sp = spearmanr(_grp["cote"], _grp["d_pars_spacer_macro"])
        _r_cdd, _p_cdd = spearmanr(_grp["cote"], _grp["d_pars_cdd"])
        _records.append({
            "parsing_model": _pm,
            "SpACER $\\rho$": round(_r_sp, 3),
            "CDD $\\rho$": round(_r_cdd, 3),
        })

    _corr_df = (
        pd.DataFrame(_records)
        .set_index("parsing_model")
        .rename(index=display_name)
    )

    latex_table(
        bold_best_cols(_corr_df, lower_cols=["SpACER $\\rho$", "CDD $\\rho$"]),
        caption=r"Spearman correlation ($\rho$) between per-page $d_\text{pars}$ "
                r"and COTe score by parsing model. Negative $\rho$ indicates that "
                r"higher COTe (better parsing geometry) corresponds to lower parsing error, "
                r"as expected.",
        label="tab:dpars_cote_spearman",
    )

    _lines = ["**Spearman correlation: d_pars vs COTe**\n"]
    for _, _row in _corr_df.iterrows():
        _lines.append(
            f"- **{_}**: SpACER ρ = {_row['SpACER $\\rho$']:.3f}, "
            f"CDD ρ = {_row['CDD $\\rho$']:.3f}"
        )

    mo.vstack([
        mo.md("\n".join(_lines)),
        mo.ui.table(_corr_df, selection=None),
    ])
    return (spearmanr,)


@app.cell
def _(
    Counter,
    cdd_decomp,
    chars_df,
    gt_ocr_df,
    jiwer_cer,
    mo,
    normalize_for_cer,
    p9,
    spacer,
):
    """Per-box CER vs d_ocr SpACER/CDD — merge-based, no loops."""

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

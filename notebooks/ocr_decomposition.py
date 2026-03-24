import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_REPO_ROOT / "cotescore/src"))

    import marimo as mo
    import numpy as np
    import pandas as pd

    from cotescore import RegionChars, cdd_decomp_spatial, spacer_decomp_spatial

    return (
        Path,
        RegionChars,
        cdd_decomp_spatial,
        mo,
        np,
        pd,
        spacer_decomp_spatial,
    )


@app.cell
def _(Path, pd):
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _CHARS_PATH = _REPO_ROOT / "data/spiritualist/characters_inferred.parquet"
    _OCR_DIR = _REPO_ROOT / "data/results_spiritualist/ocr"
    _BBOX_DIR = _REPO_ROOT / "data/results_spiritualist/bboxes"

    chars_df = pd.read_parquet(_CHARS_PATH)
    chars_df = chars_df[chars_df["char_text"] != " "].reset_index(drop=True)
    # Midpoints for point-in-box testing
    chars_df["cx"] = (chars_df["x"] + chars_df["w"] / 2).astype(int)
    chars_df["cy"] = (chars_df["y"] + chars_df["h"] / 2).astype(int)

    # Load all OCR parquets keyed by (parsing_model, ocr_model)
    # Filename pattern: spiritualist_{parsing_model}_predictions_{ocr_model}_ocr.parquet
    _ocr_dfs = {}
    for _f in sorted(_OCR_DIR.glob("*_ocr.parquet")):
        _inner = _f.stem.removeprefix("spiritualist_").removesuffix("_ocr")
        _sep = _inner.index("_predictions_")
        _pm = _inner[:_sep]
        _om = _inner[_sep + len("_predictions_"):]
        _ocr_dfs[(_pm, _om)] = pd.read_parquet(_f)

    # Load all bbox CSVs keyed by parsing_model
    # Filename pattern: spiritualist_{parsing_model}_predictions.csv
    _bbox_dfs = {}
    for _f in sorted(_BBOX_DIR.glob("*.csv")):
        _pm = _f.stem.removeprefix("spiritualist_").removesuffix("_predictions")
        _bbox_dfs[_pm] = pd.read_csv(_f)

    ocr_dfs = _ocr_dfs
    bbox_dfs = _bbox_dfs
    parsing_models = sorted(_bbox_dfs.keys())
    ocr_models = sorted({k[1] for k in _ocr_dfs})
    pages = sorted(chars_df["page_id"].unique())
    return bbox_dfs, chars_df, ocr_dfs, ocr_models, pages, parsing_models


@app.cell
def _(ocr_dfs):
    ocr_dfs
    return


@app.cell
def _(chars_df):
    chars_df
    return


@app.cell
def _(
    RegionChars,
    bbox_dfs,
    cdd_decomp_spatial,
    chars_df,
    np,
    ocr_dfs,
    ocr_models,
    pages,
    parsing_models,
    pd,
    spacer_decomp_spatial,
):
    """Precompute CDD and SpACER decompositions for all pages and model combinations.

    Data assumptions:
      chars_df columns: char_text, cx, cy, ssu_id, page_id
        (ssu_id is a string like "ssu_1_col_1" — from tagged ALTO via infer_characters.py)
      ocr_dfs[("gt", om)] columns: filename, ssu_id, ocr_text
        (OCR run on GT SSU regions; ssu_id is the same string as in chars_df)
      ocr_dfs[(pm, om)] columns: filename, x, y, width, height, ocr_text
        (OCR run on predicted regions; bbox matched by integer coordinates)
    """

    def _join_ocr_text(texts):
        """Join space-separated OCR character tokens into a plain string."""
        return " ".join(texts).replace(" ", "")

    def _bbox_key(x, y, w, h):
        return (int(x), int(y), int(w), int(h))

    from tqdm import tqdm

    _records = []
    for _page in tqdm(pages, desc="pages"):
        _chars_page = chars_df[chars_df["page_id"] == _page]

        # Factorize string ssu_ids to integers for RegionChars
        _ssu_codes, _ssu_uniques = pd.factorize(_chars_page["ssu_id"])
        _ssu_to_int = {s: i for i, s in enumerate(_ssu_uniques)}

        _gt_chars = RegionChars(
            tokens=_chars_page["char_text"].to_numpy(dtype=object),
            xs=_chars_page["cx"].to_numpy(dtype=np.intp),
            ys=_chars_page["cy"].to_numpy(dtype=np.intp),
            region_ids=_ssu_codes.astype(np.intp),
        )

        for _pm in parsing_models:
            _bbox_page = bbox_dfs[_pm][bbox_dfs[_pm]["filename"] == f"{_page}.jpg"].reset_index(drop=True)

            # (M, 4) array [x, y, w, h] — used directly by the fast bbox path
            _bbox_arr = _bbox_page[["x", "y", "width", "height"]].to_numpy(dtype=float)

            # Coordinate → 0-based-index lookup for matching OCR output rows
            _bbox_key_to_id = {
                _bbox_key(r["x"], r["y"], r["width"], r["height"]): i
                for i, r in enumerate(_bbox_page.to_dict("records"))
            }

            for _om in ocr_models:
                # pred_gt_ocr: {ssu_int -> ocr_text} from OCR on GT regions
                _gt_key = ("gt", _om)
                if _gt_key in ocr_dfs:
                    _gt_page = ocr_dfs[_gt_key][ocr_dfs[_gt_key]["filename"] == f"{_page}.jpg"]
                    _pred_gt_ocr = {
                        _ssu_to_int[row["ssu_id"]]: _join_ocr_text(row["ocr_text"].split())
                        for _, row in _gt_page.iterrows()
                        if row["ssu_id"] in _ssu_to_int
                    }
                else:
                    _pred_gt_ocr = {}

                # pred_parse_ocr: {bbox_id -> ocr_text} matched by integer coordinates
                _pred_key = (_pm, _om)
                if _pred_key in ocr_dfs:
                    _pred_page = ocr_dfs[_pred_key][ocr_dfs[_pred_key]["filename"] == f"{_page}.jpg"]
                    _pred_parse_ocr = {
                        _bbox_key_to_id[_k]: _join_ocr_text(row["ocr_text"].split())
                        for _, row in _pred_page.iterrows()
                        if (_k := _bbox_key(row["x"], row["y"], row["width"], row["height"])) in _bbox_key_to_id
                    }
                else:
                    _pred_parse_ocr = {}

                _cdd = cdd_decomp_spatial(_gt_chars, _bbox_arr, _pred_gt_ocr, _pred_parse_ocr)
                _sp = spacer_decomp_spatial(_gt_chars, _bbox_arr, _pred_gt_ocr, _pred_parse_ocr)

                from collections import Counter
                _Q = Counter(_gt_chars.tokens.tolist())
                from cotescore._distributions import build_R_spatial
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
                    # SpACER macro
                    "d_pars_spacer_macro": _sp.d_pars_macro,
                    "d_ocr_spacer_macro": _sp.d_ocr_macro,
                    "d_int_spacer_macro": _sp.d_int_macro,
                    "d_total_spacer_macro": _sp.d_total_macro,
                    # SpACER micro (d_pars_micro is always None with spatial API)
                    "d_ocr_spacer_micro": _sp.d_ocr_micro,
                    "d_int_spacer_micro": _sp.d_int_micro,
                    "d_total_spacer_micro": _sp.d_total_micro,
                })

    results_df = pd.DataFrame(_records)
    return (results_df,)


@app.cell
def _(mo, results_df):
    """
    d_ocr — mean per OCR model (independent of parsing model).
    d_ocr is identical across all parsing models (it only depends on ocr_model),
    so we deduplicate by taking the first parsing_model's rows per (page, ocr_model).
    """
    _first_pm = results_df["parsing_model"].iloc[0]
    _ocr_rows = results_df[results_df["parsing_model"] == _first_pm]

    _d_ocr_cdd = (
        _ocr_rows.groupby("ocr_model")[["d_ocr_cdd"]]
        .mean()
        .rename(columns={"d_ocr_cdd": "CDD"})
    )
    _d_ocr_spacer_macro = (
        _ocr_rows.groupby("ocr_model")[["d_ocr_spacer_macro"]]
        .mean()
        .rename(columns={"d_ocr_spacer_macro": "SpACER macro"})
    )
    _d_ocr_spacer_micro = (
        _ocr_rows.groupby("ocr_model")[["d_ocr_spacer_micro"]]
        .median()
        .rename(columns={"d_ocr_spacer_micro": "SpACER micro"})
    )
    d_ocr_table = _d_ocr_cdd.join(_d_ocr_spacer_macro).join(_d_ocr_spacer_micro).round(4)

    mo.vstack([mo.md("### d_ocr — mean by OCR model"), mo.ui.table(d_ocr_table, selection=None)])
    return


@app.cell
def _(mo, results_df):
    """
    d_pars — mean per parsing model (independent of OCR model).
    d_pars depends only on the parser, so we average across all OCR models and pages.
    """
    _parse_rows = results_df

    _d_pars_cdd = (
        _parse_rows.groupby("parsing_model")[["d_pars_cdd"]]
        .median()
        .rename(columns={"d_pars_cdd": "CDD"})
    )
    _d_pars_spacer_macro = (
        _parse_rows.groupby("parsing_model")[["d_pars_spacer_macro"]]
        .median()
        .rename(columns={"d_pars_spacer_macro": "SpACER macro"})
    )
    d_pars_table = _d_pars_cdd.join(_d_pars_spacer_macro).round(4)

    mo.vstack([mo.md("### d_pars — median by parsing model"), mo.ui.table(d_pars_table, selection=None)])
    return


@app.cell
def _(mo, results_df):
    """
    d_int — mean grouped by (parsing_model × ocr_model).
    """
    _parse_rows = results_df

    _d_int_cdd = (
        _parse_rows.groupby(["parsing_model", "ocr_model"])["d_int_cdd"]
        .mean()
        .unstack("ocr_model")
        .round(4)
    )
    _d_int_spacer_macro = (
        _parse_rows.groupby(["parsing_model", "ocr_model"])["d_int_spacer_macro"]
        .median()
        .unstack("ocr_model")
        .round(4)
    )
    _d_int_spacer_micro = (
        _parse_rows.groupby(["parsing_model", "ocr_model"])["d_int_spacer_micro"]
        .median()
        .unstack("ocr_model")
        .round(4)
    )

    mo.vstack([
        mo.md("### d_int — mean by parsing × OCR model"),
        mo.md("**CDD**"), mo.ui.table(_d_int_cdd.reset_index(), selection=None),
        mo.md("**SpACER macro**"), mo.ui.table(_d_int_spacer_macro.reset_index(), selection=None),
        mo.md("**SpACER micro**"), mo.ui.table(_d_int_spacer_micro.reset_index(), selection=None),
    ])
    return


@app.cell
def _(mo, results_df):
    """
    d_total — mean grouped by (parsing_model × ocr_model).
    """
    _parse_rows = results_df

    _d_total_cdd = (
        _parse_rows.groupby(["parsing_model", "ocr_model"])["d_total_cdd"]
        .mean()
        .unstack("ocr_model")
        .round(4)
    )
    _d_total_spacer_macro = (
        _parse_rows.groupby(["parsing_model", "ocr_model"])["d_total_spacer_macro"]
        .mean()
        .unstack("ocr_model")
        .round(4)
    )
    _d_total_spacer_micro = (
        _parse_rows.groupby(["parsing_model", "ocr_model"])["d_total_spacer_macro"]
        .median()
        .unstack("ocr_model")
        .round(4)
    )

    mo.vstack([
        mo.md("### d_total — mean by parsing × OCR model"),
        mo.md("**CDD**"), mo.ui.table(_d_total_cdd.reset_index(), selection=None),
        mo.md("**SpACER mean**"), mo.ui.table(_d_total_spacer_macro.reset_index(), selection=None),
        mo.md("**SpACER median**"), mo.ui.table(_d_total_spacer_micro.reset_index(), selection=None),
    ])
    return


@app.cell
def _(Path, bbox_dfs, mo, parsing_models, pd):
    """COTe score — Coverage, Overlap, Trespass, Excess per parsing model.

    Uses the pre-built GT SSU bboxes CSV and the existing adapter functions
    (boxes_to_gt_ssu_map, boxes_to_pred_masks, eval_shape) to evaluate each
    parsing model's predicted boxes against the ground truth.
    """
    from cotescore import cote_score
    from cotescore.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks, eval_shape

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _GT_BBOXES_PATH = _REPO_ROOT / "data/spiritualist/gt_ssu_bboxes.csv"

    # Factorize string ssu_ids to unique positive integers (0 = background)
    _gt_ssu_df = pd.read_csv(_GT_BBOXES_PATH)
    _ssu_codes, _ = pd.factorize(_gt_ssu_df["ssu_id"])
    _gt_ssu_df = _gt_ssu_df.copy()
    _gt_ssu_df["ssu_int"] = _ssu_codes + 1

    _MAX_DIM = 500  # longest-side cap; eval_shape derives the scale factor

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
            _pred_page = bbox_dfs[_pm][bbox_dfs[_pm]["filename"] == _filename]
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
    cote_table = (
        cote_df.groupby("parsing_model")[["cote", "coverage", "overlap", "trespass", "excess"]]
        .mean()
        .round(4)
    )

    mo.vstack([
        mo.md("### COTe score — mean by parsing model"),
        mo.ui.table(cote_table, selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()

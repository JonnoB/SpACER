import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # COTe-conditioned SpACER/CER validation

    Tests the thesis that CER becomes functionally undefined once parsing
    quality degrades: below some COTe level, reading order no longer has
    meaning, so CER stops correlating with anything — while SpACER (bag of
    characters) degrades gracefully.

    Method: sliding threshold split. For each threshold t in {0.1, …, 0.9},
    observations (page × parsing_model × ocr_model) are split into
    COTe < t and COTe ≥ t groups and Spearman correlations are computed
    within each group, pooled and per parsing model (6 scopes per dataset).
    Datasets are kept separate — COTe magnitudes are not comparable across
    them. Groups with fewer than 30 observations report NaN.

    All inputs are cached: `cote_score_cache.parquet` +
    `page_level_cer_comparison.parquet` per dataset (spiritualist COTe cache
    is built here on first run using cotescore's bbox fast path).
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotnine as p9
    from cotescore import GTBoxes, cote_score

    return GTBoxes, Path, cote_score, np, p9, pd


@app.cell
def _(GTBoxes, Path, cote_score, pd):
    """Spiritualist COTe cache — the one dataset without one on disk.

    Same schema as the hiertext/docbank caches (page, parsing_model, cote,
    coverage, overlap, trespass, excess), computed with the analytic bbox
    fast path. Delete the file to force a recompute.
    """
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _CACHE = _REPO_ROOT / "data/spiritualist/cote_score_cache.parquet"

    if not _CACHE.exists():
        _gt_ssu = pd.read_csv(_REPO_ROOT / "data/spiritualist/gt_ssu_bboxes.csv")
        _ssu_codes, _ = pd.factorize(_gt_ssu["ssu_id"])
        _gt_ssu = _gt_ssu.copy()
        _gt_ssu["ssu_int"] = _ssu_codes + 1

        _BBOX_DIR = _REPO_ROOT / "data/results_spiritualist/bboxes"
        _bbox_parts = []
        for _f in sorted(_BBOX_DIR.glob("*.csv")):
            _pm = _f.stem.removeprefix("spiritualist_").removesuffix("_predictions")
            _part = pd.read_csv(_f)
            _part["parsing_model"] = _pm
            _bbox_parts.append(_part)
        _bbox_df = pd.concat(_bbox_parts, ignore_index=True)

        _records = []
        for _filename, _gt_page in _gt_ssu.groupby("filename"):
            _gt_boxes = GTBoxes(
                boxes=_gt_page[["x", "y", "width", "height"]].to_numpy(dtype=float),
                ssu_ids=_gt_page["ssu_int"].to_numpy(dtype=int),
                image_width=int(_gt_page["image_width"].iloc[0]),
                image_height=int(_gt_page["image_height"].iloc[0]),
            )
            for _pm in sorted(_bbox_df["parsing_model"].unique()):
                _pred_page = _bbox_df.loc[
                    (_bbox_df["parsing_model"] == _pm) &
                    (_bbox_df["filename"] == _filename)
                ]
                _preds = _pred_page[["x", "y", "width", "height"]].to_numpy(dtype=float)
                _cote, _C, _O, _T, _E = cote_score(_gt_boxes, _preds)
                _records.append({
                    "page": Path(_filename).stem,
                    "parsing_model": _pm,
                    "cote": _cote,
                    "coverage": _C,
                    "overlap": _O,
                    "trespass": _T,
                    "excess": _E,
                })
        pd.DataFrame(_records).to_parquet(_CACHE, index=False)
        print(f"Computed and cached spiritualist COTe -> {_CACHE}")

    spiritualist_cote_ready = True
    return (spiritualist_cote_ready,)


@app.cell
def _(Path, pd, spiritualist_cote_ready):
    """Load and merge the cached inputs for all three datasets.

    One observation = (page, parsing_model, ocr_model), with that page's
    per-model COTe attached. gt parsing rows are dropped (COTe degenerate).
    """
    assert spiritualist_cote_ready

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _DATASETS = {
        "spiritualist": (
            "data/spiritualist/cote_score_cache.parquet",
            "data/results_spiritualist/page_level_cer_comparison.parquet",
        ),
        "hiertext": (
            "data/hiertext/cote_score_cache.parquet",
            "data/hiertext/page_level_cer_comparison.parquet",
        ),
        "docbank": (
            "data/docbank/cote_score_cache.parquet",
            "data/docbank/page_level_cer_comparison.parquet",
        ),
    }

    merged = {}
    for _name, (_cote_path, _page_path) in _DATASETS.items():
        _cote = pd.read_parquet(_REPO_ROOT / _cote_path)
        _pages = pd.read_parquet(_REPO_ROOT / _page_path)
        _df = (
            _pages[_pages["parsing_model"] != "gt"]
            .merge(_cote[["page", "parsing_model", "cote"]], on=["page", "parsing_model"])
            .dropna(subset=["cer", "spacer_total", "cdd_total"])
        )
        merged[_name] = _df
        print(f"{_name}: {len(_df):,} observations "
              f"({_df['page'].nunique()} pages × {_df['parsing_model'].nunique()} parsing "
              f"× {_df['ocr_model'].nunique()} OCR models)")
    return (merged,)


@app.cell
def _(np, pd):
    """Sliding-threshold split statistics."""

    THRESHOLDS = np.round(np.arange(0.1, 1.0, 0.1), 1)
    MIN_N = 30

    def threshold_split_stats(df, thresholds=THRESHOLDS, min_n=MIN_N):
        """For each scope (pooled + per parsing model) and threshold t, split
        observations into COTe < t vs COTe ≥ t and compute within-group
        Spearman correlations against CER, plus the CER-spread sanity check.
        Groups with n < min_n report NaN correlations."""
        scopes = [("pooled", df)] + [
            (pm, grp) for pm, grp in df.groupby("parsing_model")
        ]
        rows = []
        for scope, sub in scopes:
            for t in thresholds:
                for side, part in (
                    ("below", sub[sub["cote"] < t]),
                    ("above", sub[sub["cote"] >= t]),
                ):
                    row = {"scope": scope, "threshold": t, "side": side, "n": len(part)}
                    if len(part) >= min_n:
                        row["cote_cer_rho"] = part["cote"].corr(part["cer"], method="spearman")
                        row["spacer_cer_rho"] = part["spacer_total"].corr(part["cer"], method="spearman")
                        row["cdd_cer_rho"] = part["cdd_total"].corr(part["cer"], method="spearman")
                        row["cer_median"] = part["cer"].median()
                        row["cer_sd"] = part["cer"].std()
                    else:
                        row.update({
                            "cote_cer_rho": np.nan, "spacer_cer_rho": np.nan,
                            "cdd_cer_rho": np.nan, "cer_median": np.nan, "cer_sd": np.nan,
                        })
                    rows.append(row)
        return pd.DataFrame(rows).round(4)

    return (threshold_split_stats,)


@app.cell
def _(merged, threshold_split_stats):
    split_stats = {name: threshold_split_stats(df) for name, df in merged.items()}
    return (split_stats,)


@app.cell
def _(p9, pd):
    def plot_split(stats_df, dataset_name):
        """Correlation vs threshold, below/above groups, faceted by scope."""
        long = stats_df.melt(
            id_vars=["scope", "threshold", "side", "n"],
            value_vars=["spacer_cer_rho", "cote_cer_rho"],
            var_name="metric",
            value_name="rho",
        ).dropna(subset=["rho"])
        long["metric"] = long["metric"].map({
            "spacer_cer_rho": "SpACER vs CER",
            "cote_cer_rho": "COTe vs CER",
        })
        long["scope"] = pd.Categorical(
            long["scope"],
            categories=["pooled"] + sorted(s for s in long["scope"].unique() if s != "pooled"),
        )
        return (
            p9.ggplot(long, p9.aes(x="threshold", y="rho", color="side", linetype="metric"))
            + p9.geom_line()
            + p9.geom_point(size=1.5)
            + p9.geom_hline(yintercept=0, color="gray", size=0.3)
            + p9.facet_wrap("~scope", ncol=3)
            + p9.scale_color_manual(values={"below": "#D55E00", "above": "#0072B2"})
            + p9.labs(
                title=f"{dataset_name}: within-group Spearman correlation by COTe split threshold",
                x="COTe split threshold",
                y="Spearman ρ (vs CER)",
                color="Group",
                linetype="Metric",
            )
            + p9.theme_minimal()
            + p9.theme(figure_size=(12, 6))
        )

    return (plot_split,)


@app.cell
def _(mo, plot_split, split_stats):
    _name = "spiritualist"
    mo.vstack([
        mo.md(f"## {_name}"),
        mo.ui.table(split_stats[_name], selection=None),
        mo.as_html(plot_split(split_stats[_name], _name)),
    ])
    return


@app.cell
def _(mo, plot_split, split_stats):
    _name = "hiertext"
    mo.vstack([
        mo.md(f"## {_name}"),
        mo.ui.table(split_stats[_name], selection=None),
        mo.as_html(plot_split(split_stats[_name], _name)),
    ])
    return


@app.cell
def _(mo, plot_split, split_stats):
    _name = "docbank"
    mo.vstack([
        mo.md(f"## {_name}"),
        mo.ui.table(split_stats[_name], selection=None),
        mo.as_html(plot_split(split_stats[_name], _name)),
    ])
    return


@app.cell
def _(pd, split_stats):
    """Headline table: the t = 0.5 split, all datasets side by side."""
    headline = pd.concat(
        [
            df.loc[df["threshold"] == 0.5].assign(dataset=name)
            for name, df in split_stats.items()
        ],
        ignore_index=True,
    )[["dataset", "scope", "side", "n",
       "cote_cer_rho", "spacer_cer_rho", "cdd_cer_rho", "cer_median", "cer_sd"]]
    headline
    return


if __name__ == "__main__":
    app.run()

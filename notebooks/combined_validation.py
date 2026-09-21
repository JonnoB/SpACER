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
    # Combined cross-dataset validation

    Combines the two Spearman-correlation tables and the CER-agreement
    heatmap that each per-dataset notebook (`spiritualist_decomposition.py`,
    `hiertext_decomposition.py`, `docbank_decomposition.py`,
    `*_page_level_validation.py`) produces individually, into one set of
    dataset-comparison tables/figures for the paper's validation section.

    SpACER and CDD get separate tables (four total, not two) so each table
    has one column per dataset — the point of this notebook is comparing
    *across datasets*, not across metrics within one dataset:

    - **Table 0** — Dataset statistics: number of GT regions, words,
      characters, and source images per dataset, plus whether GT regions
      are annotated as polygons or axis-aligned bounding boxes.
    - **Table 1/2** — Spearman $\rho$ between per-box CER and $d_\text{ocr}$
      (SpACER, then CDD), computed over GT regions. Rows = OCR model.
    - **Table 3/4** — Spearman $\rho$ between per-page $d_\text{pars}$ and
      COTe (SpACER, then CDD). Rows = parsing model. Negative $\rho$ is
      expected (higher COTe = better geometry = lower parsing error).
    - **Table 5/6** — Spearman $\rho$ between page-level CER and
      SpACER/CDD totals, pooled and split at COTe = 0.5. Rows = dataset.
      A headline summary of the full sliding-threshold analysis in
      `cote_conditioned_validation.py`; higher $\rho$ above the split than
      below is the "CER is only a valid proxy once parsing geometry is
      usable" result.
    - **Heatmap** — per-(parsing model, OCR model) Spearman $\rho$ between
      page-level CER and SpACER/CDD totals, 6 facets (2 metrics x 3
      datasets) instead of the per-dataset notebooks' 2.

    Bolding (Tables 1-4 only) marks the best-performing model *for that
    dataset* (i.e. within a column) — not a cross-dataset comparison,
    since COTe/CER magnitudes aren't comparable across datasets, only the
    correlations are. Tables 5/6 aren't bolded: the point is the pooled
    vs. below vs. above *pattern*, not picking a winner.

    Reads only cached parquets (no fresh page/box loops): each dataset's
    `box_level_ocr_comparison.parquet`, `decomposition_results.parquet`,
    `cote_score_cache.parquet`, and `page_level_cer_comparison.parquet`,
    all produced by the per-dataset notebooks above. Re-run those first if
    you need to pick up new OCR/parsing results.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotnine as p9
    from scipy.stats import spearmanr

    from spacer_analysis.names import display_name
    from spacer_analysis.paths import DATASET_ORDER, DATASETS, REPO_ROOT
    from spacer_analysis.tables import bold_best_cols, latex_table

    return (
        DATASETS,
        DATASET_ORDER,
        Path,
        REPO_ROOT,
        bold_best_cols,
        display_name,
        latex_table,
        p9,
        pd,
        spearmanr,
    )

@app.cell
def _(DATASETS, DATASET_ORDER, REPO_ROOT, pd):
    """Table 0: dataset statistics, one row per dataset.

    Regions/words/characters are counted from each dataset's
    gt_ssu_bboxes.csv (one row per GT region, so no double-counting across
    OCR models the way box_level_ocr_comparison.parquet would). Region
    type reflects the annotation format: Spiritualist and HierText GT
    regions are polygons (`polygon_points` column present); DocBank's are
    axis-aligned bounding boxes (no `polygon_points` column).
    """
    _rows = {}
    for _name in DATASET_ORDER:
        _gt_df = pd.read_csv(DATASETS[_name].gt_ssu_bboxes)
        _text = _gt_df["gt_text"].fillna("").astype(str)
        _rows[DATASETS[_name].display] = pd.Series({
            "Regions": len(_gt_df),
            "Words": _text.str.split().str.len().sum(),
            "Characters": _text.str.len().sum(),
            "Images": _gt_df["page_id"].nunique(),
            "Region type": "Polygon" if "polygon_points" in _gt_df.columns else "Bounding box",
        })
    dataset_stats_table = pd.DataFrame(_rows).T
    dataset_stats_table.index.name = "Dataset"
    return (dataset_stats_table,)


@app.cell
def _(dataset_stats_table, latex_table, mo):
    _display_table = dataset_stats_table.copy()
    for _col in ["Regions", "Words", "Characters", "Images"]:
        _display_table[_col] = _display_table[_col].map(lambda v: f"{int(v):,}")
    latex_table(
        _display_table,
        caption=r"Dataset statistics: number of GT regions, words, and "
                r"characters, number of source images, and GT region "
                r"annotation format (polygon vs.\ axis-aligned bounding box).",
        label="tab:dataset_stats",
    )
    mo.vstack([
        mo.md("### Table 0 — Dataset statistics"),
        mo.ui.table(dataset_stats_table, selection=None),
    ])
    return


@app.cell
def _(DATASETS, DATASET_ORDER, REPO_ROOT, display_name, pd, spearmanr):
    """Tables 1 & 2: CER vs d_ocr Spearman correlation, by OCR model.

    Computed over GT regions (parsing-model independent), from each
    dataset's box_level_ocr_comparison.parquet. One column per dataset.
    """
    _spacer_cols = {}
    _cdd_cols = {}
    for _name in DATASET_ORDER:
        _box_df = pd.read_parquet(DATASETS[_name].box_level)
        _spacer_vals, _cdd_vals = {}, {}
        for _om, _grp in _box_df.groupby("ocr_model"):
            _r_sp, _ = spearmanr(_grp["cer"], _grp["d_ocr_spacer"])
            _r_cdd, _ = spearmanr(_grp["cer"], _grp["d_ocr_cdd"])
            _spacer_vals[display_name(_om)] = round(_r_sp, 3)
            _cdd_vals[display_name(_om)] = round(_r_cdd, 3)
        _spacer_cols[_name] = pd.Series(_spacer_vals)
        _cdd_cols[_name] = pd.Series(_cdd_vals)

    _display = {_n: DATASETS[_n].display for _n in DATASET_ORDER}
    cer_docr_spacer_table = pd.DataFrame(_spacer_cols)[DATASET_ORDER].rename(columns=_display)
    cer_docr_cdd_table = pd.DataFrame(_cdd_cols)[DATASET_ORDER].rename(columns=_display)
    cer_docr_spacer_table.index.name = "OCR Model"
    cer_docr_cdd_table.index.name = "OCR Model"
    return cer_docr_cdd_table, cer_docr_spacer_table


@app.cell
def _(bold_best_cols, cer_docr_spacer_table, latex_table, mo):
    latex_table(
        bold_best_cols(cer_docr_spacer_table, higher_cols=list(cer_docr_spacer_table.columns)),
        caption=r"Spearman correlation ($\rho$) between per-box CER and $d_\text{ocr}$ "
                r"(SpACER), computed over GT regions, by OCR model and dataset. "
                r"\textbf{Bold}: best OCR model for that dataset.",
        label="tab:cer_docr_spearman_spacer",
    )
    mo.vstack([
        mo.md("### Table 1 — CER vs $d_\\text{ocr}$ (SpACER) by OCR model"),
        mo.ui.table(cer_docr_spacer_table, selection=None),
    ])
    return


@app.cell
def _(bold_best_cols, cer_docr_cdd_table, latex_table, mo):
    latex_table(
        bold_best_cols(cer_docr_cdd_table, higher_cols=list(cer_docr_cdd_table.columns)),
        caption=r"Spearman correlation ($\rho$) between per-box CER and $d_\text{ocr}$ "
                r"(CDD), computed over GT regions, by OCR model and dataset. "
                r"\textbf{Bold}: best OCR model for that dataset.",
        label="tab:cer_docr_spearman_cdd",
    )
    mo.vstack([
        mo.md("### Table 2 — CER vs $d_\\text{ocr}$ (CDD) by OCR model"),
        mo.ui.table(cer_docr_cdd_table, selection=None),
    ])
    return


@app.cell
def _(DATASETS, DATASET_ORDER, REPO_ROOT, display_name, pd, spearmanr):
    """Tables 3 & 4: d_pars vs COTe Spearman correlation, by parsing model.

    Per-page d_pars (averaged over OCR model, which d_pars doesn't depend
    on) merged with that page's COTe, from each dataset's
    decomposition_results.parquet + cote_score_cache.parquet. gt parsing
    is dropped (COTe/d_pars are degenerate for it). One column per dataset.
    """
    _spacer_cols = {}
    _cdd_cols = {}
    for _name in DATASET_ORDER:
        _cote_df = pd.read_parquet(DATASETS[_name].cote_cache)
        _results_df = pd.read_parquet(DATASETS[_name].decomposition_results)
        _dpars = (
            _results_df[_results_df["parsing_model"] != "gt"]
            .groupby(["page", "parsing_model"])[["d_pars_spacer_macro", "d_pars_cdd"]]
            .mean()
            .reset_index()
            .merge(_cote_df[["page", "parsing_model", "cote"]], on=["page", "parsing_model"])
        )
        _spacer_vals, _cdd_vals = {}, {}
        for _pm, _grp in _dpars.groupby("parsing_model"):
            _r_sp, _ = spearmanr(_grp["cote"], _grp["d_pars_spacer_macro"])
            _r_cdd, _ = spearmanr(_grp["cote"], _grp["d_pars_cdd"])
            _spacer_vals[display_name(_pm)] = round(_r_sp, 3)
            _cdd_vals[display_name(_pm)] = round(_r_cdd, 3)
        _spacer_cols[_name] = pd.Series(_spacer_vals)
        _cdd_cols[_name] = pd.Series(_cdd_vals)

    _display = {_n: DATASETS[_n].display for _n in DATASET_ORDER}
    dpars_cote_spacer_table = pd.DataFrame(_spacer_cols)[DATASET_ORDER].rename(columns=_display)
    dpars_cote_cdd_table = pd.DataFrame(_cdd_cols)[DATASET_ORDER].rename(columns=_display)
    dpars_cote_spacer_table.index.name = "Parsing Model"
    dpars_cote_cdd_table.index.name = "Parsing Model"
    return dpars_cote_cdd_table, dpars_cote_spacer_table


@app.cell
def _(bold_best_cols, dpars_cote_spacer_table, latex_table, mo):
    latex_table(
        bold_best_cols(dpars_cote_spacer_table, lower_cols=list(dpars_cote_spacer_table.columns)),
        caption=r"Spearman correlation ($\rho$) between per-page $d_\text{pars}$ (SpACER) "
                r"and COTe score, by parsing model and dataset. Negative $\rho$ indicates "
                r"higher COTe (better parsing geometry) corresponds to lower parsing error, "
                r"as expected. \textbf{Bold}: best parsing model for that dataset.",
        label="tab:dpars_cote_spearman_spacer",
    )
    mo.vstack([
        mo.md("### Table 3 — $d_\\text{pars}$ (SpACER) vs COTe by parsing model"),
        mo.ui.table(dpars_cote_spacer_table, selection=None),
    ])
    return


@app.cell
def _(bold_best_cols, dpars_cote_cdd_table, latex_table, mo):
    latex_table(
        bold_best_cols(dpars_cote_cdd_table, lower_cols=list(dpars_cote_cdd_table.columns)),
        caption=r"Spearman correlation ($\rho$) between per-page $d_\text{pars}$ (CDD) "
                r"and COTe score, by parsing model and dataset. Negative $\rho$ indicates "
                r"higher COTe (better parsing geometry) corresponds to lower parsing error, "
                r"as expected. \textbf{Bold}: best parsing model for that dataset.",
        label="tab:dpars_cote_spearman_cdd",
    )
    mo.vstack([
        mo.md("### Table 4 — $d_\\text{pars}$ (CDD) vs COTe by parsing model"),
        mo.ui.table(dpars_cote_cdd_table, selection=None),
    ])
    return


@app.cell
def _(DATASETS, DATASET_ORDER, REPO_ROOT, pd):
    """Tables 5 & 6: pooled vs COTe-conditioned Spearman correlation
    (page-level CER vs SpACER/CDD totals), one row per dataset.

    Tests whether CER agreement holds up once parsing geometry crosses
    from "broken" to "usable" (see cote_conditioned_validation.py for the
    full sliding-threshold version this is a headline summary of). Split
    at COTe = 0.5, pooled across parsing/OCR model. gt parsing is dropped
    (COTe degenerate for it).
    """
    _spacer_rows = {}
    _cdd_rows = {}
    _n_rows = {}
    for _name in DATASET_ORDER:
        _cote_df = pd.read_parquet(DATASETS[_name].cote_cache)
        _page_df = pd.read_parquet(DATASETS[_name].page_level)
        _merged = (
            _page_df[_page_df["parsing_model"] != "gt"]
            .merge(_cote_df[["page", "parsing_model", "cote"]], on=["page", "parsing_model"])
            .dropna(subset=["cer", "spacer_total", "cdd_total"])
        )
        _groups = {
            "Pooled": _merged,
            "COTe < 0.5": _merged[_merged["cote"] < 0.5],
            "COTe ≥ 0.5": _merged[_merged["cote"] >= 0.5],
        }
        _spacer_rows[_name] = pd.Series({
            _label: round(_part["spacer_total"].corr(_part["cer"], method="spearman"), 3)
            for _label, _part in _groups.items()
        })
        _cdd_rows[_name] = pd.Series({
            _label: round(_part["cdd_total"].corr(_part["cer"], method="spearman"), 3)
            for _label, _part in _groups.items()
        })
        _n_rows[_name] = pd.Series({_label: len(_part) for _label, _part in _groups.items()})

    _col_order = ["Pooled", "COTe < 0.5", "COTe ≥ 0.5"]
    _display = {_n: DATASETS[_n].display for _n in DATASET_ORDER}
    cote_conditioned_spacer_table = pd.DataFrame(_spacer_rows).T[_col_order].loc[DATASET_ORDER].rename(index=_display)
    cote_conditioned_cdd_table = pd.DataFrame(_cdd_rows).T[_col_order].loc[DATASET_ORDER].rename(index=_display)
    cote_conditioned_n_table = pd.DataFrame(_n_rows).T[_col_order].loc[DATASET_ORDER].rename(index=_display)
    cote_conditioned_spacer_table.index.name = "Dataset"
    cote_conditioned_cdd_table.index.name = "Dataset"
    cote_conditioned_n_table.index.name = "Dataset"
    return (
        cote_conditioned_cdd_table,
        cote_conditioned_n_table,
        cote_conditioned_spacer_table,
    )


@app.cell
def _(
    cote_conditioned_n_table,
    cote_conditioned_spacer_table,
    latex_table,
    mo,
):
    latex_table(
        cote_conditioned_spacer_table,
        caption=r"Spearman correlation ($\rho$) between page-level CER and SpACER, "
                r"pooled and split at COTe = 0.5, by dataset. Higher $\rho$ in the "
                r"COTe $\geq 0.5$ column indicates CER recovers as a valid quality "
                r"proxy once parsing geometry is usable.",
        label="tab:cote_conditioned_spearman_spacer",
    )
    mo.vstack([
        mo.md("### Table 5 — CER vs SpACER, pooled / below / above COTe = 0.5"),
        mo.ui.table(cote_conditioned_spacer_table, selection=None),
        mo.md("Sample sizes (n observations per group):"),
        mo.ui.table(cote_conditioned_n_table, selection=None),
    ])
    return


@app.cell
def _(cote_conditioned_cdd_table, latex_table, mo):
    latex_table(
        cote_conditioned_cdd_table,
        caption=r"Spearman correlation ($\rho$) between page-level CER and CDD, "
                r"pooled and split at COTe = 0.5, by dataset. Higher $\rho$ in the "
                r"COTe $\geq 0.5$ column indicates CER recovers as a valid quality "
                r"proxy once parsing geometry is usable.",
        label="tab:cote_conditioned_spearman_cdd",
    )
    mo.vstack([
        mo.md("### Table 6 — CER vs CDD, pooled / below / above COTe = 0.5"),
        mo.ui.table(cote_conditioned_cdd_table, selection=None),
    ])
    return


@app.cell
def _(DATASETS, DATASET_ORDER, REPO_ROOT, display_name, pd):
    """Per-(parsing_model, ocr_model) Spearman corr(CER, SpACER/CDD totals),
    one row per dataset x parsing_model x ocr_model, for the 6-facet heatmap.
    """
    _records = []
    for _name in DATASET_ORDER:
        _page_df = pd.read_parquet(DATASETS[_name].page_level)
        _page_df = _page_df[_page_df["parsing_model"] != "gt"]
        _combos = _page_df[["parsing_model", "ocr_model"]].drop_duplicates()
        for _, _row in _combos.iterrows():
            _mask = (
                (_page_df["parsing_model"] == _row["parsing_model"]) &
                (_page_df["ocr_model"] == _row["ocr_model"])
            )
            _subset = _page_df.loc[_mask, ["cer", "spacer_total", "cdd_total"]]
            _corr_matrix = _subset.corr(method="spearman")
            _records.append({
                "dataset": _name,
                "parsing_model": display_name(_row["parsing_model"]),
                "ocr_model": display_name(_row["ocr_model"]),
                "SpACER vs CER": _corr_matrix.loc["spacer_total", "cer"],
                "CDD vs CER": _corr_matrix.loc["cdd_total", "cer"],
            })
    heatmap_corr_df = pd.DataFrame(_records)
    return (heatmap_corr_df,)


@app.cell
def _(DATASET_ORDER, heatmap_corr_df, p9, pd):
    """6-facet heatmap: rows = metric (SpACER, CDD), columns = dataset."""
    _long = heatmap_corr_df.melt(
        id_vars=["dataset", "parsing_model", "ocr_model"],
        value_vars=["SpACER vs CER", "CDD vs CER"],
        var_name="metric",
        value_name="correlation",
    )
    _long["dataset"] = pd.Categorical(_long["dataset"], categories=DATASET_ORDER)
    _long["metric"] = pd.Categorical(_long["metric"], categories=["SpACER vs CER", "CDD vs CER"])

    combined_heatmap = (
        p9.ggplot(_long, p9.aes(x="ocr_model", y="parsing_model", fill="correlation"))
        + p9.geom_tile()
        + p9.geom_text(p9.aes(label="correlation.round(2)"), size=11, color="white")
        + p9.facet_grid("metric ~ dataset")
        + p9.labs(
            title="Spearman correlation (CER vs SpACER/CDD) by model combination and dataset",
            x="OCR model",
            y="Parsing model",
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=(15, 6),
            axis_text_y=p9.element_text(size=11),
            axis_text_x=p9.element_text(rotation=45, hjust=1, size=10),
            panel_grid=p9.element_blank(),
            strip_text=p9.element_text(size=12, weight="bold"),
        )
    )
    combined_heatmap.save(filename='data/figures/CEV_CER_correlation.pdf', dpi = 300)
    combined_heatmap.draw()
    return


if __name__ == "__main__":
    app.run()

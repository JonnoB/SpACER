import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    # Cutoff-threshold heatmaps — all three datasets

    Reproduces the "Is OCR the dominant error source?" F1 heatmap from
    `notebooks/spiritualist_decomposition.py` for the spiritualist, DocBank
    and HierText datasets side by side.

    For each (page, parsing model, OCR model) row the classifier rule is

        predict OCR-dominant  iff  COTe >= x_thresh  AND  d_ocr / d_total >= y_thresh

    and the truth label is `d_ocr >= d_pars` (SpACER macro). F1 is evaluated
    over the full non-GT dataset for every (x_thresh, y_thresh) pair.

    This notebook reads the caches written by the three `*_decomposition.py`
    notebooks rather than recomputing the decomposition, so those notebooks
    must have been run at least once:

    - `data/{dataset}/decomposition_results.parquet` — `results_df`
    - `data/{dataset}/cote_score_cache.parquet` — `cote_df`

    Spiritualist is shown twice, since it is the one dataset with degenerate parsers
    (ppdoc_s / ppdoc_m — mean COTe ~0.13-0.26 against 0.75-0.86 for the others:
    enormous over-sized boxes), configured via `VARIANTS`:

    1. **Spiritualist** — all parsing models kept.
    2. **Spiritualist (excl. ppdoc_s/m)** — the degenerate parsers removed.

    On DocBank and HierText the same models sit inside the pack, so there is nothing
    to remove and each is shown once.

    Figures are written to `data/figures/cutoff_thresholds_{dataset}.pdf`, a
    combined faceted version to `data/figures/cutoff_thresholds_all.pdf`, and the
    ratio-only balanced-accuracy curves to `data/figures/ocr_over_total_balanced_acc_curve.pdf`.
    The original `data/figures/cutoff_thresholds.pdf` is left untouched.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotnine as p9

    from spacer_analysis.tables import latex_table

    return Path, latex_table, mo, np, p9, pd


@app.cell
def _(Path):
    REPO_ROOT = Path(__file__).resolve().parent.parent
    FIG_DIR = REPO_ROOT / "data/figures"

    DATASETS = {
        "spiritualist": "Spiritualist",
        "docbank": "DocBank",
        "hiertext": "HierText",
    }

    # Extra variants: a base dataset with some parsing models dropped.
    # key -> (base dataset, parsing models to drop, display name)
    DEGENERATE = ["ppdoc_s", "ppdoc_m"]
    VARIANTS = {
        "spiritualist_excl": ("spiritualist", DEGENERATE, "Spiritualist (excl. ppdoc_s/m)"),
    }

    X_THRESH = [round(v, 2) for v in [i / 10 for i in range(10)]]  # COTe: 0.0 .. 0.9
    Y_THRESH = [round(v, 2) for v in [i / 10 for i in range(10)]]  # d_ocr / d_total: 0.0 .. 0.9
    return DATASETS, FIG_DIR, REPO_ROOT, VARIANTS, X_THRESH, Y_THRESH


@app.cell
def _(DATASETS, REPO_ROOT, VARIANTS, mo, np, pd):
    """Load cached results_df + cote_df per dataset and build the classifier inputs."""

    _parts = {}
    _notes = []
    for _ds, _name in DATASETS.items():
        _results_path = REPO_ROOT / f"data/{_ds}/decomposition_results.parquet"
        _cote_path = REPO_ROOT / f"data/{_ds}/cote_score_cache.parquet"
        if not _results_path.exists() or not _cote_path.exists():
            _notes.append(f"- **{_name}**: missing cache, skipped (`{_results_path.name}` / `{_cote_path.name}`)")
            continue

        _results = pd.read_parquet(_results_path)
        _cote = pd.read_parquet(_cote_path)

        _df = (
            _results[_results["parsing_model"] != "gt"]
            .merge(_cote[["page", "parsing_model", "cote"]], on=["page", "parsing_model"], how="left")
        )

        # Drop rows where d_ocr / d_total is not a usable number:
        #   d_total == 0  -> no error at all; the ratio is 0/0 and the label is trivially true
        #   d_total NaN   -> the parsing model returned no regions, so d_total is undefined
        # Both must go, because `NaN >= t` is False at every threshold and would otherwise
        # count as a *predicted negative* — silently earning free true negatives, since
        # these rows are almost all genuinely parsing-dominant. (A deployment would route
        # an undefined ratio straight to "parsing" rather than through the threshold.)
        _bad_total = (_df["d_total_spacer_macro"] == 0) | ~np.isfinite(_df["d_total_spacer_macro"])
        _df = _df[~_bad_total].copy()

        _df["ocr_over_total"] = _df["d_ocr_spacer_macro"] / _df["d_total_spacer_macro"]
        _df["ocr_dominant"] = _df["d_ocr_spacer_macro"] >= _df["d_pars_spacer_macro"]

        _df["dataset"] = _ds
        _df["dataset_name"] = _name
        _parts[_ds] = _df

        _notes.append(
            f"- **{_name}**: {len(_df):,} rows, {_df['page'].nunique():,} pages, "
            f"missing COTe = {int(_df['cote'].isna().sum())}, dropped unusable d_total = {int(_bad_total.sum())}, "
            f"positive rate = {_df['ocr_dominant'].mean():.2f}"
        )

    for _key, (_base, _models, _name) in VARIANTS.items():
        if _base not in _parts:
            continue
        _df = _parts[_base][~_parts[_base]["parsing_model"].isin(_models)].copy()
        _df["dataset"] = _key
        _df["dataset_name"] = _name
        _parts[_key] = _df
        _notes.append(
            f"- **{_name}**: {len(_df):,} rows, positive rate = {_df['ocr_dominant'].mean():.2f} "
            f"({DATASETS[_base]} minus {', '.join(_models)})"
        )

    DATASET_NAMES = {k: v for k, v in DATASETS.items()} | {k: v[2] for k, v in VARIANTS.items()}
    prep_df = pd.concat(_parts.values(), ignore_index=True)
    prep_df["dataset_name"] = pd.Categorical(prep_df["dataset_name"], categories=[DATASET_NAMES[k] for k in _parts])

    mo.md("### Loaded datasets\n" + "\n".join(_notes))
    return DATASET_NAMES, prep_df


@app.cell
def _(FIG_DIR, mo, np, p9, pd, prep_df):
    """Balanced accuracy of the ratio-only rule (d_ocr / d_total >= t), one line per dataset.

    Balanced accuracy, not F1, because the classes are skewed (OCR-dominant base rate
    0.61-0.84) and F1 ignores true negatives, so its verdict depends on which class is
    nominated "positive" — on HierText the same rule loses to the trivial baseline by
    0.035 with OCR positive and beats it by 0.237 with parsing positive. Balanced
    accuracy is invariant under that relabelling, and 0.5 is exactly the score of any
    trivial always-one-class rule, so the dashed line is a real no-information floor.

    F1 is computed alongside and kept in `curve_df` for reference.
    """
    _t = np.linspace(0, 1, 201)

    def _curve(g):
        _y = g["ocr_over_total"].to_numpy()
        _label = g["ocr_dominant"].to_numpy(dtype=bool)
        _n_pos = int(_label.sum())
        _n_neg = int((~_label).sum())
        _f1, _bal = [], []
        for _x in _t:
            _pred = _y >= _x
            _tp = int((_pred & _label).sum())
            _fp = int((_pred & ~_label).sum())
            _fn = _n_pos - _tp
            _tn = _n_neg - _fp
            _denom = 2 * _tp + _fp + _fn
            _f1.append(2 * _tp / _denom if _denom > 0 else np.nan)
            _recall = _tp / _n_pos if _n_pos else np.nan
            _spec = _tn / _n_neg if _n_neg else np.nan
            _bal.append((_recall + _spec) / 2)
        return _f1, _bal

    _parts = []
    for _ds, _g in prep_df.groupby("dataset", sort=False):
        _f1, _bal = _curve(_g)
        _parts.append(pd.DataFrame({
            "threshold": _t,
            "f1": _f1,
            "balanced_acc": _bal,
            "dataset_name": _g["dataset_name"].iloc[0],
        }))
    curve_df = pd.concat(_parts, ignore_index=True)
    curve_df["dataset_name"] = pd.Categorical(curve_df["dataset_name"], categories=list(prep_df["dataset_name"].cat.categories))

    _colours = {
        "Spiritualist": "#3B6FB6",
        "Spiritualist (excl. ppdoc_s/m)": "#3B6FB6",
        "DocBank": "#E07B39",
        "HierText": "#3A9D5D",
    }
    _linetypes = {
        "Spiritualist": "solid",
        "Spiritualist (excl. ppdoc_s/m)": "dashed",
        "DocBank": "solid",
        "HierText": "solid",
    }

    bal_curve_plt = (
        p9.ggplot(curve_df, p9.aes(x="threshold", y="balanced_acc", colour="dataset_name", linetype="dataset_name"))
        + p9.geom_hline(yintercept=0.5, linetype="dashed", colour="grey", size=0.4)
        + p9.geom_line(size=1)
        + p9.scale_colour_manual(values=_colours)
        + p9.scale_linetype_manual(values=_linetypes)
        + p9.labs(
            x="SpACER d_ocr / d_total minimum threshold",
            y="Balanced accuracy (predicting d_ocr ≥ d_pars)",
            colour="Dataset",
            linetype="Dataset",
            title="Balanced accuracy of the d_ocr / d_total threshold",
        )
        + p9.theme(figure_size=(9, 5.5), legend_position="bottom")
    )
    bal_curve_plt.save(filename=str(FIG_DIR / "ocr_over_total_balanced_acc_curve.pdf"), dpi=300, verbose=False)
    mo.plain(bal_curve_plt)
    return


@app.cell
def _(mo, np, pd, prep_df):
    """Performance at the untuned t = 0.5 cutoff — the deployment setting.

    A practitioner can compute d_ocr / d_total from region-level GT (boxes + text)
    but cannot compute the label d_ocr >= d_pars, which needs character positions.
    So there is nothing to tune against: the cutoff is fixed a priori, and t = 0.5
    ("OCR accounts for at least half the total error") is the natural choice.

    Columns, read as two verdicts rather than a positive and a negative class:

      recall_ocr / recall_pars        of the pages where OCR (resp. parsing) truly
                                      dominates, the fraction the rule flags as such
                                      — i.e. sensitivity and specificity
      precision_ocr / precision_pars  when the rule returns that verdict, how often
                                      it is right — i.e. PPV and NPV. These are what
                                      a practitioner actually acts on, and
                                      precision_pars is the weak one: a "fix your
                                      parsing" verdict is the expensive one to get wrong.
      balanced_acc                    (recall_ocr + recall_pars) / 2 — the headline.

    Balanced accuracy rather than F1 because the classes are skewed (OCR-dominant base
    rate 0.61-0.84) and F1 ignores true negatives, so its verdict depends on which class
    is nominated "positive": on HierText the same rule loses to the trivial baseline by
    0.035 with OCR positive and beats it by 0.237 with parsing positive. Balanced
    accuracy is invariant under that relabelling, depends only on the class-conditional
    rates (so it does not encode this corpus's class mix, which a practitioner's will not
    match), and scores exactly 0.5 for any always-one-class rule. F1 and the trivial
    always-OCR F1 are kept as the last two columns for reference only.
    """
    NULL_THRESH = 0.5

    def _stats(y, label, t):
        _pred = y >= t
        _tp = int((_pred & label).sum())
        _fp = int((_pred & ~label).sum())
        _fn = int((~_pred & label).sum())
        _tn = int((~_pred & ~label).sum())
        _rec_ocr = _tp / (_tp + _fn) if _tp + _fn else np.nan
        _rec_pars = _tn / (_tn + _fp) if _tn + _fp else np.nan
        return {
            "recall_ocr": _rec_ocr,
            "recall_pars": _rec_pars,
            "precision_ocr": _tp / (_tp + _fp) if _tp + _fp else np.nan,
            "precision_pars": _tn / (_tn + _fn) if _tn + _fn else np.nan,
            "balanced_acc": (_rec_ocr + _rec_pars) / 2,
            "f1": 2 * _tp / (2 * _tp + _fp + _fn) if 2 * _tp + _fp + _fn else np.nan,
        }

    _grid = np.linspace(0, 1, 1001)
    _rows = []
    for _ds, _g in prep_df.groupby("dataset", sort=False):
        _y = _g["ocr_over_total"].to_numpy()
        _label = _g["ocr_dominant"].to_numpy(dtype=bool)
        _at_null = _stats(_y, _label, NULL_THRESH)
        _curve = [_stats(_y, _label, _t)["balanced_acc"] for _t in _grid]
        _bi = int(np.argmax(_curve))
        _rows.append({
            "dataset": _g["dataset_name"].iloc[0],
            "n": len(_g),
            "base_rate": _label.mean(),
            "recall_ocr": _at_null["recall_ocr"],
            "recall_pars": _at_null["recall_pars"],
            "precision_ocr": _at_null["precision_ocr"],
            "precision_pars": _at_null["precision_pars"],
            "balanced_acc": _at_null["balanced_acc"],
            "oracle_balanced_acc": _curve[_bi],
            "oracle_t": _grid[_bi],
            "f1": _at_null["f1"],
            "always_ocr_f1": _stats(_y, _label, 0.0)["f1"],
        })

    null_thresh_df = pd.DataFrame(_rows).round(3)

    mo.vstack([
        mo.md(
            f"### Performance at the untuned cutoff t = {NULL_THRESH}\n\n"
            "Any always-one-class rule scores `balanced_acc` = 0.5 exactly, so that is "
            "the no-information floor. `oracle_*` is the best balanced accuracy any "
            "cutoff could reach — a ceiling, not achievable without the label."
        ),
        mo.ui.table(null_thresh_df, selection=None),
    ])
    return (null_thresh_df,)


@app.cell
def _(latex_table, mo, null_thresh_df):
    """LaTeX table of the headline t = 0.5 numbers, in the repo's booktabs style.

    `latex_table` / `bold_best_cols` are reimplemented here rather than imported:
    the project has no installed package (`packages = []` in pyproject.toml), so
    the other notebooks each carry their own copy.

    Dataset names are rewritten to the paper's model naming (PPDoc-S/M), which also
    avoids the bare underscore in `ppdoc_s/m` that would break LaTeX.
    """
    _LATEX_DATASET_NAMES = {
        "Spiritualist (excl. ppdoc_s/m)": "Spiritualist (excl. PPDoc-S/M)",
    }

    _cols = {
        "recall_ocr": r"$\mathbf{R}_\mathbf{ocr}$",
        "recall_pars": r"$\mathbf{R}_\mathbf{pars}$",
        "balanced_acc": r"Balanced acc.",
        "oracle_balanced_acc": r"Oracle balanced acc.",
    }

    _tbl = null_thresh_df.copy()
    _tbl["dataset"] = _tbl["dataset"].astype(str).map(lambda n: _LATEX_DATASET_NAMES.get(n, n))
    _tbl = _tbl.set_index("dataset")[list(_cols)].rename(columns=_cols)
    _tbl.index.name = "Dataset"

    # No per-column bolding: the rows are datasets rather than competing methods,
    # so "best in column" would assert a cross-dataset comparison these numbers do
    # not support (base rates and corpora differ).
    latex_null_thresh = latex_table(
        _tbl.map(lambda v: f"{v:.3f}"),
        caption=(
            r"Bottleneck triage from the SpACER ratio $d_\text{ocr}/d_\text{total}$ at the "
            r"untuned cutoff $t = 0.5$, predicting whether OCR is the dominant error source "
            r"($d_\text{ocr} \ge d_\text{pars}$). Precision is reported per verdict: an "
            r"\emph{OCR} verdict is reliable on every dataset, whereas a \emph{parsing} verdict "
            r"is only reliable once degenerate parsers are removed. Balanced accuracy is used "
            r"in place of F1 because the classes are skewed; any always-one-class rule scores "
            r"0.500. The oracle column is the best balanced accuracy reachable by any cutoff, "
            r"which requires the label and so is not achievable in deployment."
        ),
        label="tab:bottleneck-triage",
        echo=False,
    )

    print(latex_null_thresh)
    mo.md(f"```latex\n{latex_null_thresh}\n```")
    return


@app.cell
def _(DATASETS, REPO_ROOT, VARIANTS, latex_table, mo, np, pd, prep_df):
    """Supplementary: COTe geometry vs per-verdict precision, one row per configuration.

    COTe components are averaged over the (page, parsing model) pairs in each configuration
    — over parsing outputs, not over the (page, parsing model, OCR model) rows used for
    precision, so a parser is not counted once per OCR model.

    Note COTe = C - O - T: Excess is reported but is not part of the composite score
    (cotescore.layout.cote_score).
    """
    _CONFIGS = [(_k, _v, None) for _k, _v in DATASETS.items()] + [
        (_k, _v[2], _v[1]) for _k, _v in VARIANTS.items()
    ]
    _BASE = {_k: _k for _k in DATASETS} | {_k: _v[0] for _k, _v in VARIANTS.items()}

    _rows = []
    for _key, _name, _dropped in _CONFIGS:
        _sub = prep_df[prep_df["dataset"] == _key]
        if _sub.empty:
            continue
        _cote = pd.read_parquet(REPO_ROOT / f"data/{_BASE[_key]}/cote_score_cache.parquet")
        _cote = _cote[_cote["parsing_model"] != "gt"]
        if _dropped:
            _cote = _cote[~_cote["parsing_model"].isin(_dropped)]

        _pred = _sub["ocr_over_total"].to_numpy() >= 0.5
        _label = _sub["ocr_dominant"].to_numpy(dtype=bool)
        _tp = int((_pred & _label).sum())
        _fp = int((_pred & ~_label).sum())
        _fn = int((~_pred & _label).sum())
        _tn = int((~_pred & ~_label).sum())

        _rows.append({
            "dataset": _name,
            "cote": _cote["cote"].mean(),
            "coverage": _cote["coverage"].mean(),
            "overlap": _cote["overlap"].mean(),
            "trespass": _cote["trespass"].mean(),
            "excess": _cote["excess"].mean(),
            "precision_ocr": _tp / (_tp + _fp) if _tp + _fp else np.nan,
            "precision_pars": _tn / (_tn + _fn) if _tn + _fn else np.nan,
        })

    cote_precision_df = (
        pd.DataFrame(_rows).sort_values("precision_pars", ascending=False).reset_index(drop=True)
    )

    _LATEX_NAMES = {"Spiritualist (excl. ppdoc_s/m)": "Spiritualist (excl. PPDoc-S/M)"}
    _COLS = {
        "cote": "COTe",
        "coverage": "Coverage",
        "overlap": "Overlap",
        "trespass": "Trespass",
        "excess": "Excess",
        "precision_ocr": r"Precision (OCR)",
        "precision_pars": r"Precision (parsing)",
    }
    _tbl = cote_precision_df.copy()
    _tbl["dataset"] = _tbl["dataset"].astype(str).map(lambda _n: _LATEX_NAMES.get(_n, _n))
    _tbl = _tbl.set_index("dataset")[list(_COLS)].rename(columns=_COLS)
    _tbl.index.name = "Dataset"

    latex_cote_precision = latex_table(
        _tbl.map(lambda _v: f"{_v:.3f}"),
        caption=(
            r"There is an inverse relationship between Trespass and Parsing Precision"
        ),
        label="tab:cote-precision",
        echo=False,
    )

    print(latex_cote_precision)
    mo.md(f"```latex\n{latex_cote_precision}\n```")
    return


@app.cell
def _(X_THRESH, Y_THRESH, np, pd):
    def f1_grid(df, x_thresh=X_THRESH, y_thresh=Y_THRESH):
        """F1 of (cote >= xt AND ocr_over_total >= yt) predicting ocr_dominant, for every (xt, yt)."""
        _x = df["cote"].to_numpy()
        _y = df["ocr_over_total"].to_numpy()
        _label = df["ocr_dominant"].to_numpy(dtype=bool)

        _rows = []
        for _xt in x_thresh:
            for _yt in y_thresh:
                _pred = (_x >= _xt) & (_y >= _yt)
                _tp = int((_pred & _label).sum())
                _fp = int((_pred & ~_label).sum())
                _fn = int((~_pred & _label).sum())
                _denom = 2 * _tp + _fp + _fn
                _rows.append({
                    "cote_thresh": _xt,
                    "ocr_over_total_thresh": _yt,
                    "n": int(_pred.sum()),
                    "tp": _tp,
                    "fp": _fp,
                    "fn": _fn,
                    "f1": float(2 * _tp / _denom) if _denom > 0 else np.nan,
                })
        _out = pd.DataFrame(_rows)
        _out["label"] = _out["f1"].map(lambda v: f"{v:.2f}").where(_out["f1"].notna(), "")
        return _out

    return (f1_grid,)


@app.cell
def _(f1_grid, pd, prep_df):
    heat_df = pd.concat(
        [
            f1_grid(_g).assign(dataset=_ds, dataset_name=_g["dataset_name"].iloc[0])
            for _ds, _g in prep_df.groupby("dataset", sort=False)
        ],
        ignore_index=True,
    )
    heat_df["dataset_name"] = pd.Categorical(
        heat_df["dataset_name"], categories=list(prep_df["dataset_name"].cat.categories)
    )
    return (heat_df,)


@app.cell
def _(X_THRESH, Y_THRESH, p9):
    _x_step = X_THRESH[1] - X_THRESH[0]
    _y_step = Y_THRESH[1] - Y_THRESH[0]

    def heatmap(df, title, figure_size=(8, 5), facet=False, text_size=7):
        _plt = (
            p9.ggplot(df, p9.aes(x="cote_thresh", y="ocr_over_total_thresh", fill="f1"))
            + p9.geom_tile(width=_x_step, height=_y_step)
            + p9.geom_text(p9.aes(label="label"), size=text_size)
            + p9.scale_x_continuous(breaks=X_THRESH, labels=[f"{v:.2f}" for v in X_THRESH])
            + p9.scale_y_continuous(breaks=Y_THRESH, labels=[f"{v:.2f}" for v in Y_THRESH])
            + p9.scale_fill_continuous(limits=(0, 1))
            + p9.labs(
                x="COTe score minimum threshold",
                y="SpACER d_ocr / d_total minimum threshold",
                fill="F1",
                title=title,
            )
            + p9.theme(
                figure_size=figure_size,
                axis_text_y=p9.element_text(size=12),
                axis_text_x=p9.element_text(rotation=45, hjust=1, size=12),
                strip_text=p9.element_text(size=14, weight="bold"),
            )
        )
        if facet:
            _plt = _plt + p9.facet_wrap("~dataset_name", nrow=2)
        return _plt

    return (heatmap,)


@app.cell
def _(DATASET_NAMES, FIG_DIR, heat_df, heatmap, mo):
    """One heatmap per dataset / variant, each saved to its own PDF."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    _plots = []
    for _ds, _name in DATASET_NAMES.items():
        _sub = heat_df[heat_df["dataset"] == _ds]
        if _sub.empty:
            continue
        _plt = heatmap(
            _sub,
            title=f"{_name}: is OCR the dominant error source? F1 of SpACER d_ocr/d_total and COTe thresholds",
        )
        _plt.save(filename=str(FIG_DIR / f"cutoff_thresholds_{_ds}.pdf"), dpi=300, verbose=False)
        _plots.append(mo.plain(_plt))

    mo.vstack(_plots)
    return


@app.cell
def _(FIG_DIR, heat_df, heatmap, mo):
    """Combined faceted heatmap across datasets and variants (shared colour scale)."""
    combined_plt = heatmap(
        heat_df,
        title="Is OCR the dominant error source? F1 of SpACER d_ocr/d_total and COTe thresholds",
        figure_size=(16, 11),
        facet=True,
        text_size=7,
    )
    combined_plt.save(filename=str(FIG_DIR / "cutoff_thresholds_all.pdf"), dpi=300, verbose=False)
    mo.plain(combined_plt)
    return


@app.cell
def _(heat_df, mo):
    """Best (x_thresh, y_thresh) per dataset, plus F1 of the trivial always-positive rule (0, 0)."""
    _best = heat_df.loc[heat_df.groupby("dataset", sort=False)["f1"].idxmax()]
    _baseline = heat_df[(heat_df["cote_thresh"] == 0) & (heat_df["ocr_over_total_thresh"] == 0)]

    best_df = (
        _best[["dataset_name", "cote_thresh", "ocr_over_total_thresh", "f1", "n", "tp", "fp", "fn"]]
        .merge(
            _baseline[["dataset_name", "f1"]].rename(columns={"f1": "f1_always_positive"}),
            on="dataset_name",
        )
        .rename(columns={"dataset_name": "dataset", "f1": "best_f1"})
        .round(3)
        .reset_index(drop=True)
    )

    mo.vstack([
        mo.md("### Best threshold pair per dataset"),
        mo.ui.table(best_df, selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()

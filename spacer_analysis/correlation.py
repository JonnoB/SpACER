"""Per-page correlations between the decomposition errors and layout metrics."""

from __future__ import annotations

import warnings

import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr

# Per-page layout metrics correlated against d_pars: (column label, source frame, column)
# mAP is deliberately absent: it is a corpus-level ranking metric with no per-page value.
DPARS_METRICS = {"COTe": "cote", "F1@0.5": "f1", "IoU": "iou"}
DPARS_ERRORS = {"spacer": "d_pars_spacer_macro", "cdd": "d_pars_cdd"}


def dpars_metric_spearman(
    results_df: pd.DataFrame, cote_df: pd.DataFrame, page_det_df: pd.DataFrame, decimals: int = 3
) -> dict[str, pd.DataFrame]:
    """Spearman rho between per-page d_pars and each layout metric, per parsing model.

    Returns ``{"spacer": df, "cdd": df}``; each frame has parsing models as rows
    (raw keys) and :data:`DPARS_METRICS` labels as columns. Negative rho is the
    expected direction (better layout -> lower parsing error). d_pars is
    OCR-model-independent, so it is averaged over OCR models per page first.
    """
    keys = ["page", "parsing_model"]
    merged = (
        results_df[results_df["parsing_model"] != "gt"]
        .groupby(keys)[list(DPARS_ERRORS.values())]
        .mean()
        .reset_index()
        .merge(cote_df[keys + ["cote"]], on=keys)
        .merge(page_det_df[keys + ["f1", "iou"]], on=keys)
    )
    out = {}
    for err_key, err_col in DPARS_ERRORS.items():
        rows = []
        for pm, grp in merged.groupby("parsing_model"):
            row = {"parsing_model": pm}
            for label, col in DPARS_METRICS.items():
                # A parser with F1 = 0 on every page has no defined rho: leave NaN.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConstantInputWarning)
                    rho, _ = spearmanr(grp[col], grp[err_col])
                row[label] = rho
            rows.append(row)
        out[err_key] = pd.DataFrame(rows).set_index("parsing_model").round(decimals)
    return out

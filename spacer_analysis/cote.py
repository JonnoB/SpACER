"""Per-page COTe scores for every parsing model, cached to parquet."""

from __future__ import annotations

import pandas as pd
from cotescore import GTBoxes, cote_score

from spacer_analysis.paths import DATASETS

COTE_COMPONENTS = ["cote", "coverage", "overlap", "trespass", "excess"]
COTE_LABELS = {"cote": "COTe", "coverage": "Coverage", "overlap": "Overlap",
               "trespass": "Trespass", "excess": "Excess"}


def compute_cote_df(
    name: str, bbox_df: pd.DataFrame, parsing_models, use_cache: bool = True
) -> pd.DataFrame:
    """COTe + components per (page, parsing_model), read from / written to the dataset cache.

    Uses cotescore's analytic bounding-box path (GTBoxes + an (M, 4) box
    array), which is exact rather than raster-resolution-limited. The cache
    is what the cross-dataset notebooks read, so all of them see the same
    numbers.
    """
    p = DATASETS[name]
    if use_cache and p.cote_cache.exists():
        return pd.read_parquet(p.cote_cache)

    gt = pd.read_csv(p.gt_ssu_bboxes).copy()
    gt["ssu_int"] = pd.factorize(gt["ssu_id"])[0] + 1

    records = []
    for filename, gt_page in gt.groupby("filename"):
        gt_boxes = GTBoxes(
            boxes=gt_page[["x", "y", "width", "height"]].to_numpy(dtype=float),
            ssu_ids=gt_page["ssu_int"].to_numpy(dtype=int),
            image_width=int(gt_page["image_width"].iloc[0]),
            image_height=int(gt_page["image_height"].iloc[0]),
        )
        for pm in parsing_models:
            pred_page = bbox_df.loc[
                (bbox_df["parsing_model"] == pm) & (bbox_df["filename"] == filename)
            ]
            preds = pred_page[["x", "y", "width", "height"]].to_numpy(dtype=float)
            c, cov, ov, tr, ex = cote_score(gt_boxes, preds)
            records.append({
                "page": p.page_id(filename), "parsing_model": pm,
                "cote": c, "coverage": cov, "overlap": ov, "trespass": tr, "excess": ex,
            })
    cote_df = pd.DataFrame(records)
    if use_cache:
        p.cote_cache.parent.mkdir(parents=True, exist_ok=True)
        cote_df.to_parquet(p.cote_cache)
    return cote_df


def mean_cote_table(cote_df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Mean COTe + components per parsing model (gt row dropped), raw model keys as index."""
    return (
        cote_df[cote_df["parsing_model"] != "gt"]
        .groupby("parsing_model")[COTE_COMPONENTS]
        .mean()
        .round(decimals)
    )

"""Single-class box-detection metrics (mAP, F1, IoU) against GT SSU boxes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from cotescore.map_metric import MAPMetric

from spacer_analysis.paths import DATASETS

BOX_COLS = ["x", "y", "width", "height"]


def iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xywh box arrays -> (N, M)."""
    def to_xyxy(b):
        return np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]])
    p, g = to_xyxy(pred_boxes), to_xyxy(gt_boxes)
    inter_x1 = np.maximum(p[:, None, 0], g[None, :, 0])
    inter_y1 = np.maximum(p[:, None, 1], g[None, :, 1])
    inter_x2 = np.minimum(p[:, None, 2], g[None, :, 2])
    inter_y2 = np.minimum(p[:, None, 3], g[None, :, 3])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def coco_match(iou: np.ndarray, confidence: np.ndarray, thresh: float = 0.5) -> list[float]:
    """COCO per-image matching: predictions in descending confidence, each takes the
    highest-IoU GT box not already claimed. Returns the IoU of every match."""
    matched_gt: list[int] = []
    matched_iou: list[float] = []
    for pi in np.argsort(-confidence, kind="stable"):
        row = iou[pi].copy()
        row[matched_gt] = -1.0
        gi = int(row.argmax())
        if row[gi] >= thresh:
            matched_gt.append(gi)
            matched_iou.append(float(row[gi]))
    return matched_iou


def page_detection_metrics(
    name: str, bbox_df: pd.DataFrame, parsing_models
) -> pd.DataFrame:
    """Per-(page, parsing_model) F1@0.5 and matched IoU, plus the counts behind them.

    Matching is COCO's (see :func:`coco_match`) at IoU >= 0.5. Columns:
    ``page, parsing_model, n_pred, n_gt, n_matched, iou_sum, f1, iou`` where
    ``iou`` is the mean IoU of that page's matched boxes (0.0 when nothing
    matched, i.e. no box was usable) and ``f1 = 2TP / (|pred| + |gt|)``.
    """
    paths = DATASETS[name]
    gt = pd.read_csv(paths.gt_ssu_bboxes)

    records = []
    for pm in [pm for pm in parsing_models if pm != "gt"]:
        for filename, gt_page in gt.groupby("filename"):
            pred_page = bbox_df.loc[
                (bbox_df["parsing_model"] == pm) & (bbox_df["filename"] == filename)
            ]
            ious: list[float] = []
            if len(pred_page):
                iou = iou_matrix(
                    pred_page[BOX_COLS].to_numpy(dtype=float),
                    gt_page[BOX_COLS].to_numpy(dtype=float),
                )
                ious = coco_match(iou, pred_page["confidence"].to_numpy(dtype=float))
            records.append({
                "page": paths.page_id(filename),
                "parsing_model": pm,
                "n_pred": len(pred_page),
                "n_gt": len(gt_page),
                "n_matched": len(ious),
                "iou_sum": float(sum(ious)),
                # F1 = 2TP / (2TP + FP + FN) = 2TP / (|pred| + |gt|)
                "f1": 2 * len(ious) / (len(pred_page) + len(gt_page)),
                "iou": float(np.mean(ious)) if ious else 0.0,
            })
    return pd.DataFrame(records)


def detection_metrics(
    name: str,
    bbox_df: pd.DataFrame,
    parsing_models,
    decimals: int = 3,
    per_page: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """mAP, mAP@0.5, F1@0.5 and mean matched IoU per parsing model, raw model keys as index.

    mAP is COCO-style (cotescore.MAPMetric, pycocotools underneath): predictions
    ranked by confidence, pooled over pages, averaged over IoU 0.50:0.05:0.95.
    F1@0.5 and IoU use the same COCO matching at IoU >= 0.5: F1 is
    macro-averaged over pages, IoU is the mean over all matched pairs.

    Predictions were exported with per-model confidence floors (heron 0.6,
    ppdoc 0.5, yolo 0.2), which truncate the low-confidence tail of each
    precision-recall curve.

    ``per_page`` may be passed from :func:`page_detection_metrics` to avoid recomputing it.
    """
    gt = pd.read_csv(DATASETS[name].gt_ssu_bboxes)
    if per_page is None:
        per_page = page_detection_metrics(name, bbox_df, parsing_models)

    def to_dicts(df, with_conf):
        cols = BOX_COLS + (["confidence"] if with_conf else [])
        return [{**r, "class": "ssu"} for r in df[cols].to_dict("records")]

    records = []
    for pm in [pm for pm in parsing_models if pm != "gt"]:
        ap_metric = MAPMetric()
        for filename, gt_page in gt.groupby("filename"):
            pred_page = bbox_df.loc[
                (bbox_df["parsing_model"] == pm) & (bbox_df["filename"] == filename)
            ]
            ap_metric.update(to_dicts(pred_page, True), to_dicts(gt_page, False))
        ap = ap_metric.compute()
        pp = per_page[per_page["parsing_model"] == pm]
        n_matched = pp["n_matched"].sum()
        records.append({
            "parsing_model": pm,
            "mAP": ap["map"],
            "mAP@0.5": ap["map_50"],
            "F1@0.5": float(pp["f1"].mean()),
            "IoU": float(pp["iou_sum"].sum() / n_matched) if n_matched else 0.0,
        })
    return pd.DataFrame(records).set_index("parsing_model").round(decimals)

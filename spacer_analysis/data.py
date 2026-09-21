"""Load a dataset's inferred characters, OCR outputs and predicted boxes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from spacer_analysis.paths import DATASETS, DatasetPaths


@dataclass
class DatasetFrames:
    paths: DatasetPaths
    chars_df: pd.DataFrame      # char_text, cx, cy, ssu_id, page_id, ...
    gt_ocr_df: pd.DataFrame     # ocr_model, filename, ssu_id, ocr_text
    pred_ocr_df: pd.DataFrame   # parsing_model, ocr_model, filename, x, y, width, height, ocr_text
    bbox_df: pd.DataFrame       # parsing_model, filename, x, y, width, height, confidence, ...
    parsing_models: list[str]
    ocr_models: list[str]
    pages: list[str]


def load_dataset(name: str) -> DatasetFrames:
    p = DATASETS[name]
    prefix = f"{name}_"

    chars_df = pd.read_parquet(p.chars)
    chars_df = chars_df[chars_df["char_text"] != " "].reset_index(drop=True)
    chars_df["cx"] = (chars_df["x"] + chars_df["w"] / 2).astype(int)
    chars_df["cy"] = (chars_df["y"] + chars_df["h"] / 2).astype(int)

    # GT OCR parquets (parsing_model == "gt"): {name}_gt_predictions_{ocr_model}_ocr.parquet
    gt_parts = []
    for f in sorted(p.ocr_dir.glob(f"{prefix}gt_predictions_*_ocr.parquet")):
        om = f.stem.removeprefix(f"{prefix}gt_predictions_").removesuffix("_ocr")
        part = pd.read_parquet(f)
        part["ocr_model"] = om
        gt_parts.append(part)
    gt_ocr_df = pd.concat(gt_parts, ignore_index=True) if gt_parts else pd.DataFrame()

    # Prediction OCR parquets: {name}_{parsing_model}_predictions_{ocr_model}_ocr.parquet
    pred_parts = []
    for f in sorted(p.ocr_dir.glob("*_ocr.parquet")):
        inner = f.stem.removeprefix(prefix).removesuffix("_ocr")
        sep = inner.index("_predictions_")
        pm = inner[:sep]
        if pm == "gt":
            continue
        om = inner[sep + len("_predictions_"):]
        part = pd.read_parquet(f)
        part["parsing_model"] = pm
        part["ocr_model"] = om
        pred_parts.append(part)
    pred_ocr_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()

    # Predicted boxes: {name}_{parsing_model}_predictions.csv
    bbox_parts = []
    for f in sorted(p.bbox_dir.glob("*.csv")):
        pm = f.stem.removeprefix(prefix).removesuffix("_predictions")
        part = pd.read_csv(f)
        part["parsing_model"] = pm
        bbox_parts.append(part)
    bbox_df = pd.concat(bbox_parts, ignore_index=True)

    # Non-gt prediction CSVs may carry the full GT box set (source == "gt");
    # score only the real predictions. The gt-baseline file has no source
    # column (NaN after concat), so it is preserved.
    if "source" in bbox_df.columns:
        bbox_df = bbox_df[bbox_df["source"] != "gt"].reset_index(drop=True)

    return DatasetFrames(
        paths=p,
        chars_df=chars_df,
        gt_ocr_df=gt_ocr_df,
        pred_ocr_df=pred_ocr_df,
        bbox_df=bbox_df,
        parsing_models=sorted(bbox_df["parsing_model"].unique()),
        ocr_models=sorted(pred_ocr_df["ocr_model"].unique()) if not pred_ocr_df.empty else [],
        pages=sorted(chars_df["page_id"].unique()),
    )

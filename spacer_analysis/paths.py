"""Dataset path registry: one place that knows where each dataset's files live."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Display order for cross-dataset tables and figures.
DATASET_ORDER = ["spiritualist", "hiertext", "docbank"]


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    display: str
    chars: Path                   # characters_inferred.parquet
    ocr_dir: Path                 # {name}_{parsing}_predictions_{ocr}_ocr.parquet
    bbox_dir: Path                # {name}_{parsing}_predictions.csv
    gt_ssu_bboxes: Path
    cote_cache: Path
    decomposition_results: Path
    box_level: Path               # box_level_ocr_comparison.parquet
    page_level: Path              # page_level_cer_comparison.parquet
    image_suffix: str = ".jpg"    # filename == page_id + image_suffix

    def page_id(self, filename: str) -> str:
        return Path(filename).name.removesuffix(self.image_suffix)

    def filename(self, page_id: str) -> str:
        return f"{page_id}{self.image_suffix}"


def _d(name, display, base, *, ocr_dir, bbox_dir, page_level, image_suffix=".jpg"):
    b = REPO_ROOT / base
    return DatasetPaths(
        name=name,
        display=display,
        chars=b / "characters_inferred.parquet",
        ocr_dir=REPO_ROOT / ocr_dir,
        bbox_dir=REPO_ROOT / bbox_dir,
        gt_ssu_bboxes=b / "gt_ssu_bboxes.csv",
        cote_cache=b / "cote_score_cache.parquet",
        decomposition_results=b / "decomposition_results.parquet",
        box_level=b / "box_level_ocr_comparison.parquet",
        page_level=REPO_ROOT / page_level,
        image_suffix=image_suffix,
    )


DATASETS: dict[str, DatasetPaths] = {
    "spiritualist": _d(
        "spiritualist", "Spiritualist", "data/spiritualist",
        ocr_dir="data/results_spiritualist/ocr",
        bbox_dir="data/results_spiritualist/bboxes",
        page_level="data/results_spiritualist/page_level_cer_comparison.parquet",
    ),
    "hiertext": _d(
        "hiertext", "HierText", "data/hiertext",
        ocr_dir="data/hiertext_results/ocr",
        bbox_dir="data/hiertext_predictions",
        page_level="data/hiertext/page_level_cer_comparison.parquet",
    ),
    "docbank": _d(
        "docbank", "DocBank", "data/docbank",
        ocr_dir="data/docbank/ocr",
        bbox_dir="data/docbank/bbox_preds",
        page_level="data/docbank/page_level_cer_comparison.parquet",
        image_suffix="_ori.jpg",
    ),
}

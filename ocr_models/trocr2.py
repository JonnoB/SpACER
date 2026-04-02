from __future__ import annotations

from PIL import Image

from ocr_models.trocr import TrOCROCR, _tesseract_split_lines, _MIN_LINE_HEIGHT, _LINE_PADDING
from scripts.crop_utils import lines_from_staircase_polygon


def _polygon_split_lines(crop: Image.Image, points_str: str) -> list[Image.Image]:
    """Produce line sub-crops from a staircase polygon string.

    Recovers the original line bounding boxes from the polygon encoding and
    returns one padded sub-crop per line. Coordinates in points_str are
    absolute (full-image space); the crop's origin is subtracted to convert
    to crop-local coordinates.

    Falls back to returning the full crop if the polygon cannot be decoded.
    """
    pts = [(int(p.split(",")[0]), int(p.split(",")[1])) for p in points_str.strip().split()]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    origin_x = max(0, min(xs))
    origin_y = max(0, min(ys))

    lines = lines_from_staircase_polygon(points_str)
    if not lines:
        return [crop]

    sub_crops = []
    for hpos, vpos, width, height in lines:
        top    = max(0,           (vpos          - origin_y) - _LINE_PADDING)
        bottom = min(crop.height, (vpos + height - origin_y) + _LINE_PADDING)
        left   = max(0,            hpos          - origin_x)
        right  = min(crop.width,   hpos + width  - origin_x)
        if bottom - top < _MIN_LINE_HEIGHT or right <= left:
            continue
        sub_crops.append(crop.crop((left, top, right, bottom)))

    return sub_crops if sub_crops else [crop]


class TrOCR2OCR(TrOCROCR):
    """TrOCR variant that uses staircase-polygon geometry for line splitting.

    When a crop has an associated polygon_points string (passed via metadata),
    line sub-crops are derived directly from the polygon encoding instead of
    running Tesseract layout analysis. This avoids Tesseract over-detecting
    lines on large polygon-masked newspaper columns.

    Falls back to Tesseract line splitting (identical to TrOCROCR) when no
    polygon_points is available, so the model works correctly on predicted
    bounding boxes that have no polygon.
    """

    def prepare(self, crops: list, metadata: list | None = None) -> tuple[list[Image.Image], list[int]]:
        if not self._split_lines:
            return (list(crops), [1] * len(crops))

        expanded: list[Image.Image] = []
        split_counts: list[int] = []
        for i, crop in enumerate(crops):
            pp = metadata[i] if metadata and i < len(metadata) else None
            if pp and isinstance(pp, str) and pp.strip():
                lines = _polygon_split_lines(crop, pp)
            else:
                lines = _tesseract_split_lines(crop)
            expanded.extend(lines)
            split_counts.append(len(lines))
        return (expanded, split_counts)

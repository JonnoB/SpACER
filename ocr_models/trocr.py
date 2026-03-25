from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from ocr_models.base import OCRModel


# Minimum sub-crop height to pass to TrOCR — avoids feeding blank slivers.
_MIN_LINE_HEIGHT = 10

# Padding (in pixels) added above/below each detected line box to avoid
# clipping ascenders/descenders.
_LINE_PADDING = 2


def _tesseract_split_lines(crop: Image.Image) -> list[Image.Image]:
    """Split a multi-line crop into individual line crops using Tesseract's
    layout analysis.

    Uses pytesseract.image_to_data to get line-level bounding boxes from
    Tesseract's segmentation, then returns one sub-crop per detected line,
    sorted top-to-bottom. Falls back to the original crop if no lines are found.
    """
    import pytesseract

    data = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DATAFRAME)

    # Filter out rows with no confidence value (separators/noise) and
    # rows that are not at word level (level 5 = word in Tesseract hierarchy).
    # We group words up to line level (block_num + par_num + line_num).
    words = data[data["conf"] != -1].copy()

    if words.empty:
        return [crop]

    # Compute right/bottom edges before grouping
    words["right"] = words["left"] + words["width"]
    words["bottom"] = words["top"] + words["height"]

    lines = (
        words.groupby(["block_num", "par_num", "line_num"], sort=False)
        .agg(
            left=("left", "min"),
            top=("top", "min"),
            right=("right", "max"),
            bottom=("bottom", "max"),
        )
        .reset_index(drop=True)
        .sort_values("top")
    )

    sub_crops = []
    for _, row in lines.iterrows():
        top = max(0, int(row["top"]) - _LINE_PADDING)
        bottom = min(crop.height, int(row["bottom"]) + _LINE_PADDING)
        if bottom - top < _MIN_LINE_HEIGHT:
            continue
        sub_crops.append(crop.crop((0, top, crop.width, bottom)))

    return sub_crops if sub_crops else [crop]


class TrOCROCR(OCRModel):
    """OCR backend using Microsoft TrOCR via HuggingFace transformers.

    TrOCR is a single-line model. When split_lines=True (default), crops
    taller than the height threshold are automatically split into individual
    line sub-crops using Tesseract's layout analysis. Each line is run through
    TrOCR independently and the results are joined with a newline.
    """

    DEFAULT_MODEL = "microsoft/trocr-base-printed"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        split_lines: bool = True,
        batch_size: int = 8,
        **kwargs,
    ):
        self._model_name = model_name
        self._device = device
        self._split_lines = split_lines
        self._batch_size = batch_size
        self._processor = None
        self._model = None

    def load(self) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._processor = TrOCRProcessor.from_pretrained(self._model_name, use_fast=True)
        self._model = VisionEncoderDecoderModel.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)
        self._model.eval()

    def run(self, crop: Image.Image) -> str:
        return self.run_batch([crop])[0]

    def prepare(self, crops: list) -> tuple[list[Image.Image], list[int]]:
        """CPU phase: split each crop into individual text lines using Tesseract.

        Returns (expanded_crops, split_counts) where split_counts records how
        many line sub-crops came from each original crop so they can be
        reassembled after inference.
        """
        if not self._split_lines:
            return (list(crops), [1] * len(crops))
        expanded: list[Image.Image] = []
        split_counts: list[int] = []
        for crop in crops:
            lines = _tesseract_split_lines(crop)
            expanded.extend(lines)
            split_counts.append(len(lines))
        return (expanded, split_counts)

    def run_prepared(self, prepared: tuple) -> list:
        """GPU phase: run TrOCR on pre-split line crops and reassemble."""
        import torch

        expanded, split_counts = prepared
        rgb_crops = [c.convert("RGB") for c in expanded]
        line_texts = []
        for i in range(0, len(rgb_crops), self._batch_size):
            batch = rgb_crops[i : i + self._batch_size]
            pixel_values = self._processor(images=batch, return_tensors="pt").pixel_values.to(self._device)
            with torch.no_grad():
                generated_ids = self._model.generate(pixel_values)
            line_texts.extend(
                t.strip()
                for t in self._processor.batch_decode(generated_ids, skip_special_tokens=True)
            )

        results, idx = [], 0
        for count in split_counts:
            results.append("\n".join(t for t in line_texts[idx : idx + count] if t))
            idx += count
        return results

    def run_batch(self, crops: list) -> list:
        return self.run_prepared(self.prepare(crops))
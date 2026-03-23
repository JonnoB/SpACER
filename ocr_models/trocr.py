from __future__ import annotations

import numpy as np
from PIL import Image

from ocr_models.base import OCRModel

# Crops taller than this (in pixels) are candidates for line splitting.
_LINE_SPLIT_HEIGHT_THRESHOLD = 60

# Minimum sub-crop height to pass to TrOCR — avoids feeding blank slivers.
_MIN_LINE_HEIGHT = 10


def _split_lines(crop: Image.Image) -> list[Image.Image]:
    """Split a multi-line crop into individual line crops using a horizontal
    projection profile.

    Converts the crop to grayscale, inverts it so ink is bright, computes a
    row-wise sum (the projection), then finds valley rows (low ink density)
    that separate text lines. Returns a list of sub-crops, one per line.
    If no clear split is found, returns the original crop in a list.
    """
    gray = np.array(crop.convert("L"), dtype=np.float32)
    # Invert: ink becomes high values, background becomes low
    inverted = 255.0 - gray
    row_profile = inverted.sum(axis=1)

    # Smooth with a small window to reduce noise
    window = max(3, crop.height // 40)
    kernel = np.ones(window) / window
    smoothed = np.convolve(row_profile, kernel, mode="same")

    # Threshold: rows with total ink below this are considered gaps
    ink_threshold = smoothed.max() * 0.15

    in_text = False
    regions = []
    start = 0
    for i, val in enumerate(smoothed):
        if not in_text and val > ink_threshold:
            in_text = True
            start = i
        elif in_text and val <= ink_threshold:
            in_text = False
            regions.append((start, i))
    if in_text:
        regions.append((start, len(smoothed)))

    if len(regions) <= 1:
        return [crop]

    sub_crops = []
    for top, bottom in regions:
        if bottom - top < _MIN_LINE_HEIGHT:
            continue
        sub_crops.append(crop.crop((0, top, crop.width, bottom)))

    return sub_crops if sub_crops else [crop]


class TrOCROCR(OCRModel):
    """OCR backend using Microsoft TrOCR via HuggingFace transformers.

    TrOCR is a single-line model. When split_lines=True (default), crops
    taller than the height threshold are automatically split into individual
    line sub-crops using a horizontal projection profile. Each line is run
    through TrOCR independently and the results are joined with a space.
    """

    DEFAULT_MODEL = "microsoft/trocr-base-printed"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        split_lines: bool = True,
        **kwargs,
    ):
        self._model_name = model_name
        self._device = device
        self._split_lines = split_lines
        self._processor = None
        self._model = None

    def load(self) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self._processor = TrOCRProcessor.from_pretrained(self._model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)
        self._model.eval()

    def run(self, crop: Image.Image) -> str:
        return self.run_batch([crop])[0]

    def run_batch(self, crops: list) -> list:
        import torch

        if self._split_lines:
            # Expand crops that need line splitting; track how to reassemble
            expanded: list[Image.Image] = []
            split_counts: list[int] = []
            for crop in crops:
                if crop.height > _LINE_SPLIT_HEIGHT_THRESHOLD:
                    lines = _split_lines(crop)
                else:
                    lines = [crop]
                expanded.extend(lines)
                split_counts.append(len(lines))
        else:
            expanded = crops
            split_counts = [1] * len(crops)

        # Run TrOCR on all sub-crops in one batched pass
        rgb_crops = [c.convert("RGB") for c in expanded]
        pixel_values = self._processor(images=rgb_crops, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)
        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values)
        line_texts = [
            t.strip()
            for t in self._processor.batch_decode(generated_ids, skip_special_tokens=True)
        ]

        # Reassemble: join lines belonging to the same original crop
        results = []
        idx = 0
        for count in split_counts:
            chunk = line_texts[idx : idx + count]
            results.append(" ".join(t for t in chunk if t))
            idx += count
        return results

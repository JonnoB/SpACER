from __future__ import annotations

import numpy as np
from PIL import Image

from ocr_models.base import OCRModel

# Crops taller than this (in pixels) are candidates for line splitting.
_LINE_SPLIT_HEIGHT_THRESHOLD = 60

# Minimum sub-crop height to pass to TrOCR — avoids feeding blank slivers.
_MIN_LINE_HEIGHT = 10


def _easyocr_split_lines(crop: Image.Image, reader) -> list[Image.Image]:
    """Split a multi-line crop into individual line crops using EasyOCR's CRAFT detection.

    Returns a list of sub-crops sorted top-to-bottom, one per detected line.
    If no boxes are detected, returns the original crop in a list.
    """
    img_array = np.array(crop.convert("RGB"))
    # detect() returns (horizontal_list, free_list); each is a list-per-image.
    # horizontal boxes are [x_min, x_max, y_min, y_max].
    horizontal_list, _ = reader.detect(img_array)
    boxes = horizontal_list[0] if horizontal_list else []

    if not boxes:
        return [crop]

    regions = []
    for box in boxes:
        y_min = max(0, int(box[2]))
        y_max = min(crop.height, int(box[3]))
        if y_max - y_min < _MIN_LINE_HEIGHT:
            continue
        regions.append((y_min, y_max))

    if not regions:
        return [crop]

    regions.sort()
    return [crop.crop((0, top, crop.width, bottom)) for top, bottom in regions]


class TrOCROCR(OCRModel):
    """OCR backend using Microsoft TrOCR via HuggingFace transformers.

    TrOCR is a single-line model. When split_lines=True (default), crops
    taller than the height threshold are automatically split into individual
    line sub-crops using CRAFT text detection. Each line is run through TrOCR
    independently and the results are joined with a space.
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
        self._craft = None

    def load(self) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        if self._split_lines:
            import easyocr
            self._craft = easyocr.Reader(["en"], gpu=False)

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
                    lines = _easyocr_split_lines(crop, self._craft)
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

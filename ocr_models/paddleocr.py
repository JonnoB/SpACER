from __future__ import annotations

import numpy as np
from PIL import Image

from ocr_models.base import OCRModel


class PaddleOCROCR(OCRModel):
    """OCR backend using PaddleOCR >= 3.0. Requires paddlepaddle + paddleocr installed.

    PaddleOCR 3.x replaced use_gpu/use_angle_cls with a device parameter.
    Set device='gpu' for GPU inference, device='cpu' for CPU.
    """

    def __init__(self, lang: str = "en", device: str = "cpu:0", **kwargs):
        self._lang = lang
        self._device = device
        self._extra = kwargs
        self._engine = None

    def load(self) -> None:
        import paddle
        try:
            paddle.set_flags({"FLAGS_use_mkldnn": False})
        except Exception:
            pass

        from paddleocr import PaddleOCR
        self._engine = PaddleOCR(lang=self._lang, device=self._device, **self._extra)

    def run(self, crop: Image.Image) -> str:
        img_array = np.array(crop.convert("RGB"))
        result = self._engine.ocr(img_array)
        if not result or not result[0]:
            return ""
        texts = [line[1][0] for line in result[0] if line and line[1]]
        return " ".join(texts).strip()

    def run_batch(self, crops: list) -> list:
        arrays = [np.array(c.convert("RGB")) for c in crops]
        results = self._engine.ocr(arrays)
        texts = []
        for page_result in (results or []):
            if not page_result:
                texts.append("")
                continue
            page_texts = [line[1][0] for line in page_result if line and line[1]]
            texts.append(" ".join(page_texts).strip())
        while len(texts) < len(crops):
            texts.append("")
        return texts

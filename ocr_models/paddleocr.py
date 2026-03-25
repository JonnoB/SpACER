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
        from paddleocr import PaddleOCR
        self._engine = PaddleOCR(
            lang=self._lang,
            device=self._device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **self._extra,
        )

    def run(self, crop: Image.Image) -> str:
        img_array = np.array(crop.convert("RGB"))
        result = self._engine.predict(img_array)
        texts = []
        for res in result:
            if res and res.get("rec_texts"):
                texts.extend(res["rec_texts"])
        return " ".join(t for t in texts if t).strip()

    def run_batch(self, crops: list) -> list:
        return [self.run(crop) for crop in crops]

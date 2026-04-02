from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from PIL import Image


class OCRModel(ABC):
    """Base class for all OCR model backends."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Called once before inference."""

    @abstractmethod
    def run(self, crop: Image.Image) -> str:
        """Run OCR on a PIL image crop. Returns extracted text string."""

    def run_batch(self, crops: list) -> list:
        """Run OCR on a list of crops. Override for GPU-efficient batching."""
        return [self.run(crop) for crop in crops]

    def prepare(self, crops: list, metadata: list | None = None) -> Any:
        """CPU pre-processing step (e.g. line splitting). Default: identity.

        Designed to run on a background thread so CPU work can overlap with
        GPU inference on the previous batch. Override in models that have a
        distinct CPU pre-processing phase.

        Args:
            crops:    Per-region PIL image crops.
            metadata: Optional per-crop metadata (e.g. polygon_points strings).
                      Ignored by the default implementation.
        """
        return crops

    def run_prepared(self, prepared: Any) -> list:
        """Run inference on pre-processed data returned by prepare().

        Default delegates to run_batch(), so models that don't override
        prepare() work unchanged.
        """
        return self.run_batch(prepared)


class MockOCR(OCRModel):
    """Test stub — returns a fixed string regardless of input."""

    def __init__(self, return_text: str = "mock text"):
        self.return_text = return_text
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def run(self, crop: Image.Image) -> str:
        return self.return_text

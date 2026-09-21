"""Model display names for tables and figures."""

# lowercase, underscores stripped -> display label
MODEL_DISPLAY_NAMES = {
    # OCR models
    "trocr": "TrOCR",
    "paddleocr": "PaddleOCR",
    "tesseract": "Tesseract",
    "craft": "CRAFT",
    "easyocr": "EasyOCR",
    # Parsing models
    "heron": "Heron",
    "ppdocl": "PPDoc-L",
    "ppdocm": "PPDoc-M",
    "ppdocs": "PPDoc-S",
    "yolo": "YOLO",
}


def display_name(name: str) -> str:
    """Map a raw model key (e.g. ``ppdoc_l``) to its display label (``PPDoc-L``)."""
    lower = name.lower().replace("_", "")
    return MODEL_DISPLAY_NAMES.get(lower, name.replace("_", "-").title())

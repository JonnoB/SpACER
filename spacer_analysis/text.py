"""Text normalisation applied to GT and OCR output before CER."""

import re
import unicodedata

_SINGLE_QUOTES = r"[‘’‚‛‹›`]"
_DOUBLE_QUOTES = r"[“”„‟«»]"
_DASHES = r"[–—―‒]"


def normalize_for_cer(text: str) -> str:
    """Lowercase, NFKC, fold typographic quotes/dashes, collapse whitespace.

    Single newlines become spaces (line wraps); blank lines are preserved.
    """
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(_SINGLE_QUOTES, "'", text)
    text = re.sub(_DOUBLE_QUOTES, '"', text)
    text = re.sub(_DASHES, "-", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

"""Internal provider helpers."""

import re


def estimate_tokens(text: str) -> int:
    """
    Fast token estimation using character-based heuristic.

    Rule of thumb: ~4 characters per token for English text.
    """
    if not text:
        return 0

    words = len(re.findall(r"\w+", text))
    special_chars = len(re.findall(r"[^\w\s]", text))

    return int(words * 1.3 + special_chars * 0.5)

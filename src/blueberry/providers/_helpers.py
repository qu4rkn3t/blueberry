"""Internal provider helpers."""

import re


def get_tiktoken_tokenizer(model: str = "gpt-4"):
    """
    Get tiktoken tokenizer for OpenAI-compatible models.

    Returns None if tiktoken is not installed.
    """
    try:
        import tiktoken
        # Try to get exact encoding for the model
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to cl100k_base (GPT-4, GPT-3.5-turbo default)
            return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


def count_tokens_with_tiktoken(text: str, model: str = "gpt-4") -> int | None:
    """
    Count tokens using tiktoken.

    Returns None if tiktoken is not available.
    """
    tokenizer = get_tiktoken_tokenizer(model)
    if tokenizer:
        return len(tokenizer.encode(text))
    return None


def estimate_tokens(text: str) -> int:
    """
    Fast token estimation using character-based heuristic.

    Fallback for when proper tokenizer is unavailable.
    Rule of thumb: ~4 characters per token for English text.
    """
    if not text:
        return 0

    words = len(re.findall(r"\w+", text))
    special_chars = len(re.findall(r"[^\w\s]", text))

    return int(words * 1.3 + special_chars * 0.5)

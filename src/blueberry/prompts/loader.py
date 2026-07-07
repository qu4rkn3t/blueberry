"""Prompt loading utilities."""

from pathlib import Path


def load_prompt(source: str | Path) -> str:
    """
    Load a prompt from a string or file.

    Args:
        source: Direct prompt string or file path

    Returns:
        The prompt text
    """
    source_str = str(source)

    # Check if it looks like a file path
    path = Path(source_str)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8")

    # Otherwise treat as direct prompt
    return source_str

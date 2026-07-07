"""Prompt optimization through external services."""

from typing import Any


def optimize_prompt(
    prompt: str,
    optimizer: str = "headroom",
    api_key: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Optimize a prompt using external optimization services.

    Args:
        prompt: The prompt to optimize
        optimizer: Optimizer to use ("headroom", "promptlayer", etc.)
        api_key: API key for the optimizer service
        **kwargs: Additional optimizer-specific parameters

    Returns:
        Optimized prompt

    Examples:
        >>> prompt = "Explain quantum computing"
        >>> optimized = optimize_prompt(prompt, optimizer="headroom", api_key="...")
    """
    if optimizer == "headroom":
        return _optimize_headroom(prompt, api_key, **kwargs)
    elif optimizer == "none" or optimizer is None:
        return prompt
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


def _optimize_headroom(prompt: str, api_key: str | None, **kwargs: Any) -> str:
    """
    Optimize prompt using Headroom.

    Headroom compresses prompts while preserving semantic meaning.
    https://headroom.dev
    """
    if not api_key:
        raise ValueError("Headroom requires an API key")

    try:
        import httpx
    except ImportError:
        raise ImportError("httpx required for Headroom: pip install httpx")

    # Headroom API endpoint (adjust based on actual API)
    url = kwargs.get("endpoint", "https://api.headroom.dev/v1/optimize")

    response = httpx.post(
        url,
        json={"prompt": prompt, **kwargs},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()

    result = response.json()
    return result.get("optimized_prompt", prompt)

"""Prompt optimization through external services."""

from pathlib import Path
from typing import Any


def optimize_prompt(
    prompt: str,
    optimizer: str = "headroom",
    api_key: str | None = None,
    endpoint: str | None = None,
    config_path: str | Path | None = None,
    **kwargs: Any,
) -> str:
    """
    Optimize a prompt using external optimization services.

    Args:
        prompt: The prompt to optimize
        optimizer: Optimizer to use ("headroom", etc.)
        api_key: API key for the optimizer service (overrides config)
        endpoint: Endpoint URL (overrides config)
        config_path: Path to config file containing optimizer settings
        **kwargs: Additional optimizer-specific parameters

    Returns:
        Optimized prompt
    """
    if config_path:
        from blueberry.config import get_optimizer_config, load_config

        config = load_config(config_path)
        optimizer_config = get_optimizer_config(config, optimizer)
        api_key = api_key or optimizer_config.api_key
        endpoint = endpoint or optimizer_config.endpoint
        kwargs = {**optimizer_config.metadata, **kwargs}

    if optimizer == "headroom":
        return _optimize_headroom(prompt, api_key, endpoint, **kwargs)
    elif optimizer == "none" or optimizer is None:
        return prompt
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


def _optimize_headroom(
    prompt: str, api_key: str | None, endpoint: str | None, **kwargs: Any
) -> str:
    """
    Optimize prompt using Headroom.

    Headroom compresses prompts while preserving semantic meaning.
    """
    if not api_key:
        raise ValueError("Headroom requires an API key")

    if not endpoint:
        raise ValueError("Headroom endpoint must be configured")

    try:
        import httpx
    except ImportError:
        raise ImportError("httpx required for Headroom: pip install httpx")

    response = httpx.post(
        endpoint,
        json={"prompt": prompt, **kwargs},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()

    result = response.json()
    return result.get("optimized_prompt", prompt)

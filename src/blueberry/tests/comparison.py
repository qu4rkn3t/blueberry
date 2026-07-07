"""Multi-model comparison tests."""

import asyncio
from typing import Any

from blueberry.atomic import calculate_cost, measure_latency
from blueberry.core.provider import Provider


async def compare_models(
    providers: list[Provider] | dict[str, Provider],
    prompt: str,
    max_tokens: int = 200,
    runs: int = 3,
    include_cost: bool = True,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """
    Compare multiple models on the same prompt.

    Args:
        providers: List of providers or dict of {name: provider}
        prompt: Test prompt
        max_tokens: Max output tokens
        runs: Number of latency measurement runs
        include_cost: Whether to calculate costs
        **kwargs: Additional completion parameters

    Returns:
        {
            "model_name": {
                "latency": {...},
                "cost": {...} (if include_cost),
                "summary": {
                    "avg_latency_ms": float,
                    "cost_usd": float (if include_cost)
                }
            },
            ...
        }

    Example:
        >>> providers = {
        ...     "pro": GeminiProvider(model="gemini-1.5-pro", ...),
        ...     "flash": GeminiProvider(model="gemini-1.5-flash", ...)
        ... }
        >>> results = await compare_models(providers, "Explain AI")
        >>> for name, result in results.items():
        ...     print(f"{name}: {result['summary']['avg_latency_ms']:.2f}ms")
    """
    # Convert list to dict if needed
    if isinstance(providers, list):
        provider_dict = {}
        for provider in providers:
            metadata = provider.get_metadata()
            name = f"{metadata.provider}:{metadata.model_name}"
            provider_dict[name] = provider
    else:
        provider_dict = providers

    async def test_provider(name: str, provider: Provider) -> tuple[str, dict]:
        latency = await measure_latency(
            provider, prompt, max_tokens=max_tokens, runs=runs, **kwargs
        )

        result = {
            "latency": latency,
            "summary": {
                "avg_latency_ms": latency["mean_ms"],
            },
        }

        if include_cost:
            cost = await calculate_cost(
                provider, prompt, max_tokens=max_tokens, **kwargs
            )
            result["cost"] = cost
            result["summary"]["cost_usd"] = cost["cost_usd"]

        return name, result

    # Test all models concurrently
    results_list = await asyncio.gather(
        *[test_provider(name, provider) for name, provider in provider_dict.items()]
    )

    return dict(results_list)

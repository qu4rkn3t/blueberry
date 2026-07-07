"""Standard cost analysis tests."""

from typing import Any

from blueberry.atomic import calculate_cost, estimate_cost, measure_latency
from blueberry.core.provider import Provider


async def cost_analysis(
    provider: Provider,
    prompt: str,
    output_sizes: list[int] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Analyze costs at different output sizes.

    Args:
        provider: LLM provider
        prompt: Test prompt
        output_sizes: List of output token sizes to test (default: [100, 500, 1000])
        **kwargs: Additional completion parameters

    Returns:
        {
            "input_tokens": int,
            "costs_by_size": {
                100: {"estimated": float, "actual": float},
                500: {...},
                ...
            },
            "summary": {
                "cheapest_actual": float,
                "most_expensive_actual": float,
            }
        }
    """
    if output_sizes is None:
        output_sizes = [100, 500, 1000]

    input_tokens = provider.count_tokens(prompt)
    costs_by_size = {}

    for size in output_sizes:
        # Estimate
        estimated = estimate_cost(provider, prompt, size)

        # Actual
        actual_result = await calculate_cost(
            provider, prompt, max_tokens=size, **kwargs
        )

        costs_by_size[size] = {
            "estimated_usd": estimated,
            "actual_usd": actual_result["cost_usd"],
            "actual_tokens": actual_result["output_tokens"],
        }

    actual_costs = [c["actual_usd"] for c in costs_by_size.values() if c["actual_usd"]]

    return {
        "input_tokens": input_tokens,
        "costs_by_size": costs_by_size,
        "summary": {
            "cheapest_actual_usd": min(actual_costs) if actual_costs else None,
            "most_expensive_actual_usd": max(actual_costs) if actual_costs else None,
        },
    }


async def cost_efficiency(
    provider: Provider,
    prompt: str,
    max_tokens: int = 200,
    runs: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Calculate cost efficiency: cost per millisecond.

    Lower is better - you want minimum cost for maximum speed.

    Args:
        provider: LLM provider
        prompt: Test prompt
        max_tokens: Max output tokens
        runs: Number of runs
        **kwargs: Additional completion parameters

    Returns:
        {
            "latency_ms": float,
            "cost_usd": float,
            "efficiency": {
                "cost_per_ms": float,
                "ms_per_dollar": float,
                "tokens_per_dollar": float,
            }
        }
    """
    latency = await measure_latency(
        provider, prompt, max_tokens=max_tokens, runs=runs, **kwargs
    )
    cost = await calculate_cost(provider, prompt, max_tokens=max_tokens, **kwargs)

    cost_usd = cost["cost_usd"] or 0
    latency_ms = latency["mean_ms"]

    return {
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "total_tokens": cost["total_tokens"],
        "efficiency": {
            "cost_per_ms": cost_usd / latency_ms if latency_ms > 0 else 0,
            "ms_per_dollar": latency_ms / cost_usd if cost_usd > 0 else float("inf"),
            "tokens_per_dollar": cost["total_tokens"] / cost_usd
            if cost_usd > 0
            else float("inf"),
        },
    }

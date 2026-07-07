"""Standard scale and capacity tests."""

from typing import Any

from blueberry.atomic import (
    calculate_cost,
    count_tokens,
    find_context_limit,
    measure_throughput,
)
from blueberry.core.provider import Provider


async def context_capacity(
    provider: Provider,
    test_string: str = "x",
    max_attempts: int = 20,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Test context window capacity and utilization.

    Args:
        provider: LLM provider
        test_string: String to repeat for testing
        max_attempts: Max binary search iterations
        **kwargs: Additional parameters

    Returns:
        {
            "theoretical_max": int,
            "actual_max": int,
            "utilization_rate": float (0-100),
            "wasted_context": int
        }
    """
    metadata = provider.get_metadata()
    theoretical = metadata.context_window or 0

    actual = await find_context_limit(
        provider, test_string=test_string, max_attempts=max_attempts, **kwargs
    )

    utilization = (actual / theoretical * 100) if theoretical > 0 else 0

    return {
        "theoretical_max": theoretical,
        "actual_max": actual,
        "utilization_rate": utilization,
        "wasted_context": theoretical - actual,
    }


async def prompt_stress_test(
    provider: Provider,
    prompt: str,
    context_percentages: list[float] | None = None,
    max_tokens: int = 200,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Test performance at different context utilization levels.

    Args:
        provider: LLM provider
        prompt: Base prompt
        context_percentages: Context fill levels (default: [0.1, 0.5, 0.9])
        max_tokens: Max output tokens
        **kwargs: Additional completion parameters

    Returns:
        {
            "by_utilization": {
                10: {
                    "prompt_tokens": int,
                    "throughput": {...},
                    "cost": {...}
                },
                ...
            },
            "degradation": {
                "throughput_drop": float (0-100),
                "cost_increase": float (0-100)
            }
        }
    """
    if context_percentages is None:
        context_percentages = [0.1, 0.5, 0.9]

    metadata = provider.get_metadata()
    context_limit = metadata.context_window or 100000

    results = {}
    filler = "Lorem ipsum dolor sit amet. "

    for pct in context_percentages:
        # Fill context to target percentage
        target_tokens = int(context_limit * pct)
        test_prompt = prompt

        current = count_tokens(provider, test_prompt)
        while current < target_tokens:
            test_prompt += filler
            current = count_tokens(provider, test_prompt)

        # Test performance at this utilization
        throughput = await measure_throughput(
            provider, test_prompt, max_tokens=max_tokens, **kwargs
        )
        cost = await calculate_cost(
            provider, test_prompt, max_tokens=max_tokens, **kwargs
        )

        pct_label = int(pct * 100)
        results[pct_label] = {
            "prompt_tokens": current,
            "throughput": throughput,
            "cost": cost,
        }

    # Calculate degradation from lowest to highest utilization
    sorted_utils = sorted(context_percentages)
    if len(sorted_utils) >= 2:
        low = int(sorted_utils[0] * 100)
        high = int(sorted_utils[-1] * 100)

        low_tps = results[low]["throughput"]["tokens_per_second"]
        high_tps = results[high]["throughput"]["tokens_per_second"]
        throughput_drop = ((low_tps - high_tps) / low_tps * 100) if low_tps > 0 else 0

        low_cost = results[low]["cost"]["cost_usd"] or 0
        high_cost = results[high]["cost"]["cost_usd"] or 0
        cost_increase = ((high_cost - low_cost) / low_cost * 100) if low_cost > 0 else 0

        degradation = {
            "throughput_drop_pct": throughput_drop,
            "cost_increase_pct": cost_increase,
        }
    else:
        degradation = None

    return {
        "by_utilization": results,
        "degradation": degradation,
    }

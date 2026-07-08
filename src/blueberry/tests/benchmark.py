"""Standard benchmarking tests."""

from typing import Any

from blueberry.atomic import measure_latency, measure_throughput
from blueberry.core.provider import Provider


async def performance_profile(
    provider: Provider,
    prompt: str,
    max_tokens: int = 200,
    runs: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Comprehensive performance profile: latency + throughput.

    Args:
        provider: LLM provider
        prompt: Test prompt
        max_tokens: Max output tokens
        runs: Number of measurement runs
        **kwargs: Additional completion parameters

    Returns:
        {
            "latency": {...},
            "throughput": {...},
            "summary": {
                "avg_latency_ms": float,
                "tokens_per_second": float,
            }
        }
    """
    latency = await measure_latency(
        provider, prompt, max_tokens=max_tokens, runs=runs, **kwargs
    )
    throughput = await measure_throughput(
        provider, prompt, max_tokens=max_tokens, runs=runs, **kwargs
    )

    return {
        "latency": latency,
        "throughput": throughput,
        "summary": {
            "avg_latency_ms": latency["mean_ms"],
            "tokens_per_second": throughput["tokens_per_second"],
        },
    }


async def benchmark_suite(
    provider: Provider,
    prompts: list[str] | dict[str, str],
    max_tokens: int = 200,
    runs: int = 3,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """
    Run performance profiles on multiple prompts.

    Args:
        provider: LLM provider
        prompts: List of prompts or dict of {name: prompt}
        max_tokens: Max output tokens
        runs: Number of runs per prompt
        **kwargs: Additional completion parameters

    Returns:
        {
            "prompt_name": {
                "latency": {...},
                "throughput": {...},
                "summary": {...}
            },
            ...
        }
    """

    if isinstance(prompts, list):
        prompts = {f"prompt_{i}": p for i, p in enumerate(prompts)}

    results = {}
    for name, prompt in prompts.items():
        results[name] = await performance_profile(
            provider, prompt, max_tokens=max_tokens, runs=runs, **kwargs
        )

    return results

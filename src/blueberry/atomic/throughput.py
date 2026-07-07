"""Throughput measurement primitives."""

import time
from typing import Any

from blueberry.core.provider import CompletionRequest, Provider


async def measure_throughput(
    provider: Provider,
    prompt: str,
    max_tokens: int = 500,
    runs: int = 1,
    **kwargs: Any,
) -> dict[str, float]:
    """
    Measure tokens per second throughput.

    Args:
        provider: LLM provider
        prompt: Prompt to test
        max_tokens: Max tokens to generate
        runs: Number of runs to average
        **kwargs: Additional completion parameters

    Returns:
        {
            "tokens_per_second": average throughput,
            "total_tokens": total tokens generated,
            "total_time_s": total time in seconds
        }
    """
    total_tokens = 0
    total_time = 0.0

    for _ in range(runs):
        start = time.perf_counter()
        response = await provider.complete(
            CompletionRequest(prompt=prompt, max_tokens=max_tokens, **kwargs)
        )
        elapsed = time.perf_counter() - start

        total_tokens += response.output_tokens
        total_time += elapsed

    tokens_per_second = total_tokens / total_time if total_time > 0 else 0

    return {
        "tokens_per_second": tokens_per_second,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
    }

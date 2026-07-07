"""Latency measurement primitives."""

import time
from typing import Any

from blueberry.core.provider import CompletionRequest, Provider


async def measure_latency(
    provider: Provider,
    prompt: str,
    max_tokens: int = 100,
    runs: int = 1,
    **kwargs: Any,
) -> dict[str, float]:
    """
    Measure end-to-end completion latency.

    Args:
        provider: LLM provider
        prompt: Prompt to test
        max_tokens: Max tokens to generate
        runs: Number of runs to average
        **kwargs: Additional completion parameters

    Returns:
        {
            "mean_ms": average latency,
            "min_ms": minimum latency,
            "max_ms": maximum latency,
            "std_ms": standard deviation
        }
    """
    latencies = []

    for _ in range(runs):
        start = time.perf_counter()
        await provider.complete(
            CompletionRequest(prompt=prompt, max_tokens=max_tokens, **kwargs)
        )
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        latencies.append(elapsed)

    mean = sum(latencies) / len(latencies)
    variance = sum((x - mean) ** 2 for x in latencies) / len(latencies)
    std = variance**0.5

    return {
        "mean_ms": mean,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "std_ms": std,
    }


async def measure_ttft(
    provider: Provider, prompt: str, max_tokens: int = 100, **kwargs: Any
) -> float:
    """
    Measure time to first token (TTFT) using streaming.

    Args:
        provider: LLM provider
        prompt: Prompt to test
        max_tokens: Max tokens to generate
        **kwargs: Additional completion parameters

    Returns:
        Time to first token in milliseconds
    """
    start = time.perf_counter()
    stream = provider.stream(
        CompletionRequest(prompt=prompt, max_tokens=max_tokens, stream=True, **kwargs)
    )

    # Get first chunk
    async for _ in stream:
        ttft = (time.perf_counter() - start) * 1000
        # Consume rest of stream
        async for _ in stream:
            pass
        return ttft

    return 0.0  # No chunks received

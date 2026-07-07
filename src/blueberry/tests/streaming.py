"""Streaming performance tests."""

from typing import Any

from blueberry.atomic import measure_throughput, measure_ttft
from blueberry.core.provider import Provider


async def streaming_performance(
    provider: Provider,
    prompt: str,
    max_tokens: int = 500,
    runs: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Measure streaming-specific performance: TTFT and throughput.

    Args:
        provider: LLM provider
        prompt: Test prompt
        max_tokens: Max output tokens
        runs: Number of measurement runs
        **kwargs: Additional completion parameters

    Returns:
        {
            "ttft_ms": time to first token,
            "throughput": {...},
            "summary": {
                "ttft_ms": float,
                "tokens_per_second": float,
            }
        }
    """
    ttft = await measure_ttft(provider, prompt, max_tokens=max_tokens, **kwargs)
    throughput = await measure_throughput(
        provider, prompt, max_tokens=max_tokens, runs=runs, **kwargs
    )

    return {
        "ttft_ms": ttft,
        "throughput": throughput,
        "summary": {
            "ttft_ms": ttft,
            "tokens_per_second": throughput["tokens_per_second"],
        },
    }

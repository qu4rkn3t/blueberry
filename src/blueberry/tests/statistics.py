"""Statistical analysis tests."""

import asyncio
import time
from typing import Any

from blueberry.core.provider import CompletionRequest, Provider


async def latency_percentiles(
    provider: Provider,
    prompt: str,
    max_tokens: int = 100,
    runs: int = 100,
    percentiles: list[float] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """
    Measure latency percentiles (P50, P95, P99, P999).

    Args:
        provider: LLM provider
        prompt: Test prompt
        max_tokens: Max output tokens
        runs: Number of runs (recommend 100+ for accurate percentiles)
        percentiles: Custom percentiles (default: [50, 95, 99, 99.9])
        **kwargs: Additional completion parameters

    Returns:
        {
            "p50": median latency (ms),
            "p95": 95th percentile (ms),
            "p99": 99th percentile (ms),
            "p999": 99.9th percentile (ms),
            "mean": average latency (ms),
            "min": minimum latency (ms),
            "max": maximum latency (ms),
            "runs": number of runs
        }
    """
    if percentiles is None:
        percentiles = [50, 95, 99, 99.9]

    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        await provider.complete(
            CompletionRequest(prompt=prompt, max_tokens=max_tokens, **kwargs)
        )
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    # Sort for percentile calculation
    latencies.sort()

    def get_percentile(data: list[float], p: float) -> float:
        """Calculate percentile from sorted data."""
        if not data:
            return 0.0
        k = (len(data) - 1) * (p / 100)
        f = int(k)
        c = int(k) + 1
        if c >= len(data):
            return data[-1]
        return data[f] + (k - f) * (data[c] - data[f])

    result = {
        "mean": sum(latencies) / len(latencies),
        "min": latencies[0],
        "max": latencies[-1],
        "runs": runs,
    }

    for p in percentiles:
        key = f"p{str(p).replace('.', '')}"  # p99.9 -> p999
        result[key] = get_percentile(latencies, p)

    return result


async def statistical_comparison(
    provider_a: Provider,
    provider_b: Provider,
    prompt: str,
    max_tokens: int = 100,
    runs: int = 30,
    alpha: float = 0.05,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Compare two models with statistical significance testing.

    Uses Mann-Whitney U test (non-parametric) to determine if latency
    distributions are significantly different.

    Args:
        provider_a: First provider to compare
        provider_b: Second provider to compare
        prompt: Test prompt
        max_tokens: Max output tokens
        runs: Number of runs per provider (recommend 30+)
        alpha: Significance level (default 0.05 for 95% confidence)
        **kwargs: Additional completion parameters

    Returns:
        {
            "provider_a": {
                "mean_ms": float,
                "median_ms": float,
                "name": str
            },
            "provider_b": {
                "mean_ms": float,
                "median_ms": float,
                "name": str
            },
            "difference_ms": mean difference (A - B),
            "faster_provider": "a" or "b" or "tie",
            "statistically_significant": boolean,
            "p_value": float (approximation),
            "confidence": confidence level (1 - alpha)
        }
    """

    async def measure_provider(provider: Provider) -> list[float]:
        latencies = []
        for _ in range(runs):
            start = time.perf_counter()
            await provider.complete(
                CompletionRequest(prompt=prompt, max_tokens=max_tokens, **kwargs)
            )
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
        return latencies

    latencies_a, latencies_b = await asyncio.gather(
        measure_provider(provider_a), measure_provider(provider_b)
    )

    mean_a = sum(latencies_a) / len(latencies_a)
    mean_b = sum(latencies_b) / len(latencies_b)

    sorted_a = sorted(latencies_a)
    sorted_b = sorted(latencies_b)
    median_a = sorted_a[len(sorted_a) // 2]
    median_b = sorted_b[len(sorted_b) // 2]

    def mann_whitney_u(x: list[float], y: list[float]) -> tuple[float, float]:
        """Simplified Mann-Whitney U test."""
        n1, n2 = len(x), len(y)
        ranked = sorted([(val, 0) for val in x] + [(val, 1) for val in y])
        rank_sum_x = sum(i + 1 for i, (val, group) in enumerate(ranked) if group == 0)

        u1 = rank_sum_x - (n1 * (n1 + 1)) / 2
        u2 = n1 * n2 - u1
        u = min(u1, u2)

        # Normal approximation for p-value
        mu = n1 * n2 / 2
        sigma = ((n1 * n2 * (n1 + n2 + 1)) / 12) ** 0.5
        z = abs((u - mu) / sigma) if sigma > 0 else 0

        # Two-tailed p-value approximation (standard normal)
        p_value = 2 * (1 - _norm_cdf(z))

        return u, min(p_value, 1.0)

    def _norm_cdf(x: float) -> float:
        """Approximation of standard normal CDF."""
        return (1.0 + _erf(x / (2**0.5))) / 2.0

    def _erf(x: float) -> float:
        """Approximation of error function."""
        # Abramowitz and Stegun approximation
        a1, a2, a3, a4, a5 = (
            0.254829592,
            -0.284496736,
            1.421413741,
            -1.453152027,
            1.061405429,
        )
        p = 0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (
            2.718281828 ** (-(x**2))
        )

        return sign * y

    u_stat, p_value = mann_whitney_u(latencies_a, latencies_b)
    is_significant = p_value < alpha

    difference = mean_a - mean_b
    if abs(difference) < 0.01:  # Within 0.01ms
        faster = "tie"
    else:
        faster = "b" if difference > 0 else "a"

    metadata_a = provider_a.get_metadata()
    metadata_b = provider_b.get_metadata()

    return {
        "provider_a": {
            "name": f"{metadata_a.provider}:{metadata_a.model_name}",
            "mean_ms": mean_a,
            "median_ms": median_a,
        },
        "provider_b": {
            "name": f"{metadata_b.provider}:{metadata_b.model_name}",
            "mean_ms": mean_b,
            "median_ms": median_b,
        },
        "difference_ms": difference,
        "faster_provider": faster,
        "statistically_significant": is_significant,
        "p_value": p_value,
        "confidence": 1 - alpha,
    }

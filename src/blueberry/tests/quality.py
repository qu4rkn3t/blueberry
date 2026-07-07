"""Standard quality and reliability tests."""

from typing import Any

from blueberry.atomic import check_consistency, check_instruction_following, measure_latency
from blueberry.core.provider import Provider


async def quality_check(
    provider: Provider,
    prompt: str,
    expected_keyword: str | None = None,
    max_tokens: int = 200,
    consistency_runs: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Multi-faceted quality check: consistency + optional instruction following.

    Args:
        provider: LLM provider
        prompt: Test prompt
        expected_keyword: If provided, check for this keyword in output
        max_tokens: Max output tokens
        consistency_runs: Number of consistency checks
        **kwargs: Additional completion parameters

    Returns:
        {
            "consistency": {...},
            "instruction_following": {...} or None,
            "quality_score": float (0-100)
        }
    """
    consistency = await check_consistency(
        provider, prompt, max_tokens=max_tokens, runs=consistency_runs, **kwargs
    )

    instruction = None
    if expected_keyword:
        instruction = await check_instruction_following(
            provider, prompt, expected_keyword, max_tokens=max_tokens, **kwargs
        )

    # Quality score: consistency rate + instruction following bonus
    quality_score = consistency["consistency_rate"]
    if instruction and instruction["follows_instruction"]:
        quality_score = min(100, quality_score + 20)  # Bonus for following instructions

    return {
        "consistency": consistency,
        "instruction_following": instruction,
        "quality_score": quality_score,
    }


async def reliability_score(
    provider: Provider,
    prompt: str,
    max_tokens: int = 100,
    runs: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Reliability score: consistency × latency stability.

    High reliability = consistent outputs with stable latency.

    Args:
        provider: LLM provider
        prompt: Test prompt
        max_tokens: Max output tokens
        runs: Number of test runs
        **kwargs: Additional completion parameters

    Returns:
        {
            "consistency": {...},
            "latency": {...},
            "reliability": {
                "consistency_rate": float,
                "latency_cv": float (coefficient of variation),
                "score": float (0-100)
            }
        }
    """
    consistency = await check_consistency(
        provider, prompt, max_tokens=max_tokens, runs=runs, **kwargs
    )
    latency = await measure_latency(
        provider, prompt, max_tokens=max_tokens, runs=runs, **kwargs
    )

    # Coefficient of variation for latency
    latency_cv = latency["std_ms"] / latency["mean_ms"] if latency["mean_ms"] > 0 else 1

    # Reliability score: high consistency + low latency variance
    # CV capped at 1 for scoring
    score = consistency["consistency_rate"] * (1 - min(latency_cv, 1))

    return {
        "consistency": consistency,
        "latency": latency,
        "reliability": {
            "consistency_rate": consistency["consistency_rate"],
            "latency_cv": latency_cv,
            "score": score,
        },
    }

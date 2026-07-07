"""Reliability and error rate tests."""

import asyncio
from typing import Any

from blueberry.core.provider import CompletionRequest, Provider


async def error_rate(
    provider: Provider,
    prompt: str,
    max_tokens: int = 100,
    attempts: int = 20,
    timeout_seconds: float = 30.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Measure error rate and failure modes under repeated calls.

    Args:
        provider: LLM provider
        prompt: Test prompt
        max_tokens: Max output tokens
        attempts: Number of attempts to make
        timeout_seconds: Timeout per request
        **kwargs: Additional completion parameters

    Returns:
        {
            "total_attempts": int,
            "successful": int,
            "failed": int,
            "timeouts": int,
            "error_rate": float (0-100),
            "success_rate": float (0-100),
            "errors": list of error messages
        }

    Example:
        >>> result = await error_rate(provider, "Hello", attempts=50)
        >>> print(f"Success rate: {result['success_rate']:.1f}%")
        >>> print(f"Errors: {len(result['errors'])}")
    """
    successful = 0
    failed = 0
    timeouts = 0
    errors = []

    for _ in range(attempts):
        try:
            await asyncio.wait_for(
                provider.complete(
                    CompletionRequest(prompt=prompt, max_tokens=max_tokens, **kwargs)
                ),
                timeout=timeout_seconds,
            )
            successful += 1
        except asyncio.TimeoutError:
            timeouts += 1
            errors.append("Timeout")
        except Exception as e:
            failed += 1
            errors.append(str(e))

    total_failed = failed + timeouts
    error_rate = (total_failed / attempts) * 100
    success_rate = (successful / attempts) * 100

    return {
        "total_attempts": attempts,
        "successful": successful,
        "failed": failed,
        "timeouts": timeouts,
        "error_rate": error_rate,
        "success_rate": success_rate,
        "errors": errors,
    }

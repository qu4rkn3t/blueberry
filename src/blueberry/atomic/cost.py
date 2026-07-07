"""Cost calculation primitives."""

from blueberry.core.provider import CompletionRequest, Provider


def estimate_cost(
    provider: Provider, prompt: str, estimated_output_tokens: int
) -> float | None:
    """
    Estimate cost without making API call.

    Args:
        provider: LLM provider
        prompt: Input prompt
        estimated_output_tokens: Expected output size

    Returns:
        Estimated cost in USD, or None if pricing unavailable
    """
    metadata = provider.get_metadata()

    if (
        metadata.input_cost_per_million is None
        or metadata.output_cost_per_million is None
    ):
        return None

    input_tokens = provider.count_tokens(prompt)

    input_cost = (input_tokens / 1_000_000) * metadata.input_cost_per_million
    output_cost = (
        estimated_output_tokens / 1_000_000
    ) * metadata.output_cost_per_million

    return input_cost + output_cost


async def calculate_cost(
    provider: Provider, prompt: str, max_tokens: int = 100, **kwargs
) -> dict[str, float | int | None]:
    """
    Calculate actual cost by making API call.

    Args:
        provider: LLM provider
        prompt: Input prompt
        max_tokens: Max output tokens
        **kwargs: Additional completion parameters

    Returns:
        {
            "input_tokens": input token count,
            "output_tokens": output token count,
            "total_tokens": total tokens,
            "cost_usd": cost in USD (or None)
        }
    """
    response = await provider.complete(
        CompletionRequest(prompt=prompt, max_tokens=max_tokens, **kwargs)
    )

    metadata = provider.get_metadata()
    cost = None

    if (
        metadata.input_cost_per_million is not None
        and metadata.output_cost_per_million is not None
    ):
        input_cost = (
            response.input_tokens / 1_000_000
        ) * metadata.input_cost_per_million
        output_cost = (
            response.output_tokens / 1_000_000
        ) * metadata.output_cost_per_million
        cost = input_cost + output_cost

    return {
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.input_tokens + response.output_tokens,
        "cost_usd": cost,
    }

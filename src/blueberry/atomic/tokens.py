"""Token-related primitives."""

from blueberry.core.provider import CompletionRequest, Provider


def count_tokens(provider: Provider, text: str) -> int:
    """
    Count tokens in text using provider's tokenizer.

    Args:
        provider: LLM provider
        text: Text to tokenize

    Returns:
        Token count
    """
    return provider.count_tokens(text)


async def find_context_limit(
    provider: Provider,
    test_string: str = "x",
    max_attempts: int = 20,
    verification_prompt: str = "Repeat the last character",
) -> int:
    """
    Find actual context window size via binary search.

    Args:
        provider: LLM provider
        test_string: String to repeat
        max_attempts: Max binary search iterations
        verification_prompt: Prompt to verify context acceptance

    Returns:
        Maximum context size in tokens
    """
    metadata = provider.get_metadata()
    theoretical_max = metadata.context_window or 1_000_000

    low, high = 0, theoretical_max
    actual_max = 0

    for _ in range(max_attempts):
        if low > high:
            break

        mid = (low + high) // 2
        test_prompt = test_string * mid + "\n\n" + verification_prompt

        try:
            tokens = provider.count_tokens(test_prompt)
            response = await provider.complete(
                CompletionRequest(prompt=test_prompt, max_tokens=10, temperature=0.0)
            )

            if response.content:
                actual_max = tokens
                low = mid + 1
            else:
                high = mid - 1
        except Exception:
            high = mid - 1

    return actual_max

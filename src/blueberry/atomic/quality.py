"""Quality measurement primitives."""

from typing import Any

from blueberry.core.provider import CompletionRequest, Provider


async def check_consistency(
    provider: Provider,
    prompt: str,
    max_tokens: int = 100,
    runs: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Check output consistency with temperature=0.

    Args:
        provider: LLM provider
        prompt: Prompt to test
        max_tokens: Max output tokens
        runs: Number of runs
        **kwargs: Additional completion parameters

    Returns:
        {
            "unique_outputs": number of unique outputs,
            "consistency_rate": percentage (0-100),
            "outputs": list of outputs
        }
    """
    outputs = []

    for _ in range(runs):
        response = await provider.complete(
            CompletionRequest(
                prompt=prompt, max_tokens=max_tokens, temperature=0.0, **kwargs
            )
        )
        outputs.append(response.content)

    unique = len(set(outputs))
    consistency_rate = (runs - unique + 1) / runs * 100

    return {
        "unique_outputs": unique,
        "consistency_rate": consistency_rate,
        "outputs": outputs,
    }


async def check_instruction_following(
    provider: Provider,
    prompt: str,
    expected_keyword: str,
    max_tokens: int = 200,
    case_sensitive: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Check if model follows simple instructions.

    Args:
        provider: LLM provider
        prompt: Instruction prompt
        expected_keyword: Keyword that should appear in output
        max_tokens: Max output tokens
        case_sensitive: Whether keyword match is case-sensitive
        **kwargs: Additional completion parameters

    Returns:
        {
            "follows_instruction": boolean,
            "output": model output,
            "expected": expected keyword
        }
    """
    response = await provider.complete(
        CompletionRequest(prompt=prompt, max_tokens=max_tokens, **kwargs)
    )

    output = response.content
    keyword = expected_keyword

    if not case_sensitive:
        output = output.lower()
        keyword = keyword.lower()

    follows = keyword in output

    return {
        "follows_instruction": follows,
        "output": response.content,
        "expected": expected_keyword,
    }

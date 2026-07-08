"""OpenAI-compatible API specification."""

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from blueberry.core.provider import (
    BatchCapabilities,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    EmbeddingResponse,
    ModelMetadata,
    RateLimitInfo,
)
from blueberry.providers._helpers import (
    count_tokens_with_tiktoken,
    estimate_tokens,
    get_tiktoken_tokenizer,
)


class OpenAISpec:
    """OpenAI-compatible API specification implementation."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs,
    ):
        self.model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = AsyncOpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._metadata = self._create_metadata(**kwargs)

    def _create_metadata(self, **kwargs) -> ModelMetadata:
        """Create model metadata. Override in model implementations."""
        return ModelMetadata(
            model_name=self.model,
            provider="openai-compatible",
            context_window=kwargs.get("context_window"),
            max_output_tokens=kwargs.get("max_output_tokens"),
            input_cost_per_million=kwargs.get("input_cost_per_million"),
            output_cost_per_million=kwargs.get("output_cost_per_million"),
            supports_streaming=True,
            metadata=kwargs,
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Execute a completion request."""
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop_sequences,
            stream=False,
            **request.extra_params,
        )

        choice = response.choices[0]
        return CompletionResponse(
            content=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            metadata={"response_id": response.id},
        )

    async def stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionChunk]:
        """Stream a completion request."""
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop_sequences,
            stream=True,
            **request.extra_params,
        )

        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice:
                yield CompletionChunk(
                    content=choice.delta.content or "",
                    finish_reason=choice.finish_reason,
                )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken if available, otherwise estimate.
        Override in model implementations with provider-specific tokenizers.
        """
        token_count = count_tokens_with_tiktoken(text, self.model)
        if token_count is not None:
            return token_count

        return estimate_tokens(text)

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        return self._metadata

    def get_rate_limits(self) -> RateLimitInfo:
        """
        Get rate limit information.

        Override in model implementations with provider-specific limits.
        """
        return RateLimitInfo()

    def get_batch_capabilities(self) -> BatchCapabilities:
        """
        Get batch processing capabilities.

        Override in model implementations that support batching.
        """
        return BatchCapabilities()

    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs using tiktoken.

        Falls back to empty list if tiktoken unavailable.
        """
        tokenizer = get_tiktoken_tokenizer(self.model)
        if tokenizer:
            return tokenizer.encode(text)
        return []

    def decode(self, tokens: list[int]) -> str:
        """
        Decode token IDs to text using tiktoken.

        Falls back to empty string if tiktoken unavailable.
        """
        tokenizer = get_tiktoken_tokenizer(self.model)
        if tokenizer:
            return tokenizer.decode(tokens)
        return ""

    async def embed(self, text: str) -> EmbeddingResponse:
        """
        Generate embeddings using OpenAI embeddings API.

        Only works if model supports embeddings (text-embedding-* models).
        """
        if not self.model.startswith("text-embedding-"):
            raise NotImplementedError(f"Model {self.model} does not support embeddings")

        response = await self._client.embeddings.create(model=self.model, input=text)

        embedding_data = response.data[0]
        return EmbeddingResponse(
            embedding=embedding_data.embedding,
            model=response.model,
            dimensions=len(embedding_data.embedding),
        )

    def validate_request(self, request: CompletionRequest) -> dict[str, Any]:
        """
        Validate completion request against model constraints.

        Checks token limits, parameter ranges, etc.
        """
        errors = []
        warnings = []

        token_count = self.count_tokens(request.prompt)

        metadata = self.get_metadata()
        if metadata.context_window and token_count > metadata.context_window:
            errors.append(
                f"Prompt has {token_count} tokens, exceeds context window of {metadata.context_window}"
            )

        if request.max_tokens:
            if (
                metadata.max_output_tokens
                and request.max_tokens > metadata.max_output_tokens
            ):
                errors.append(
                    f"max_tokens {request.max_tokens} exceeds model limit of {metadata.max_output_tokens}"
                )

            if metadata.context_window:
                total = token_count + request.max_tokens
                if total > metadata.context_window:
                    warnings.append(
                        f"Prompt ({token_count}) + max_tokens ({request.max_tokens}) = {total} may exceed context window ({metadata.context_window})"
                    )

        if not 0 <= request.temperature <= 2:
            warnings.append(
                f"temperature {request.temperature} outside typical range [0, 2]"
            )

        if not 0 <= request.top_p <= 1:
            errors.append(f"top_p {request.top_p} must be between 0 and 1")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "token_count": token_count,
        }

    async def close(self) -> None:
        """Cleanup resources."""
        await self._client.close()

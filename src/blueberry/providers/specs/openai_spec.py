"""OpenAI-compatible API specification."""

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from blueberry.core.provider import (
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    ModelMetadata,
)
from blueberry.providers._helpers import estimate_tokens


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
        Count tokens. Uses estimation by default.
        Override in model implementations with provider-specific tokenizers.
        """
        return estimate_tokens(text)

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        return self._metadata

    async def close(self) -> None:
        """Cleanup resources."""
        await self._client.close()

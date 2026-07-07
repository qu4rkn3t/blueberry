from collections.abc import AsyncIterator
from typing import Any, Protocol

from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """Metadata about a model's capabilities and pricing."""

    model_name: str
    provider: str
    context_window: int | None = None
    max_output_tokens: int | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    supports_streaming: bool = True
    supports_function_calling: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompletionRequest(BaseModel):
    """Standardized completion request."""

    prompt: str
    max_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    stop_sequences: list[str] | None = None
    stream: bool = False
    extra_params: dict[str, Any] = Field(default_factory=dict)


class CompletionResponse(BaseModel):
    """Standardized completion response."""

    content: str
    model: str
    finish_reason: str | None = None
    input_tokens: int
    output_tokens: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompletionChunk(BaseModel):
    """Streaming completion chunk."""

    content: str
    finish_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Provider(Protocol):
    """Protocol defining the provider interface."""

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Execute a completion request."""
        ...

    async def stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionChunk]:
        """Stream a completion request."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        ...

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        ...

    async def close(self) -> None:
        """Cleanup resources."""
        ...

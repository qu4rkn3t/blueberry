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


class RateLimitInfo(BaseModel):
    """Rate limit information for a provider."""

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    requests_per_day: int | None = None
    concurrent_requests: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchCapabilities(BaseModel):
    """Batch processing capabilities."""

    supports_batch: bool = False
    max_batch_size: int | None = None
    batch_input_cost_multiplier: float = 1.0
    batch_output_cost_multiplier: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingResponse(BaseModel):
    """Response from embedding request."""

    embedding: list[float]
    model: str
    dimensions: int
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
        """
        Count tokens in text using provider-specific tokenizer.

        Models should override this with their native tokenizer.
        """
        ...

    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs using provider's tokenizer.

        Returns list of token IDs. Override with native tokenizer.
        """
        ...

    def decode(self, tokens: list[int]) -> str:
        """
        Decode token IDs back to text using provider's tokenizer.

        Override with native tokenizer.
        """
        ...

    async def embed(self, text: str) -> EmbeddingResponse:
        """
        Generate embeddings for text.

        Only supported by models with embedding capabilities.
        Raises NotImplementedError if not supported.
        """
        ...

    def validate_request(self, request: CompletionRequest) -> dict[str, Any]:
        """
        Validate request against model constraints.

        Returns dict with validation results:
        {
            "valid": bool,
            "errors": list[str],
            "warnings": list[str],
            "token_count": int
        }
        """
        ...

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        ...

    def get_rate_limits(self) -> RateLimitInfo:
        """
        Get rate limit information for this provider.

        Returns default unlimited if not implemented.
        """
        ...

    def get_batch_capabilities(self) -> BatchCapabilities:
        """
        Get batch processing capabilities.

        Returns default no-batch-support if not implemented.
        """
        ...

    async def close(self) -> None:
        """Cleanup resources."""
        ...

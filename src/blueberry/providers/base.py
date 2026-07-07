"""Base provider protocol."""

from collections.abc import AsyncIterator
from typing import Protocol

from blueberry.core.provider import (
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    ModelMetadata,
)


class BaseProvider(Protocol):
    """Base protocol that all providers must implement."""

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

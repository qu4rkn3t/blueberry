"""Blueberry - Composable LLM performance testing."""

# Core provider abstraction
from blueberry.core.provider import (
    CompletionRequest,
    CompletionResponse,
    ModelMetadata,
    Provider,
)

# Providers
from blueberry.providers import GeminiProvider, OpenAISpec

# Test framework for user-defined tests
from blueberry.test import ComposedTest, Test, TestResult

__version__ = "0.1.0"

__all__ = [
    "CompletionRequest",
    "CompletionResponse",
    "ComposedTest",
    "GeminiProvider",
    "ModelMetadata",
    "OpenAISpec",
    "Provider",
    "Test",
    "TestResult",
]

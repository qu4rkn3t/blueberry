"""Test base class for creating reusable, composable tests."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from blueberry.core.provider import Provider


class TestResult(BaseModel):
    """
    Standardized test result format.

    All tests return this format for comparability.
    """

    test_name: str
    provider_name: str
    model_name: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    passed: bool | None = None


class Test(ABC):
    """
    Base class for creating reusable tests.

    Implement run() using atomic operations and return TestResult.
    Results are comparable across tests and models.
    """

    @abstractmethod
    async def run(self, provider: Provider, *args: Any, **kwargs: Any) -> TestResult:
        """
        Run the test using atomic operations.

        Args:
            provider: LLM provider to test
            *args: Test-specific arguments
            **kwargs: Test-specific arguments

        Returns:
            TestResult with metrics and metadata
        """
        ...

    async def setup(self) -> None:
        """Optional setup before run."""
        pass

    async def teardown(self) -> None:
        """Optional cleanup after run."""
        pass

    async def execute(
        self, provider: Provider, *args: Any, **kwargs: Any
    ) -> TestResult:
        """
        Execute full test lifecycle: setup -> run -> teardown.

        Args:
            provider: LLM provider to test
            *args: Passed to run()
            **kwargs: Passed to run()

        Returns:
            TestResult from run()
        """
        try:
            await self.setup()
            return await self.run(provider, *args, **kwargs)
        finally:
            await self.teardown()


class ComposedTest(Test):
    """
    Helper for composing multiple tests into one.

    Use when combining several tests with aggregated results.
    """

    pass


__all__ = ["Test", "TestResult", "ComposedTest"]

"""Examples: How to define custom tests the Ember way."""

import asyncio

from blueberry import ComposedTest, Test, TestResult
from blueberry.atomic import calculate_cost, count_tokens, measure_latency
from blueberry.providers import GeminiProvider
from blueberry.tests import cost_efficiency, performance_profile


# Example 1: Simple custom test using atomics
class CostPerMs(Test):
    """
    Custom metric: cost per millisecond.

    Combines latency and cost atomics.
    """

    async def run(self, provider, prompt, runs=5):
        # Use atomic operations
        latency = await measure_latency(provider, prompt, runs=runs)
        cost = await calculate_cost(provider, prompt)

        # Calculate custom metric
        cost_per_ms = cost["cost_usd"] / latency["mean_ms"]

        # Return standardized TestResult
        metadata = provider.get_metadata()
        return TestResult(
            test_name="cost_per_ms",
            provider_name=metadata.provider,
            model_name=metadata.model_name,
            metrics={
                "latency_ms": latency["mean_ms"],
                "cost_usd": cost["cost_usd"],
                "cost_per_ms": cost_per_ms,
            },
            passed=cost_per_ms < 0.00001,  # Optional threshold
        )


# Example 2: Custom test with your own logic
class PromptEfficiency(Test):
    """
    Custom test: how efficiently does model handle prompt size?

    Measures cost per input token.
    """

    async def run(self, provider, prompt):
        # Your custom logic
        input_tokens = count_tokens(provider, prompt)
        cost_result = await calculate_cost(provider, prompt, max_tokens=100)

        cost_per_input_token = cost_result["cost_usd"] / input_tokens

        metadata = provider.get_metadata()
        return TestResult(
            test_name="prompt_efficiency",
            provider_name=metadata.provider,
            model_name=metadata.model_name,
            metrics={
                "input_tokens": input_tokens,
                "cost_usd": cost_result["cost_usd"],
                "cost_per_input_token": cost_per_input_token,
            },
            metadata={
                "prompt_length": len(prompt),
                "output_tokens": cost_result["output_tokens"],
            },
        )


# Example 3: Composed test combining multiple tests
class ProductionReadiness(ComposedTest):
    """
    Combined test: performance + cost checks.

    Uses standard tests + custom thresholds.
    """

    async def run(self, provider, prompt):
        # Run multiple standard tests
        perf = await performance_profile(provider, prompt, runs=3)
        cost_eff = await cost_efficiency(provider, prompt, runs=3)

        # Extract key metrics
        latency_ms = perf["summary"]["avg_latency_ms"]
        throughput = perf["summary"]["tokens_per_second"]
        cost_per_ms = cost_eff["efficiency"]["cost_per_ms"]

        # Your pass/fail criteria
        passed = latency_ms < 500 and cost_per_ms < 0.00001

        metadata = provider.get_metadata()
        return TestResult(
            test_name="production_readiness",
            provider_name=metadata.provider,
            model_name=metadata.model_name,
            metrics={
                "latency_ms": latency_ms,
                "throughput": throughput,
                "cost_per_ms": cost_per_ms,
            },
            metadata={
                "full_performance": perf,
                "full_cost_efficiency": cost_eff,
            },
            passed=passed,
        )


# Example 4: Test with setup/teardown
class ContextStressTest(Test):
    """
    Test with lifecycle hooks.

    Demonstrates setup/teardown pattern.
    """

    def __init__(self):
        self.test_prompts = []

    async def setup(self):
        # Generate test data
        self.test_prompts = [
            "Short prompt",
            "Medium " * 100,
            "Long " * 1000,
        ]
        print("Setup: Generated test prompts")

    async def run(self, provider):
        results = []

        for i, prompt in enumerate(self.test_prompts):
            tokens = count_tokens(provider, prompt)
            latency = await measure_latency(provider, prompt)

            results.append(
                {"size": ["short", "medium", "long"][i], "tokens": tokens, **latency}
            )

        metadata = provider.get_metadata()
        return TestResult(
            test_name="context_stress",
            provider_name=metadata.provider,
            model_name=metadata.model_name,
            metrics={
                "results": results,
                "avg_latency": sum(r["mean_ms"] for r in results) / len(results),
            },
        )

    async def teardown(self):
        # Cleanup
        self.test_prompts = []
        print("Teardown: Cleaned up test data")


async def main():
    api_key = "your-api-key"  # Or use os.getenv("GEMINI_API_KEY")
    provider = GeminiProvider(model="gemini-1.5-flash", api_key=api_key)
    prompt = "Explain quantum computing"

    # Example 1: Simple custom test
    print("=== Example 1: Cost Per Millisecond ===")
    test1 = CostPerMs()
    result1 = await test1.execute(provider, prompt)
    print(f"Test: {result1.test_name}")
    print(f"Model: {result1.model_name}")
    print(f"Cost per ms: ${result1.metrics['cost_per_ms']:.8f}")
    print(f"Passed: {result1.passed}")

    # Example 2: Custom logic
    print("\n=== Example 2: Prompt Efficiency ===")
    test2 = PromptEfficiency()
    result2 = await test2.execute(provider, prompt)
    print(f"Cost per input token: ${result2.metrics['cost_per_input_token']:.6f}")

    # Example 3: Composed test
    print("\n=== Example 3: Production Readiness ===")
    test3 = ProductionReadiness()
    result3 = await test3.execute(provider, prompt)
    print(f"Latency: {result3.metrics['latency_ms']:.2f}ms")
    print(f"Cost per ms: ${result3.metrics['cost_per_ms']:.8f}")
    print(f"Production ready: {result3.passed}")

    # Example 4: With lifecycle
    print("\n=== Example 4: Context Stress Test ===")
    test4 = ContextStressTest()
    result4 = await test4.execute(provider)
    print(f"Average latency: {result4.metrics['avg_latency']:.2f}ms")

    # Compare results across tests (all have same format!)
    print("\n=== Comparing Test Results ===")
    all_results = [result1, result2, result3]
    for result in all_results:
        print(f"{result.test_name:25} | {result.model_name:20}")

    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())

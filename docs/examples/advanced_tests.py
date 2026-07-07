"""Examples of advanced testing: streaming, reliability, comparison."""

import asyncio

from blueberry.providers import GeminiProvider
from blueberry.tests import compare_models, error_rate, streaming_performance


async def main():
    api_key = "your-api-key"  # Or use os.getenv("GEMINI_API_KEY")

    # 1. Streaming performance - TTFT is critical for UX
    print("Testing streaming performance...")
    provider = GeminiProvider(model="gemini-1.5-flash", api_key=api_key)

    streaming = await streaming_performance(
        provider, "Write a short story about a robot", max_tokens=500, runs=3
    )

    print("\nStreaming Performance:")
    print(f"  Time to first token: {streaming['summary']['ttft_ms']:.2f}ms")
    print(
        f"  Throughput:          {streaming['summary']['tokens_per_second']:.2f} tok/s"
    )

    # 2. Error rate - reliability under load
    print("\n\nTesting reliability...")
    reliability = await error_rate(
        provider, "Hello world", attempts=20, timeout_seconds=10.0
    )

    print("\nReliability Test (20 attempts):")
    print(f"  Success rate:  {reliability['success_rate']:.1f}%")
    print(f"  Error rate:    {reliability['error_rate']:.1f}%")
    print(f"  Successful:    {reliability['successful']}")
    print(f"  Failed:        {reliability['failed']}")
    print(f"  Timeouts:      {reliability['timeouts']}")

    if reliability["errors"]:
        print("\n  Errors encountered:")
        for error in set(reliability["errors"]):
            count = reliability["errors"].count(error)
            print(f"    - {error} ({count}x)")

    # 3. Model comparison - compare multiple models at once
    print("\n\nComparing models...")

    providers = {
        "pro": GeminiProvider(model="gemini-1.5-pro", api_key=api_key),
        "flash": GeminiProvider(model="gemini-1.5-flash", api_key=api_key),
    }

    comparison = await compare_models(
        providers, "Explain artificial intelligence", runs=3, include_cost=True
    )

    print("\nModel Comparison:")
    print(f"{'Model':<15} {'Latency':<12} {'Cost':<15}")
    print("-" * 45)

    for name, result in comparison.items():
        latency = result["summary"]["avg_latency_ms"]
        cost = result["summary"].get("cost_usd", 0)
        print(f"{name:<15} {latency:>8.2f}ms    ${cost:.6f}")

    # Cleanup
    await provider.close()
    for p in providers.values():
        await p.close()


if __name__ == "__main__":
    asyncio.run(main())

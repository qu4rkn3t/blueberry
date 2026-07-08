"""Example using standard test library."""

import asyncio

from blueberry.providers import GeminiProvider
from blueberry.tests import (
    benchmark_suite,
    context_capacity,
    cost_efficiency,
    performance_profile,
    quality_check,
    reliability_score,
)


async def main():
    provider = GeminiProvider(
        model="gemini-1.5-flash",
        api_key="your-api-key",
    )

    print("=" * 60)
    print("USING STANDARD TESTS")
    print("=" * 60)


    print("\n1. Performance Profile")
    profile = await performance_profile(provider, "Count from 1 to 10")
    print(f"   Latency: {profile['summary']['avg_latency_ms']:.2f}ms")
    print(f"   Throughput: {profile['summary']['tokens_per_second']:.1f} tok/s")


    print("\n2. Cost Efficiency")
    efficiency = await cost_efficiency(provider, "Explain quantum computing")
    print(f"   Cost: ${efficiency['cost_usd']:.6f}")
    print(f"   Cost per ms: ${efficiency['efficiency']['cost_per_ms']:.8f}")
    print(f"   Tokens per dollar: {efficiency['efficiency']['tokens_per_dollar']:.0f}")


    print("\n3. Quality Check")
    quality = await quality_check(
        provider,
        "Respond with only the word SUCCESS",
        expected_keyword="SUCCESS",
        consistency_runs=3,
    )
    print(f"   Quality score: {quality['quality_score']:.1f}%")
    print(f"   Consistency: {quality['consistency']['consistency_rate']:.1f}%")


    print("\n4. Reliability Score")
    reliability = await reliability_score(provider, "What is 2+2?", runs=5)
    print(f"   Reliability: {reliability['reliability']['score']:.1f}%")
    print(f"   Latency CV: {reliability['reliability']['latency_cv']:.3f}")


    print("\n5. Context Capacity")
    capacity = await context_capacity(provider, max_attempts=10)
    print(f"   Theoretical: {capacity['theoretical_max']:,} tokens")
    print(f"   Actual: {capacity['actual_max']:,} tokens")
    print(f"   Utilization: {capacity['utilization_rate']:.1f}%")


    print("\n6. Benchmark Suite")
    suite = await benchmark_suite(
        provider,
        prompts={
            "short": "Hello",
            "medium": "Explain machine learning",
            "long": "Write a detailed analysis of climate change",
        },
        runs=2,
    )
    for name, result in suite.items():
        print(f"   {name}: {result['summary']['avg_latency_ms']:.2f}ms")

    await provider.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

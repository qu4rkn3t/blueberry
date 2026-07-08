"""Example showing how to customize standard tests."""

import asyncio

from blueberry.providers import GeminiProvider


from blueberry.tests import cost_efficiency, performance_profile


from blueberry.atomic import calculate_cost, measure_latency



async def use_as_is(provider):
    """Standard tests work out of the box."""
    return await performance_profile(provider, "Hello world")



async def customize_params(provider):
    """Pass different parameters to standard tests."""
    return await cost_efficiency(
        provider,
        "Explain AI",
        max_tokens=500,
        runs=10,
        temperature=0.5,
    )



async def extended_profile(provider, prompt):
    """Add your own metrics to standard test."""

    profile = await performance_profile(provider, prompt, runs=5)


    tokens = profile["throughput"]["total_tokens"]
    latency_ms = profile["latency"]["mean_ms"]
    tokens_per_ms = tokens / latency_ms if latency_ms > 0 else 0


    profile["custom_metrics"] = {"tokens_per_ms": tokens_per_ms}

    return profile



async def combined_analysis(provider, prompt):
    """Combine multiple standard tests."""
    perf = await performance_profile(provider, prompt)
    cost = await cost_efficiency(provider, prompt)

    return {
        "performance": perf,
        "cost": cost,
        "value_score": (
            perf["summary"]["tokens_per_second"] / cost["cost_usd"]
            if cost["cost_usd"] > 0
            else 0
        ),
    }



async def custom_test(provider, prompt):
    """
    Build your own test using atomic operations.

    This is how standard tests are built - you can do the same.
    """

    latency = await measure_latency(provider, prompt, runs=3)
    cost = await calculate_cost(provider, prompt, max_tokens=100)


    return {
        "latency_ms": latency["mean_ms"],
        "cost_usd": cost["cost_usd"],
        "my_custom_score": latency["mean_ms"] * cost["cost_usd"],
    }


async def main():
    provider = GeminiProvider(
        model="gemini-1.5-flash",
        api_key="your-api-key",
    )

    prompt = "Explain quantum computing"

    print("1. Use as-is:")
    result1 = await use_as_is(provider)
    print(f"   {result1['summary']}\n")

    print("2. Customize parameters:")
    result2 = await customize_params(provider)
    print(f"   Cost: ${result2['cost_usd']:.6f}\n")

    print("3. Extend standard test:")
    result3 = await extended_profile(provider, prompt)
    print(f"   Tokens/ms: {result3['custom_metrics']['tokens_per_ms']:.2f}\n")

    print("4. Combine standard tests:")
    result4 = await combined_analysis(provider, prompt)
    print(f"   Value score: {result4['value_score']:.2f}\n")

    print("5. Build your own:")
    result5 = await custom_test(provider, prompt)
    print(f"   Custom score: {result5['my_custom_score']:.6f}\n")

    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())

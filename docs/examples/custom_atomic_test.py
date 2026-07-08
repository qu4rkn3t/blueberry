"""Example showing how to combine atomic operations to create custom tests."""

import asyncio

from blueberry.atomic import calculate_cost, check_consistency, measure_latency
from blueberry.prompts import load_prompt
from blueberry.providers import GeminiProvider


async def cost_per_ms_test(provider, prompt):
    """
    Custom test: calculate cost per millisecond of latency.

    Combines latency and cost atomics to create new metric.
    """
    latency = await measure_latency(provider, prompt, runs=5)
    cost = await calculate_cost(provider, prompt, max_tokens=100)

    cost_per_ms = (
        cost["cost_usd"] / latency["mean_ms"] if latency["mean_ms"] > 0 else 0
    )

    return {
        "latency_ms": latency["mean_ms"],
        "cost_usd": cost["cost_usd"],
        "cost_per_ms": cost_per_ms,
        "efficiency_score": 1 / cost_per_ms if cost_per_ms > 0 else 0,
    }


async def reliability_test(provider, prompt):
    """
    Custom test: check both consistency and latency stability.

    Combines multiple atomics to assess reliability.
    """
    consistency = await check_consistency(provider, prompt, runs=5)
    latency = await measure_latency(provider, prompt, runs=10)


    consistency_score = consistency["consistency_rate"]
    latency_cv = latency["std_ms"] / latency["mean_ms"] if latency["mean_ms"] > 0 else 1
    reliability_score = consistency_score * (1 - min(latency_cv, 1))

    return {
        "consistency_rate": consistency_score,
        "latency_cv": latency_cv,
        "reliability_score": reliability_score,
    }


async def main():
    provider = GeminiProvider(
        model="gemini-1.5-flash",
        api_key="your-api-key",
    )

    prompt = load_prompt("What is 2+2? Answer with only the number.")


    print("Running cost efficiency test...")
    efficiency = await cost_per_ms_test(provider, prompt)
    print(f"  Efficiency score: {efficiency['efficiency_score']:.2f}")
    print(f"  Cost per ms: ${efficiency['cost_per_ms']:.8f}")

    print("\nRunning reliability test...")
    reliability = await reliability_test(provider, prompt)
    print(f"  Reliability score: {reliability['reliability_score']:.2f}")
    print(f"  Consistency: {reliability['consistency_rate']:.1f}%")

    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())

"""Compare multiple models using atomic functions."""

import asyncio

from blueberry.atomic import calculate_cost, measure_latency
from blueberry.providers import GeminiProvider


async def test_model(provider, name, prompt):
    """Test a single model."""
    latency = await measure_latency(provider, prompt, runs=3)
    cost = await calculate_cost(provider, prompt, max_tokens=200)

    return {
        "model": name,
        "latency_ms": latency["mean_ms"],
        "cost_usd": cost["cost_usd"],
        "tokens": cost["total_tokens"],
    }


async def main():
    prompt = "Explain quantum computing in simple terms"

    # Create providers
    providers = [
        (
            GeminiProvider(model="gemini-1.5-pro", api_key="your-key"),
            "gemini-pro",
        ),
        (
            GeminiProvider(model="gemini-1.5-flash", api_key="your-key"),
            "gemini-flash",
        ),
    ]

    # Test all models
    results = await asyncio.gather(
        *[test_model(provider, name, prompt) for provider, name in providers]
    )

    # Display comparison
    print("\nModel Comparison:")
    print("-" * 60)
    for result in results:
        print(f"{result['model']:15} | {result['latency_ms']:8.2f}ms | "
              f"${result['cost_usd']:.6f} | {result['tokens']} tokens")

    # Cleanup
    for provider, _ in providers:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())

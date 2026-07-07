"""Simple latency test using atomic functions."""

import asyncio

from blueberry.atomic import measure_latency
from blueberry.providers import GeminiProvider


async def main():
    provider = GeminiProvider(
        model="gemini-1.5-flash",
        api_key="your-api-key",  # Or use os.getenv("GEMINI_API_KEY")
    )

    prompt = "Count from 1 to 10"

    # Measure latency - simple function call
    result = await measure_latency(provider, prompt, runs=5)

    print(f"Average latency: {result['mean_ms']:.2f}ms")
    print(f"Min: {result['min_ms']:.2f}ms, Max: {result['max_ms']:.2f}ms")

    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())

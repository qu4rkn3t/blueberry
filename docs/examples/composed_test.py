"""Composed test using Test base class."""

import asyncio

from blueberry import Test
from blueberry.atomic import calculate_cost, measure_latency, measure_throughput
from blueberry.prompts import load_prompt
from blueberry.providers import GeminiProvider


class ComprehensiveTest(Test):
    """Test that measures latency, throughput, and cost."""

    async def run(self, provider, prompt):

        latency = await measure_latency(provider, prompt, runs=3)
        throughput = await measure_throughput(provider, prompt, runs=2)
        cost = await calculate_cost(provider, prompt, max_tokens=200)

        return {
            "latency_ms": latency["mean_ms"],
            "tokens_per_second": throughput["tokens_per_second"],
            "cost_usd": cost["cost_usd"],
            "total_tokens": cost["total_tokens"],
        }


async def main():
    provider = GeminiProvider(
        model="gemini-1.5-flash",
        api_key="your-api-key",
    )


    prompt = load_prompt("prompts/analyze_code.txt")



    test = ComprehensiveTest()
    result = await test.execute(provider, prompt)

    print("Test Results:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())

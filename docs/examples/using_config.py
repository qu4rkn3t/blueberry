"""Example: Using configuration file for providers."""

import asyncio

from blueberry.atomic import calculate_cost, measure_latency
from blueberry.config import load_config
from blueberry.providers import GeminiProvider, OpenAISpec


async def main():
    config = load_config("blueberry_config.yaml")


    gemini_flash = GeminiProvider.from_config(
        config, "gemini-1.5-flash", api_key="your-gemini-key"
    )

    gemini_pro = GeminiProvider.from_config(
        config, "gemini-1.5-pro", api_key="your-gemini-key"
    )


    from blueberry.config import get_model_config

    gpt4_config = get_model_config(config, "openai", "gpt-4")
    gpt4 = OpenAISpec(
        model=gpt4_config.model_name,
        api_key="your-openai-key",
        base_url=gpt4_config.base_url,
        context_window=gpt4_config.context_window,
        max_output_tokens=gpt4_config.max_output_tokens,
        input_cost_per_million=gpt4_config.input_cost_per_million,
        output_cost_per_million=gpt4_config.output_cost_per_million,
    )


    prompt = "Explain quantum computing in one sentence"

    print("=== Testing Gemini Flash ===")
    latency = await measure_latency(gemini_flash, prompt, runs=3)
    cost = await calculate_cost(gemini_flash, prompt)
    print(f"Latency: {latency['mean_ms']:.2f}ms")
    print(f"Cost: ${cost['cost_usd']:.6f}")

    print("\n=== Testing Gemini Pro ===")
    latency = await measure_latency(gemini_pro, prompt, runs=3)
    cost = await calculate_cost(gemini_pro, prompt)
    print(f"Latency: {latency['mean_ms']:.2f}ms")
    print(f"Cost: ${cost['cost_usd']:.6f}")


    await gemini_flash.close()
    await gemini_pro.close()
    await gpt4.close()


if __name__ == "__main__":
    asyncio.run(main())

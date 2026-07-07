"""Examples of statistical testing."""

import asyncio

from blueberry.providers import GeminiProvider
from blueberry.tests import latency_percentiles, statistical_comparison


async def main():
    # Setup providers
    api_key = "your-api-key"  # Or use os.getenv("GEMINI_API_KEY")

    pro = GeminiProvider(model="gemini-1.5-pro", api_key=api_key)
    flash = GeminiProvider(model="gemini-1.5-flash", api_key=api_key)

    prompt = "Explain quantum computing in one sentence"

    # 1. Latency percentiles - production-grade metrics
    print("Testing Flash model percentiles...")
    percentiles = await latency_percentiles(flash, prompt, runs=100)

    print("\nLatency Percentiles (100 runs):")
    print(f"  P50 (median): {percentiles['p50']:.2f}ms")
    print(f"  P95:          {percentiles['p95']:.2f}ms")
    print(f"  P99:          {percentiles['p99']:.2f}ms")
    print(f"  P99.9:        {percentiles['p999']:.2f}ms")
    print(f"  Mean:         {percentiles['mean']:.2f}ms")

    # 2. Statistical comparison - is Pro significantly faster?
    print("\n\nComparing Pro vs Flash with statistical significance...")
    comparison = await statistical_comparison(
        pro,
        flash,
        prompt,
        runs=50,  # Recommend 30+ for statistical validity
        alpha=0.05,  # 95% confidence
    )

    print("\nStatistical Comparison (50 runs each):")
    print(
        f"  {comparison['provider_a']['name']:20} {comparison['provider_a']['mean_ms']:8.2f}ms (mean)"
    )
    print(
        f"  {comparison['provider_b']['name']:20} {comparison['provider_b']['mean_ms']:8.2f}ms (mean)"
    )
    print(f"\n  Difference:         {abs(comparison['difference_ms']):.2f}ms")
    print(f"  Faster provider:    {comparison['faster_provider']}")
    print(f"  P-value:            {comparison['p_value']:.4f}")
    print(f"  Significant?        {comparison['statistically_significant']}")
    print(f"  Confidence:         {comparison['confidence'] * 100:.0f}%")

    if comparison["statistically_significant"]:
        print("\n  ✓ The difference IS statistically significant at 95% confidence")
    else:
        print("\n  ✗ The difference is NOT statistically significant")

    await pro.close()
    await flash.close()


if __name__ == "__main__":
    asyncio.run(main())

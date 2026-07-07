# Blueberry 🫐

Composable primitives for LLM performance testing.

Blueberry measures latency, cost, throughput, and quality through atomic functions, pre-built tests, and an optional `Test` framework for standardized, comparable results.

## Install

```bash
pip install -e .
```

Python 3.10+. Dependencies: `pydantic`, `openai`, `pyyaml`. Prompt optimization requires `pip install -e ".[optimizers]"` (`httpx`).

## Quick start

```python
import asyncio
from blueberry.atomic import measure_latency, calculate_cost
from blueberry.providers import GeminiProvider

async def main():
    provider = GeminiProvider(model="gemini-1.5-flash", api_key="...")
    prompt = "Explain quantum computing"

    latency = await measure_latency(provider, prompt, runs=5)
    cost = await calculate_cost(provider, prompt, max_tokens=200)

    print(f"{latency['mean_ms']:.0f}ms, ${cost['cost_usd']:.6f}")
    await provider.close()

asyncio.run(main())
```

Atomic functions are async and return dicts.

## Configuration

Model metadata (base URLs, pricing, context windows) is defined in YAML. No values are hardcoded.

```yaml
# blueberry_config.yaml
providers:
  gemini:
    models:
      gemini-1.5-flash:
        model_name: gemini-1.5-flash
        provider: gemini
        base_url: https://generativelanguage.googleapis.com/v1beta/openai/
        context_window: 1048576
        input_cost_per_million: 0.075
        output_cost_per_million: 0.30
```

```python
from blueberry.config import get_model_config, load_config
from blueberry.providers import GeminiProvider, OpenAISpec

config = load_config("blueberry_config.yaml")
gemini = GeminiProvider.from_config(config, "gemini-1.5-flash", api_key="...")

gpt4_cfg = get_model_config(config, "openai", "gpt-4")
gpt4 = OpenAISpec(
    model=gpt4_cfg.model_name,
    api_key="...",
    base_url=gpt4_cfg.base_url,
    context_window=gpt4_cfg.context_window,
    input_cost_per_million=gpt4_cfg.input_cost_per_million,
    output_cost_per_million=gpt4_cfg.output_cost_per_million,
)
```

Providers also accept configuration fields directly, without a YAML file.

## API overview

### Atomic operations (`blueberry.atomic`)

| Category | Functions |
|----------|-----------|
| Latency | `measure_latency`, `measure_ttft` |
| Throughput | `measure_throughput` |
| Tokens | `count_tokens`, `find_context_limit` |
| Cost | `estimate_cost`, `calculate_cost` |
| Quality | `check_consistency`, `check_instruction_following` |

```python
latency = await measure_latency(provider, prompt, runs=5)
cost = await calculate_cost(provider, prompt)
cost_per_ms = cost["cost_usd"] / latency["mean_ms"]
```

### Standard tests (`blueberry.tests`)

`performance_profile`, `benchmark_suite`, `cost_analysis`, `cost_efficiency`, `quality_check`, `reliability_score`, `context_capacity`, `prompt_stress_test`, `streaming_performance`, `error_rate`, `compare_models`, `latency_percentiles`, `statistical_comparison`

### Test framework (`blueberry.Test`)

Subclass `Test`, implement `run()`, return `TestResult`. The `execute()` method handles setup and teardown. See [docs/DEFINING_TESTS.md](docs/DEFINING_TESTS.md).

### Prompts (`blueberry.prompts`)

`load_prompt` loads from a file or returns the input string. `optimize_prompt` compresses prompts via Headroom (`httpx` required).

## Providers

All providers implement the `Provider` protocol: `complete`, `stream`, `count_tokens`, `get_metadata`.

| Class | Description |
|-------|-------------|
| `GeminiProvider` | Gemini via OpenAI-compatible endpoint |
| `OpenAISpec` | OpenAI-compatible APIs |

Custom providers implement the protocol defined in `blueberry.core`.

## Examples

| File | Topic |
|------|-------|
| `docs/examples/simple_latency.py` | Atomic operations |
| `docs/examples/using_config.py` | YAML configuration |
| `docs/examples/compare_models.py` | Multi-model comparison |
| `docs/examples/using_standard_tests.py` | Standard test library |
| `docs/examples/defining_custom_tests.py` | Custom `Test` subclasses |

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for a minimal end-to-end walkthrough.

## License

MIT

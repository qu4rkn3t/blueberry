# Quickstart

Get testing in 5 minutes.

## Install

```bash
pip install pydantic openai pyyaml
```

## Setup Configuration

Create `blueberry_config.yaml` with your model configurations:

```yaml
providers:
  gemini:
    models:
      gemini-1.5-flash:
        model_name: gemini-1.5-flash
        provider: gemini
        base_url: https://generativelanguage.googleapis.com/v1beta/openai/
        context_window: 1048576
        max_output_tokens: 8192
        input_cost_per_million: 0.075
        output_cost_per_million: 0.30
```

See README.md for complete configuration examples.

## Your First Test

Create `test.py`:

```python
import asyncio
from blueberry.atomic import measure_latency
from blueberry.config import load_config
from blueberry.providers import GeminiProvider

async def main():
    # Load configuration
    config = load_config("blueberry_config.yaml")
    
    # Create provider from config
    provider = GeminiProvider.from_config(
        config, "gemini-1.5-flash", api_key="your-api-key"
    )
    
    result = await measure_latency(
        provider,
        prompt="Count from 1 to 10",
        runs=3
    )
    
    print(f"Latency: {result['mean_ms']:.2f}ms")
    await provider.close()

asyncio.run(main())
```

Run it:
```bash
python test.py
```

## Available Atomics

```python
from blueberry.atomic import (
    # Latency
    measure_latency,    # End-to-end latency
    measure_ttft,       # Time to first token
    
    # Throughput
    measure_throughput, # Tokens per second
    
    # Tokens
    count_tokens,       # Count tokens
    find_context_limit, # Find max context
    
    # Cost
    estimate_cost,      # Estimate cost
    calculate_cost,     # Actual cost
    
    # Quality
    check_consistency,          # Output consistency
    check_instruction_following, # Follows instructions?
)
```

## Combine Them

```python
# Measure multiple things
latency = await measure_latency(provider, prompt, runs=5)
cost = await calculate_cost(provider, prompt, max_tokens=100)
throughput = await measure_throughput(provider, prompt)

# Create custom metrics
efficiency = cost["cost_usd"] / latency["mean_ms"]
```

## Use Prompts

```python
from blueberry.prompts import load_prompt, optimize_prompt

# Load from file
prompt = load_prompt("test.txt")

# Or optimize
prompt = optimize_prompt(
    "Explain quantum computing",
    optimizer="headroom",
    api_key="..."
)
```

## Compare Models

```python
providers = [
    GeminiProvider(model="gemini-1.5-pro", api_key=key),
    GeminiProvider(model="gemini-1.5-flash", api_key=key),
]

results = await asyncio.gather(
    *[measure_latency(p, prompt) for p in providers]
)

for i, result in enumerate(results):
    print(f"Provider {i}: {result['mean_ms']:.2f}ms")
```

## Structured Tests (Optional)

```python
from blueberry import Test

class MyTest(Test):
    async def run(self, provider, prompt):
        return {
            "latency": await measure_latency(provider, prompt),
            "cost": await calculate_cost(provider, prompt),
        }

test = MyTest()
result = await test.execute(provider, "Hello")
```

## Project Structure

Blueberry is just a library:

```python
your_project/
├── tests/
│   ├── test_latency.py      # Your latency tests
│   ├── test_cost.py         # Your cost tests
│   └── test_quality.py      # Your quality tests
├── prompts/
│   └── test_prompt.txt      # Your prompts
└── requirements.txt         # pydantic, openai
```

Write normal Python. Import atomics. Compose tests.

## Common Patterns

**Batch test multiple prompts:**
```python
prompts = ["prompt1", "prompt2", "prompt3"]
results = await asyncio.gather(
    *[measure_latency(provider, p) for p in prompts]
)
```

**Error handling:**
```python
try:
    result = await measure_latency(provider, prompt)
except Exception as e:
    print(f"Test failed: {e}")
```

**Custom test function:**
```python
async def test_efficiency(provider, prompt):
    latency = await measure_latency(provider, prompt)
    cost = await calculate_cost(provider, prompt)
    return cost["cost_usd"] / latency["mean_ms"]
```

## Next Steps

- See `examples/` for more patterns
- Read [MODEL.md](MODEL.md) for architecture
- Check `src/blueberry/atomic/` for all available functions

You're ready to test.

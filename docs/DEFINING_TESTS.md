# Defining Custom Tests: The Blueberry Way

This guide explains how to create your own tests using Blueberry's atomic operations and Test framework.

## Philosophy

**Blueberry provides:**
1. **9 atomic operations** - Building blocks (latency, cost, tokens, quality, throughput)
2. **Test base class** - Standardized structure for reusable tests
3. **TestResult format** - Makes all results comparable and composable

**You provide:**
- Custom metrics by combining atomics
- Domain-specific logic
- Pass/fail criteria

## The Three Levels

### Level 1: Direct Composition (No Test Class)

For one-off tests, just compose atomics directly:

```python
from blueberry.atomic import measure_latency, calculate_cost

# Your use case: "cost per millisecond"
latency = await measure_latency(provider, prompt, runs=5)
cost = await calculate_cost(provider, prompt)
efficiency = cost['cost_usd'] / latency['mean_ms']

print(f"Efficiency: ${efficiency:.8f} per ms")
```

**When to use:** Quick scripts, notebooks, one-time analysis

### Level 2: Test Class (Reusable)

For tests you'll run repeatedly, use the Test class:

```python
from blueberry import Test, TestResult
from blueberry.atomic import measure_latency, calculate_cost

class CostPerMs(Test):
    async def run(self, provider, prompt):
        latency = await measure_latency(provider, prompt, runs=5)
        cost = await calculate_cost(provider, prompt)
        
        metadata = provider.get_metadata()
        return TestResult(
            test_name="cost_per_ms",
            provider_name=metadata.provider,
            model_name=metadata.model_name,
            metrics={
                "cost_per_ms": cost['cost_usd'] / latency['mean_ms']
            }
        )

# Reuse it
test = CostPerMs()
result = await test.execute(provider, "Hello")
```

**When to use:** 
- Tests you run on multiple prompts/models
- Tests you want to share with teammates
- Tests you track over time

**Benefits:**
- `execute()` provides lifecycle (setup/teardown)
- `TestResult` makes results comparable
- Can be composed into larger test suites

### Level 3: ComposedTest (Combining Tests)

For combining multiple tests:

```python
from blueberry import ComposedTest, TestResult
from blueberry.tests import performance_profile, cost_efficiency

class ProductionReady(ComposedTest):
    async def run(self, provider, prompt):
        # Run multiple tests
        perf = await performance_profile(provider, prompt)
        cost = await cost_efficiency(provider, prompt)
        
        # Aggregate results
        passed = (
            perf['summary']['avg_latency_ms'] < 500 and
            cost['efficiency']['cost_per_ms'] < 0.00001
        )
        
        metadata = provider.get_metadata()
        return TestResult(
            test_name="production_ready",
            provider_name=metadata.provider,
            model_name=metadata.model_name,
            metrics={
                "latency_ms": perf['summary']['avg_latency_ms'],
                "cost_per_ms": cost['efficiency']['cost_per_ms'],
            },
            passed=passed
        )
```

**When to use:** Combining multiple standard tests with your criteria

## TestResult Format

All tests (standard and custom) return `TestResult`:

```python
TestResult(
    test_name: str,              # Unique identifier for your test
    provider_name: str,           # e.g., "gemini", "openai"
    model_name: str,              # e.g., "gemini-1.5-flash"
    metrics: dict,                # Your test's measurements
    metadata: dict = {},          # Optional: raw data, debug info
    passed: bool | None = None    # Optional: pass/fail threshold
)
```

**Why this format?**
- **Comparable**: All results have same structure
- **Composable**: Easy to aggregate multiple results
- **Filterable**: Query by provider, model, pass/fail
- **Serializable**: Can save to JSON, database

## Common Patterns

### Pattern 1: Custom Metric from Atomics

Combine atomics to create your metric:

```python
class TokensPerDollar(Test):
    async def run(self, provider, prompt):
        cost_result = await calculate_cost(provider, prompt)
        
        tokens_per_dollar = (
            cost_result['total_tokens'] / cost_result['cost_usd']
        )
        
        metadata = provider.get_metadata()
        return TestResult(
            test_name="tokens_per_dollar",
            provider_name=metadata.provider,
            model_name=metadata.model_name,
            metrics={"tokens_per_dollar": tokens_per_dollar}
        )
```

### Pattern 2: Pass/Fail Threshold

Add criteria to your test:

```python
class LatencyBudget(Test):
    def __init__(self, budget_ms: float = 500):
        self.budget_ms = budget_ms
    
    async def run(self, provider, prompt):
        latency = await measure_latency(provider, prompt, runs=10)
        
        metadata = provider.get_metadata()
        return TestResult(
            test_name="latency_budget",
            provider_name=metadata.provider,
            model_name=metadata.model_name,
            metrics={"mean_ms": latency['mean_ms']},
            passed=latency['mean_ms'] <= self.budget_ms
        )
```

### Pattern 3: Multi-Prompt Test

Test across different inputs:

```python
class MultiPromptConsistency(Test):
    async def run(self, provider, prompts: list[str]):
        results = []
        
        for prompt in prompts:
            consistency = await check_consistency(provider, prompt, runs=5)
            results.append({
                "prompt": prompt[:20] + "...",
                "consistency_rate": consistency['consistency_rate']
            })
        
        avg_consistency = sum(r['consistency_rate'] for r in results) / len(results)
        
        metadata = provider.get_metadata()
        return TestResult(
            test_name="multi_prompt_consistency",
            provider_name=metadata.provider,
            model_name=metadata.model_name,
            metrics={
                "average_consistency": avg_consistency,
                "per_prompt": results
            },
            passed=avg_consistency > 90
        )
```

### Pattern 4: With Setup/Teardown

Load data or initialize state:

```python
class DatabaseQueryTest(Test):
    def __init__(self):
        self.queries = []
    
    async def setup(self):
        # Load test queries from file
        with open("test_queries.txt") as f:
            self.queries = f.readlines()
    
    async def run(self, provider):
        results = []
        for query in self.queries:
            latency = await measure_latency(provider, query)
            results.append(latency['mean_ms'])
        
        metadata = provider.get_metadata()
        return TestResult(
            test_name="database_query_test",
            provider_name=metadata.provider,
            model_name=metadata.model_name,
            metrics={
                "avg_latency": sum(results) / len(results),
                "max_latency": max(results)
            }
        )
    
    async def teardown(self):
        self.queries = []
```

## Your Use Cases: How to Implement

Based on your requirements:

### "What prompts fit in 75% of context?"

```python
# Direct composition (no Test class needed)
from blueberry.atomic import count_tokens

def prompts_that_fit(provider, prompts, target_percentage=75):
    context = provider.get_metadata().context_window
    max_tokens = context * (target_percentage / 100)
    
    fitting = []
    for prompt in prompts:
        tokens = count_tokens(provider, prompt)
        if tokens <= max_tokens:
            fitting.append({
                "prompt": prompt,
                "tokens": tokens,
                "usage_pct": (tokens / context) * 100
            })
    
    return fitting
```

### "How expensive for 2000 token output?"

```python
from blueberry.atomic import estimate_cost, count_tokens

# Direct composition
prompt_tokens = count_tokens(provider, prompt)
cost = estimate_cost(provider, prompt, estimated_output_tokens=2000)
print(f"Estimated cost: ${cost:.6f}")

# As a test
class OutputCostTest(Test):
    def __init__(self, target_output_tokens: int):
        self.target_output_tokens = target_output_tokens
    
    async def run(self, provider, prompt):
        cost = estimate_cost(
            provider, 
            prompt, 
            estimated_output_tokens=self.target_output_tokens
        )
        
        metadata = provider.get_metadata()
        return TestResult(
            test_name="output_cost",
            provider_name=metadata.provider,
            model_name=metadata.model_name,
            metrics={
                "estimated_cost_usd": cost,
                "target_output_tokens": self.target_output_tokens
            }
        )
```

### "Tokens per dollar efficiency?"

```python
# Direct composition
from blueberry.atomic import calculate_cost

cost_result = await calculate_cost(provider, prompt)
efficiency = cost_result['total_tokens'] / cost_result['cost_usd']
print(f"Efficiency: {efficiency:,.0f} tokens/$")
```

## Comparing Results

Because all tests return `TestResult`, comparison is easy:

```python
# Run test on multiple models
providers = [gemini_pro, gemini_flash, gpt4]
test = CostPerMs()

results = await asyncio.gather(
    *[test.execute(p, prompt) for p in providers]
)

# Find cheapest
cheapest = min(results, key=lambda r: r.metrics['cost_per_ms'])
print(f"Winner: {cheapest.model_name}")

# Filter passed
passed = [r for r in results if r.passed]

# Group by provider
from collections import defaultdict
by_provider = defaultdict(list)
for r in results:
    by_provider[r.provider_name].append(r)
```

## Best Practices

### 1. Name Tests Clearly
```python
# Good
class CostPerMillisecond(Test): ...

# Bad  
class Test1(Test): ...
```

### 2. Include Context in Metrics
```python
# Good - includes both inputs and outputs
metrics={
    "latency_ms": latency['mean_ms'],
    "runs": 5,
    "prompt_tokens": tokens
}

# Bad - just the final number
metrics={"value": 123.45}
```

### 3. Use Metadata for Raw Data
```python
# Keep metrics clean, put details in metadata
return TestResult(
    test_name="quality_check",
    metrics={
        "consistency_rate": 95.0,
        "instruction_following": True
    },
    metadata={
        "all_outputs": [...],  # Raw data here
        "run_timestamps": [...]
    }
)
```

### 4. Make Thresholds Configurable
```python
class LatencyBudget(Test):
    def __init__(self, budget_ms: float = 500):  # ✓ Configurable
        self.budget_ms = budget_ms
    
    # Not this:
    # passed = latency < 500  # ✗ Hardcoded
```

## Summary

**The Blueberry Way:**
1. Use **atomic operations** for measurements
2. **Compose** them to create your metrics
3. Wrap in **Test class** for reusability
4. Return **TestResult** for comparability
5. Results work with standard tests - **same format**

You control the logic, Blueberry provides the primitives and structure.

See `examples/defining_custom_tests.py` for complete working examples.

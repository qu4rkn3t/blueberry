"""
Standard test library - common patterns built from atomic operations.

These are pre-composed tests for common scenarios. Use as-is or customize.
All are built from atomic operations, so you can see how they work.
"""

from blueberry.tests.benchmark import benchmark_suite, performance_profile
from blueberry.tests.comparison import compare_models
from blueberry.tests.cost import cost_analysis, cost_efficiency
from blueberry.tests.quality import quality_check, reliability_score
from blueberry.tests.reliability import error_rate
from blueberry.tests.scale import context_capacity, prompt_stress_test
from blueberry.tests.statistics import latency_percentiles, statistical_comparison
from blueberry.tests.streaming import streaming_performance

__all__ = [
    "benchmark_suite",
    "compare_models",
    "context_capacity",
    "cost_analysis",
    "cost_efficiency",
    "error_rate",
    "latency_percentiles",
    "performance_profile",
    "prompt_stress_test",
    "quality_check",
    "reliability_score",
    "statistical_comparison",
    "streaming_performance",
]

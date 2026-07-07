"""Configuration loading from YAML files."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RateLimitConfig(BaseModel):
    """Rate limit configuration."""

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    requests_per_day: int | None = None
    concurrent_requests: int | None = None


class BatchConfig(BaseModel):
    """Batch processing configuration."""

    supports_batch: bool = False
    max_batch_size: int | None = None
    batch_input_cost_multiplier: float = 1.0
    batch_output_cost_multiplier: float = 1.0


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    model_name: str
    provider: str
    base_url: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    supports_streaming: bool = True
    supports_function_calling: bool = False
    rate_limits: RateLimitConfig | None = None
    batch: BatchConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderConfig(BaseModel):
    """Configuration for a provider."""

    name: str
    models: dict[str, ModelConfig]


class OptimizerConfig(BaseModel):
    """Configuration for prompt optimizers."""

    name: str
    endpoint: str
    api_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with 'providers' and 'optimizers' keys
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    providers = {}
    for provider_name, provider_data in data.get("providers", {}).items():
        models = {}
        for model_name, model_data in provider_data.get("models", {}).items():
            models[model_name] = ModelConfig(**model_data)

        providers[provider_name] = ProviderConfig(name=provider_name, models=models)

    optimizers = {}
    for optimizer_name, optimizer_data in data.get("optimizers", {}).items():
        optimizers[optimizer_name] = OptimizerConfig(
            name=optimizer_name, **optimizer_data
        )

    return {"providers": providers, "optimizers": optimizers}


def get_model_config(config: dict[str, Any], provider: str, model: str) -> ModelConfig:
    """
    Get configuration for a specific model.

    Args:
        config: Loaded provider configuration
        provider: Provider name
        model: Model name

    Returns:
        ModelConfig for the specified model
    """
    providers = config.get("providers", {})
    if provider not in providers:
        raise ValueError(
            f"Provider '{provider}' not found in config. "
            f"Available: {list(providers.keys())}"
        )

    provider_config = providers[provider]
    if model not in provider_config.models:
        raise ValueError(
            f"Model '{model}' not found for provider '{provider}'. "
            f"Available: {list(provider_config.models.keys())}"
        )

    return provider_config.models[model]


def get_optimizer_config(config: dict[str, Any], optimizer: str) -> OptimizerConfig:
    """
    Get configuration for a specific optimizer.

    Args:
        config: Loaded configuration
        optimizer: Optimizer name

    Returns:
        OptimizerConfig for the specified optimizer
    """
    optimizers = config.get("optimizers", {})
    if optimizer not in optimizers:
        raise ValueError(
            f"Optimizer '{optimizer}' not found in config. "
            f"Available: {list(optimizers.keys())}"
        )

    return optimizers[optimizer]

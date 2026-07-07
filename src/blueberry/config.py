"""Configuration loading from YAML files."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


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
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderConfig(BaseModel):
    """Configuration for a provider."""

    name: str
    models: dict[str, ModelConfig]


def load_config(config_path: str | Path) -> dict[str, ProviderConfig]:
    """
    Load provider configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary mapping provider name to ProviderConfig
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

    return providers


def get_model_config(
    config: dict[str, ProviderConfig], provider: str, model: str
) -> ModelConfig:
    """
    Get configuration for a specific model.

    Args:
        config: Loaded provider configuration
        provider: Provider name
        model: Model name

    Returns:
        ModelConfig for the specified model
    """
    if provider not in config:
        raise ValueError(
            f"Provider '{provider}' not found in config. "
            f"Available: {list(config.keys())}"
        )

    provider_config = config[provider]
    if model not in provider_config.models:
        raise ValueError(
            f"Model '{model}' not found for provider '{provider}'. "
            f"Available: {list(provider_config.models.keys())}"
        )

    return provider_config.models[model]

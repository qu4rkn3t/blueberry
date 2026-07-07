"""Google Gemini model implementation."""

from blueberry.config import ModelConfig
from blueberry.core.provider import ModelMetadata
from blueberry.providers.specs.openai_spec import OpenAISpec


class GeminiProvider(OpenAISpec):
    """
    Google Gemini model implementation.

    Uses OpenAI-compatible API specification.
    All configuration via YAML config file.
    """

    @classmethod
    def from_config(
        cls,
        config: dict,
        model: str,
        api_key: str,
        **kwargs,
    ) -> "GeminiProvider":
        """
        Create provider from configuration.

        Args:
            config: Loaded config from load_config()
            model: Model name
            api_key: API key
            **kwargs: Override config values

        Returns:
            GeminiProvider instance
        """
        from blueberry.config import get_model_config

        model_config = get_model_config(config, "gemini", model)

        config_field_names = set(ModelConfig.model_fields.keys())
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in config_field_names}

        return cls(
            model=model_config.model_name,
            api_key=api_key,
            base_url=kwargs.get("base_url", model_config.base_url),
            context_window=kwargs.get("context_window", model_config.context_window),
            max_output_tokens=kwargs.get(
                "max_output_tokens", model_config.max_output_tokens
            ),
            input_cost_per_million=kwargs.get(
                "input_cost_per_million", model_config.input_cost_per_million
            ),
            output_cost_per_million=kwargs.get(
                "output_cost_per_million", model_config.output_cost_per_million
            ),
            **extra_kwargs,
        )

    def _create_metadata(self, **kwargs) -> ModelMetadata:
        """Create Gemini-specific metadata."""
        return ModelMetadata(
            model_name=self.model,
            provider="gemini",
            context_window=kwargs.get("context_window"),
            max_output_tokens=kwargs.get("max_output_tokens"),
            input_cost_per_million=kwargs.get("input_cost_per_million"),
            output_cost_per_million=kwargs.get("output_cost_per_million"),
            supports_streaming=True,
            supports_function_calling=True,
            metadata=kwargs,
        )

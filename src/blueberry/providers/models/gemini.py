"""Google Gemini model implementation."""

from blueberry.config import ModelConfig
from blueberry.core.provider import BatchCapabilities, ModelMetadata, RateLimitInfo
from blueberry.providers.specs.openai_spec import OpenAISpec


def _get_gemini_tokenizer(model: str):
    """
    Get the appropriate Gemini tokenizer.

    Returns None if google-genai is not installed.
    """
    try:
        from google import genai

        client = genai.Client()
        return client, model
    except ImportError:
        return None


class GeminiProvider(OpenAISpec):
    """
    Google Gemini model implementation.

    Uses OpenAI-compatible API specification.
    All configuration via YAML config file.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = _get_gemini_tokenizer(self.model)
        self._rate_limits_override = kwargs.get("rate_limits")
        self._batch_override = kwargs.get("batch")

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

        rate_limits = None
        if model_config.rate_limits:
            rate_limits = model_config.rate_limits.model_dump()

        batch = None
        if model_config.batch:
            batch = model_config.batch.model_dump()

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
            rate_limits=kwargs.get("rate_limits", rate_limits),
            batch=kwargs.get("batch", batch),
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

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Gemini's native tokenizer.

        Falls back to estimation if google-genai is not installed.
        """
        if self._tokenizer:
            try:
                client, model = self._tokenizer
                result = client.models.count_tokens(model=model, contents=text)
                return result.total_tokens
            except Exception:
                pass

        return super().count_tokens(text)

    def get_rate_limits(self) -> RateLimitInfo:
        """
        Get Gemini-specific rate limits.

        Uses config if provided, otherwise defaults to free tier limits.
        """
        if self._rate_limits_override:
            return RateLimitInfo(**self._rate_limits_override)

        return RateLimitInfo(
            requests_per_minute=15,
            tokens_per_minute=1_000_000,
            requests_per_day=1500,
            concurrent_requests=1,
        )

    def get_batch_capabilities(self) -> BatchCapabilities:
        """
        Get Gemini batch capabilities.

        Uses config if provided, otherwise defaults to batch support with 50% cost reduction.
        """
        if self._batch_override:
            return BatchCapabilities(**self._batch_override)

        return BatchCapabilities(
            supports_batch=True,
            max_batch_size=1000,
            batch_input_cost_multiplier=0.5,
            batch_output_cost_multiplier=0.5,
        )

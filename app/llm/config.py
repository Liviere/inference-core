"""
LLM Configuration module

Handles configuration for different LLM models and providers.
Supports OpenAI-compatible endpoints including open source models.
"""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

load_dotenv(
    dotenv_path=".env.test" if os.getenv("ENVIRONMENT") == "testing" else ".env"
)


class ModelProvider(str, Enum):
    """Supported LLM providers"""

    OPENAI = "openai"
    CUSTOM_OPENAI_COMPATIBLE = "custom_openai_compatible"
    GEMINI = "gemini"
    CLAUDE = "claude"


class ProviderConfig(BaseModel):
    """Configuration for a specific provider"""

    name: str
    requires_api_key: bool = False
    openai_compatible: bool = False
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None


class BatchRetryConfig(BaseModel):
    """Configuration for batch retry settings"""

    max_attempts: int = Field(
        default=5, ge=1, le=20, description="Maximum number of retry attempts"
    )
    base_delay: float = Field(
        default=2.0,
        ge=0.1,
        le=60.0,
        description="Base delay in seconds between retries",
    )
    max_delay: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="Maximum delay in seconds between retries",
    )

    @field_validator("max_delay")
    @classmethod
    def max_delay_must_be_greater_than_base(cls, v, info):
        if info.data.get("base_delay") is not None and v < info.data["base_delay"]:
            raise ValueError("max_delay must be greater than or equal to base_delay")
        return v


class BatchModelConfig(BaseModel):
    """Configuration for a batch-enabled model"""

    name: str = Field(..., description="Model name", max_length=100)
    mode: str = Field(
        ...,
        description="Processing mode (chat, embedding, completion, custom)",
        max_length=20,
    )
    max_prompts_per_batch: int = Field(
        default=100, ge=1, le=1000, description="Maximum prompts per batch"
    )
    pricing_tier: Optional[str] = Field(None, description="Pricing tier for the model")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        allowed_modes = ["chat", "embedding", "completion", "custom"]
        if v not in allowed_modes:
            raise ValueError(f'mode must be one of: {", ".join(allowed_modes)}')
        return v


class BatchProviderConfig(BaseModel):
    """Configuration for a batch-enabled provider"""

    enabled: bool = Field(
        default=True,
        description="Whether batch processing is enabled for this provider",
    )
    models: List[BatchModelConfig] = Field(
        default_factory=list, description="List of batch-enabled models"
    )


class BatchDefaultsConfig(BaseModel):
    """Default configuration for batch processing"""

    retry: BatchRetryConfig = Field(
        default_factory=BatchRetryConfig, description="Default retry configuration"
    )


class BatchConfig(BaseModel):
    """Main batch configuration"""

    enabled: bool = Field(
        default=True, description="Whether batch processing is globally enabled"
    )
    default_poll_interval_seconds: int = Field(
        default=30, ge=5, le=3600, description="Default polling interval in seconds"
    )
    default_dispatch_interval_seconds: int = Field(
        default=15,
        ge=1,
        le=3600,
        description="Default dispatch interval (CREATED -> submit) in seconds",
    )
    max_concurrent_provider_polls: int = Field(
        default=5, ge=1, le=20, description="Maximum concurrent provider polls"
    )
    defaults: BatchDefaultsConfig = Field(
        default_factory=BatchDefaultsConfig, description="Default batch settings"
    )
    providers: Dict[str, BatchProviderConfig] = Field(
        default_factory=dict, description="Provider-specific batch configurations"
    )

    def get_provider_models(self, provider: str) -> List[BatchModelConfig]:
        """Get list of batch-enabled models for a specific provider"""
        provider_config = self.providers.get(provider)
        if not provider_config or not provider_config.enabled:
            return []
        return provider_config.models

    def is_provider_enabled(self, provider: str) -> bool:
        """Check if batch processing is enabled for a specific provider"""
        if not self.enabled:
            return False
        provider_config = self.providers.get(provider)
        return provider_config is not None and provider_config.enabled

    def get_model_config(self, provider: str, model: str) -> Optional[BatchModelConfig]:
        """Get batch configuration for a specific model"""
        models = self.get_provider_models(provider)
        for model_config in models:
            if model_config.name == model:
                return model_config
        return None


class ModelConfig(BaseModel):
    """Configuration for a specific model"""

    name: str
    provider: ModelProvider
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = Field(default=2048, ge=1, le=1047576)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    timeout: int = Field(default=60, ge=1)

    model_config = ConfigDict(use_enum_values=True)


class LLMConfig:
    """Main LLM configuration class"""

    def __init__(self):
        self.providers: Dict[str, Any] = {}
        self.models: Dict[str, ModelConfig] = {}
        self.task_models: Dict[str, str] = {}
        self.enable_caching: bool = True
        self.cache_ttl: int = 3600
        self.max_concurrent_requests: int = 5
        self.enable_monitoring: bool = True
        self.default_timeout: int = 60
        self.retry_attempts: int = 3
        self.retry_delay: float = 1.0
        self.batch_config: BatchConfig = BatchConfig()  # Initialize with defaults
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent.parent.parent / "llm_config.yaml"

        # Load YAML configuration
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
            self._yaml_config = yaml_config  # Store for fallback method usage
        except FileNotFoundError:
            print(
                f"Warning: LLM config file not found at {config_path}. Using fallback configuration."
            )
            self._load_fallback_config()
            return
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config: {e}. Using fallback configuration.")
            self._load_fallback_config()
            return

        # Parse providers configuration
        self.providers = yaml_config.get("providers", {})

        # Parse models configuration
        models_config = yaml_config.get("models", {})
        self.models = {}

        for model_name, model_data in models_config.items():
            provider_name = model_data.get("provider")
            provider_config = self.providers.get(provider_name, {})
            provider_config = ProviderConfig(**provider_config)

            # Get API key from environment if required
            api_key = None
            if provider_config.requires_api_key:
                api_key_env = provider_config.api_key_env
                if api_key_env:
                    api_key = os.getenv(api_key_env)

            # Get base URL for OpenAI-compatible models
            base_url = provider_config.base_url
            if (
                provider_config.openai_compatible
                and provider_name != ModelProvider.OPENAI.value
            ):
                if not base_url:
                    logging.warning(
                        "Custom OpenAI-compatible models must have a base URL"
                    )

            self.models[model_name] = ModelConfig(
                name=model_name,
                provider=ModelProvider(provider_name),
                api_key=api_key,
                base_url=base_url,
                max_tokens=model_data.get("max_tokens", 2048),
                temperature=model_data.get("temperature", 0.7),
            )

        # Parse task model assignments
        tasks_config = yaml_config.get("tasks", {})
        self.task_models = {}

        for task_name, task_data in tasks_config.items():
            # Check for environment variable override
            env_var = (
                yaml_config.get("settings", {}).get("env_overrides", {}).get(task_name)
            )
            if env_var:
                override_model = os.getenv(env_var)
                if override_model:
                    self.task_models[task_name] = override_model
                    continue

            # Use primary model from config
            primary_model = task_data.get("primary")
            if primary_model:
                self.task_models[task_name] = primary_model

        # Parse general settings
        settings = yaml_config.get("settings", {})
        self.enable_caching = settings.get("enable_caching", True)
        self.cache_ttl = settings.get("cache_ttl_seconds", 3600)
        self.max_concurrent_requests = settings.get("max_concurrent_requests", 5)
        self.enable_monitoring = settings.get("enable_monitoring", True)
        self.default_timeout = settings.get("default_timeout", 60)
        self.retry_attempts = settings.get("retry_attempts", 3)
        self.retry_delay = settings.get("retry_delay", 1.0)

        # Parse batch configuration
        self._load_batch_config(yaml_config)

    def _load_batch_config(self, yaml_config: Dict[str, Any]):
        """Load and validate batch configuration from YAML"""
        batch_data = yaml_config.get("batch", {})

        try:
            # If batch section is missing or empty, use defaults
            if not batch_data:
                self.batch_config = BatchConfig()
                return

            # Parse provider configurations
            batch_providers = {}
            providers_data = batch_data.get("providers", {})

            for provider_name, provider_data in providers_data.items():
                # Validate provider exists in main providers config
                if provider_name not in self.providers:
                    logging.warning(
                        f"Batch configuration references unknown provider: {provider_name}"
                    )
                    continue

                # Parse models for this provider
                models_data = provider_data.get("models", [])
                models = []

                for model_data in models_data:
                    try:
                        model_config = BatchModelConfig(**model_data)
                        models.append(model_config)
                    except Exception as e:
                        logging.error(
                            f"Invalid batch model configuration for {provider_name}: {e}"
                        )
                        continue

                batch_providers[provider_name] = BatchProviderConfig(
                    enabled=provider_data.get("enabled", True), models=models
                )

            # Parse defaults
            defaults_data = batch_data.get("defaults", {})
            retry_data = defaults_data.get("retry", {})
            defaults = BatchDefaultsConfig(retry=BatchRetryConfig(**retry_data))

            # Create batch configuration
            self.batch_config = BatchConfig(
                enabled=batch_data.get("enabled", True),
                default_poll_interval_seconds=batch_data.get(
                    "default_poll_interval_seconds", 30
                ),
                default_dispatch_interval_seconds=batch_data.get(
                    "default_dispatch_interval_seconds", 15
                ),
                max_concurrent_provider_polls=batch_data.get(
                    "max_concurrent_provider_polls", 5
                ),
                defaults=defaults,
                providers=batch_providers,
            )

        except Exception as e:
            logging.error(
                f"Error loading batch configuration: {e}. Using default configuration."
            )
            self.batch_config = BatchConfig()

    def _load_fallback_config(self):
        """Load fallback configuration when YAML file is not available"""

        # Default model configurations
        self.models = {
            "gpt-5": ModelConfig(
                name="gpt-5",
                provider=ModelProvider.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=8192,
                temperature=0.7,
            ),
            "gpt-5-mini": ModelConfig(
                name="gpt-5-mini",
                provider=ModelProvider.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=8192,
                temperature=0.7,
            ),
            " gpt-5-nano": ModelConfig(
                name="gpt-5-nano",
                provider=ModelProvider.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=8192,
                temperature=0.7,
            ),
        }

        # Default model preferences for different tasks
        self.task_models = {
            "explain": "gpt-5-mini",
            "conversation": "gpt-5-mini",
        }

        # General settings
        self.enable_caching = os.getenv("LLM_ENABLE_CACHING", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("LLM_CACHE_TTL", "3600"))  # 1 hour
        self.max_concurrent_requests = int(os.getenv("LLM_MAX_CONCURRENT", "5"))
        self.enable_monitoring = (
            os.getenv("LLM_ENABLE_MONITORING", "true").lower() == "true"
        )

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return self.models.get(model_name)

    def get_task_model(self, task: str) -> str:
        """Get preferred model for a specific task"""
        return self.task_models.get(task, "gpt-5-mini")

    def add_custom_model(self, model_config: ModelConfig):
        """Add a custom model configuration"""
        self.models[model_config.name] = model_config

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available"""
        config = self.get_model_config(model_name)
        if not config:
            return False
        provider_config = self.get_provider_config(config.provider)

        # For OpenAI and DeepInfra models, check if API key is available
        if provider_config.requires_api_key:
            return bool(config.api_key and config.api_key.strip())

        # Custom OpenAI-compatible endpoints
        if config.provider == ModelProvider.CUSTOM_OPENAI_COMPATIBLE:
            return bool(config.base_url and config.base_url.strip())

        # Gemini (explicit check retained though requires_api_key should cover)
        if config.provider == ModelProvider.GEMINI:
            return bool(config.api_key and config.api_key.strip())

        return False

    def get_task_model_with_fallback(self, task: str) -> str:
        """Get preferred model for a task, with fallback to available models"""
        primary_model = self.get_task_model(task)

        # Check if primary model is available
        if self.is_model_available(primary_model):
            return primary_model

        # Try fallback models from YAML config
        if hasattr(self, "_yaml_config"):
            task_config = self._yaml_config.get("tasks", {}).get(task, {})
            fallback_models = task_config.get("fallback", [])

            for fallback_model in fallback_models:
                fallback_model_str = str(fallback_model)
                if self.is_model_available(fallback_model_str):
                    return fallback_model_str

        # Last resort: find any available model
        for model_name in self.models.keys():
            if self.is_model_available(model_name):
                return model_name

        # If nothing is available, return the primary model anyway
        return primary_model

    def get_provider_config(self, provider_name: str) -> ProviderConfig:
        """Get provider configuration"""
        provider_config: Dict[str, Any] = self.providers.get(provider_name, {})
        return ProviderConfig(**provider_config)

    def get_provider_runtime_config(self, provider_name: str) -> Dict[str, Any]:
        """Build runtime provider configuration for batch providers.

        Combines raw provider YAML definition with resolved API key (from env var
        or inline value if present) so that provider implementations (e.g. batch
        providers) receive a flat dict containing 'api_key' and any other hints
        such as base_url.

        Args:
            provider_name: Name of provider (e.g. "openai")

        Returns:
            Dict with runtime config (may be empty if unknown provider)
        """
        provider_raw: Dict[str, Any] = self.providers.get(provider_name, {}) or {}
        runtime: Dict[str, Any] = dict(provider_raw)  # shallow copy
        api_key = runtime.get("api_key")
        # Resolve via ProviderConfig model to access api_key_env metadata
        try:
            provider_cfg = ProviderConfig(**provider_raw) if provider_raw else None
        except Exception:
            provider_cfg = None
        if (not api_key) and provider_cfg and provider_cfg.requires_api_key:
            if provider_cfg.api_key_env:
                env_val = os.getenv(provider_cfg.api_key_env)
                if env_val:
                    api_key = env_val
        if api_key:
            runtime["api_key"] = api_key
        return runtime

    def list_available_models(self) -> List[str]:
        """List all available models"""
        return [name for name in self.models.keys() if self.is_model_available(name)]

    def list_models_by_task(self, task: str) -> List[str]:
        """List models suitable for a specific task"""
        # Get primary and fallback models for the task
        models = []

        primary = self.get_task_model(task)
        if primary:
            models.append(primary)

        if hasattr(self, "_yaml_config"):
            task_config = self._yaml_config.get("tasks", {}).get(task, {})
            fallback_models = task_config.get("fallback", [])
            models.extend(fallback_models)

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for model in models:
            if model not in seen and model in self.models:
                seen.add(model)
                result.append(model)

        return result

    def is_development_mode(self) -> bool:
        """Check if we're running in development mode (no real API keys)"""
        return not any(self.is_model_available(model) for model in self.models.keys())

    def get_model_debug_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed debug information about a model's availability"""
        config = self.get_model_config(model_name)
        if not config:
            return {"error": "Model not found"}

        provider_config = self.get_provider_config(config.provider)

        debug_info = {
            "name": model_name,
            "provider": config.provider.name,
            "requires_api_key": provider_config.requires_api_key,
            "api_key_present": config.api_key is not None,
            "api_key_is_placeholder": False,
            "base_url": config.base_url,
            "available": self.is_model_available(model_name),
        }

        if config.api_key:
            debug_info["api_key_is_placeholder"] = (
                config.api_key.startswith("your-")
                or config.api_key == "your-openai-api-key-here"
            )
            debug_info["api_key_preview"] = (
                config.api_key[:10] + "..."
                if len(config.api_key) > 10
                else config.api_key
            )

        return debug_info


# Global configuration instance
llm_config = LLMConfig()

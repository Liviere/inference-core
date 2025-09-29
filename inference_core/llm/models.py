"""
LLM Models Factory

Factory class for creating and managing different LLM model instances.
Supports multiple providers through OpenAI-compatible interfaces.
"""

import logging
from typing import Any, Dict, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .config import LLMConfig, ModelConfig, ModelProvider
from .param_policy import normalize_params

logger = logging.getLogger(__name__)


class LLMModelFactory:
    """Factory for creating LLM model instances"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._model_cache: Dict[str, BaseChatModel] = {}

    def create_model(self, model_name: str, **kwargs) -> Optional[BaseChatModel]:
        """
        Create a model instance

        Args:
            model_name: Name of the model to create
            **kwargs: Additional parameters to override model config

        Returns:
            BaseChatModel instance or None if model cannot be created
        """
        # Check cache first
        cache_key = f"{model_name}_{hash(str(sorted(kwargs.items())))}"
        if cache_key in self._model_cache:
            logger.debug(f"Using cached model instance for {model_name}")
            return self._model_cache[cache_key]

        model_config = self.config.get_model_config(model_name)
        if not model_config:
            logger.error(f"Model configuration not found: {model_name}")
            return None

        try:
            # Ensure streaming flag is preserved (some providers require 'streaming=True' at init)
            if "streaming" not in kwargs:
                kwargs["streaming"] = True if kwargs.get("callbacks") else False

            model = self._create_model_instance(model_config, **kwargs)

            # Force-enable attribute if it exists but got lost
            if model and kwargs.get("callbacks") and hasattr(model, "streaming"):
                try:
                    setattr(model, "streaming", True)
                    setattr(model, "stream_usage", True)
                except Exception:
                    logger.debug("Failed to set streaming attributes on model instance")
            if model and self.config.enable_caching:
                self._model_cache[cache_key] = model
            return model
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {str(e)}")
            return None

    def _create_model_instance(
        self, config: ModelConfig, **kwargs
    ) -> Optional[BaseChatModel]:
        """Create model instance based on provider"""

        # Merge config with kwargs
        # Start with provided kwargs (new dynamic params like reasoning_effort, verbosity etc.)
        raw_params: Dict[str, Any] = dict(kwargs)

        # Backward compatibility: fill legacy sampling params only if not explicitly provided
        if "temperature" not in raw_params:
            raw_params["temperature"] = config.temperature
        if "max_tokens" not in raw_params:
            raw_params["max_tokens"] = config.max_tokens
        if "top_p" not in raw_params:
            raw_params["top_p"] = config.top_p
        if "frequency_penalty" not in raw_params:
            raw_params["frequency_penalty"] = config.frequency_penalty
        if "presence_penalty" not in raw_params:
            raw_params["presence_penalty"] = config.presence_penalty
        # Unified timeout key (internal) -> request_timeout or provider mapping via policy rename
        if "request_timeout" not in raw_params and "timeout" in raw_params:
            raw_params["request_timeout"] = raw_params.pop("timeout")
        elif "request_timeout" not in raw_params:
            raw_params["request_timeout"] = config.timeout

        # Normalize parameters for the specific provider
        try:
            model_params = normalize_params(
                config.provider, raw_params, model_name=config.name
            )
        except ValueError as e:
            logger.error(f"Parameter normalization failed: {e}")
            return None

        if config.provider == ModelProvider.OPENAI:
            return self._create_openai_model(config, model_params)

        elif config.provider in [ModelProvider.CUSTOM_OPENAI_COMPATIBLE]:
            return self._create_openai_compatible_model(config, model_params)

        elif config.provider == ModelProvider.GEMINI:
            return self._create_gemini_model(config, model_params)

        elif config.provider == ModelProvider.CLAUDE:
            return self._create_claude_model(config, model_params)

        else:
            logger.error(f"Unsupported provider: {config.provider}")
            return None

    def _create_openai_model(
        self, config: ModelConfig, params: Dict[str, Any]
    ) -> Optional[ChatOpenAI]:
        """Create OpenAI model instance"""
        if not config.api_key:
            logger.error("OpenAI API key not provided")
            return None

        try:
            api_key = SecretStr(config.api_key) if config.api_key else None
            return ChatOpenAI(model=config.name, api_key=api_key, **params)
        except Exception as e:
            logger.error(f"Failed to create OpenAI model: {str(e)}")
            return None

    def _create_openai_compatible_model(
        self, config: ModelConfig, params: Dict[str, Any]
    ) -> Optional[ChatOpenAI]:
        """Create OpenAI-compatible model instance (LM Studio, etc.)"""
        try:
            api_key_str = config.api_key or "not-needed"
            api_key = SecretStr(api_key_str)
            return ChatOpenAI(
                model=config.name,
                base_url=config.base_url,
                api_key=api_key,
                **params,
            )
        except Exception as e:
            logger.error(f"Failed to create OpenAI-compatible model: {str(e)}")
            return None

    def _create_gemini_model(
        self, config: ModelConfig, params: Dict[str, Any]
    ) -> Optional[ChatGoogleGenerativeAI]:
        """Create Gemini model instance.

        Requires GOOGLE_API_KEY environment variable (loaded earlier into config.api_key).
        """
        if not config.api_key:
            logger.error("Gemini API key (GOOGLE_API_KEY) not provided")
            return None
        try:
            # Parameters are already normalized by param_policy
            return ChatGoogleGenerativeAI(
                model=config.name, api_key=config.api_key, **params
            )
        except Exception as e:
            logger.error(f"Failed to create Gemini model: {str(e)}")
            return None

    def _create_claude_model(
        self, config: ModelConfig, params: Dict[str, Any]
    ) -> Optional[ChatAnthropic]:
        """Create Claude (Anthropic) model instance.

        Parameters are already normalized by param_policy.
        """
        if not config.api_key:
            logger.error("Anthropic API key (ANTHROPIC_API_KEY) not provided")
            return None
        try:
            # Parameters are already normalized by param_policy
            return ChatAnthropic(model=config.name, api_key=config.api_key, **params)
        except Exception as e:
            logger.error(f"Failed to create Claude model: {str(e)}")
            return None

    def get_available_models(self) -> Dict[str, bool]:
        """Get list of available models with their availability status"""
        available_models = {}
        for model_name in self.config.models.keys():
            available_models[model_name] = self.config.is_model_available(model_name)
        return available_models

    def clear_cache(self):
        """Clear the model cache"""
        self._model_cache.clear()
        logger.info("Model cache cleared")

    def get_model_for_task(self, task: str, **kwargs) -> Optional[BaseChatModel]:
        """Get the preferred model for a specific task"""
        model_name = self.config.get_task_model(task)
        return self.create_model(model_name, **kwargs)


# Global factory instance
def get_model_factory() -> LLMModelFactory:
    """Get global model factory instance"""
    from .config import llm_config

    return LLMModelFactory(llm_config)

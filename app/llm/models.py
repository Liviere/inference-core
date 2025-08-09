"""
LLM Models Factory

Factory class for creating and managing different LLM model instances.
Supports multiple providers through OpenAI-compatible interfaces.
"""

import logging
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .config import LLMConfig, ModelConfig, ModelProvider

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
            model = self._create_model_instance(model_config, **kwargs)
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
        model_params = {
            "temperature": kwargs.get("temperature", config.temperature),
            "max_tokens": kwargs.get("max_tokens", config.max_tokens),
            "top_p": kwargs.get("top_p", config.top_p),
            "frequency_penalty": kwargs.get(
                "frequency_penalty", config.frequency_penalty
            ),
            "presence_penalty": kwargs.get("presence_penalty", config.presence_penalty),
            "request_timeout": kwargs.get("timeout", config.timeout),
        }

        if config.provider == ModelProvider.OPENAI:
            return self._create_openai_model(config, model_params)

        elif config.provider in [ModelProvider.CUSTOM_OPENAI_COMPATIBLE]:
            return self._create_openai_compatible_model(config, model_params)

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

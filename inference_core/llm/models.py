"""
LLM Models Factory

Factory class for creating and managing different LLM model instances.
Supports multiple providers through OpenAI-compatible interfaces.
"""

import logging
from contextlib import contextmanager
from contextvars import ContextVar
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
            user_callbacks = kwargs.get("callbacks")
            # Ensure streaming flag is preserved (some providers require 'streaming=True' at init)
            if "streaming" not in kwargs:
                kwargs["streaming"] = True if user_callbacks else False

            model = self._create_model_instance(model_config, **kwargs)

            # Some providers (or our param normalization) may strip callbacks; re-attach if missing
            if model and user_callbacks:
                try:
                    # If underlying constructor accepted callbacks they are already active.
                    # For safety, if model has 'callbacks' attr and it's empty, assign ours.
                    existing = getattr(model, "callbacks", None)
                    if existing is None or (
                        isinstance(existing, list) and not existing
                    ):
                        setattr(model, "callbacks", list(user_callbacks))
                except Exception:
                    logger.debug(
                        "Could not introspect/assign callbacks on model instance"
                    )

            # Force-enable streaming attributes if present
            if model and hasattr(model, "streaming"):
                try:
                    setattr(model, "streaming", True)
                    # Some LangChain chat models honor stream_usage for per-chunk usage
                    if not getattr(model, "stream_usage", False):
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
        # Honor task override (used by LLMService to map custom task_type to model)
        effective_task = _TASK_OVERRIDE.get() or task
        model_name = self.config.get_task_model(effective_task)
        return self.create_model(model_name, **kwargs)

    def get_agent_model_name(self, agent_name: BaseChatModel) -> Optional[str]:
        """Get the preferred model name for a specific agent"""
        effective_agent = _AGENT_OVERRIDE.get() or agent_name
        return self.config.get_agent_model(effective_agent)

    def get_model_for_agent(self, agent_name: str, **kwargs) -> Optional[BaseChatModel]:
        """Get the preferred model for a specific agent"""
        # Honor agent override (used by AgentService to map custom agent_type to model)
        model_name = self.get_agent_model_name(agent_name)
        # Extend cache key with agent to avoid reuse between agents
        cache_key = f"{model_name}_{agent_name}_{hash(str(sorted(kwargs.items())))}"
        if cache_key in self._model_cache:
            logger.debug(
                f"Using cached model instance for {model_name} (agent: {agent_name})"
            )
            return self._model_cache[cache_key]
        model = self.create_model(model_name, **kwargs)
        if model and self.config.enable_caching:
            self._model_cache[cache_key] = model
        return model


# Global factory instance
def get_model_factory() -> LLMModelFactory:
    """Get global model factory instance"""
    from .config import get_llm_config

    return LLMModelFactory(get_llm_config())


# -------- Task override support (thread/async-task local) --------
# LLMService sets this to ensure model selection honors custom task types
_TASK_OVERRIDE: ContextVar[Optional[str]] = ContextVar(
    "llm_task_override", default=None
)

# -------- Agent override support (thread/async-task local) --------
# AgentService sets this to ensure model selection honors custom agent types
_AGENT_OVERRIDE: ContextVar[Optional[str]] = ContextVar(
    "llm_agent_override", default=None
)


def current_task_override() -> Optional[str]:
    """Return current effective task override if set."""
    return _TASK_OVERRIDE.get()


def current_agent_override() -> Optional[str]:
    """Return current effective agent override if set."""
    return _AGENT_OVERRIDE.get()


@contextmanager
def task_override(task: Optional[str]):
    """Temporarily override the task used for default model resolution.

    Usage:
        with task_override("my_custom_task"):
            # any factory.get_model_for_task("chat") will resolve using "my_custom_task"
            ...
    """
    token = _TASK_OVERRIDE.set(task)
    try:
        yield
    finally:
        _TASK_OVERRIDE.reset(token)


@contextmanager
def agent_override(agent: Optional[str]):
    """Temporarily override the agent used for default model resolution.

    Usage:
        with agent_override("my_custom_agent"):
            # any factory.get_model_for_agent("default_agent") will resolve using "my_custom_agent"
            ...
    """
    token = _AGENT_OVERRIDE.set(agent)
    try:
        yield
    finally:
        _AGENT_OVERRIDE.reset(token)

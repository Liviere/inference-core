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
from langchain_community.chat_models.deepinfra import ChatDeepInfra
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
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

        # Start with config params (including extras)
        # Exclude internal fields specific to ModelConfig infrastructure
        exclude_fields = {
            "name",
            "provider",
            "api_key",
            "base_url",
            "pricing",
            "display_name",
            "description",
            "reasoning_config",
        }
        # Dump config to dict, which now includes extra fields (nested dicts etc.)
        config_params = config.model_dump(exclude=exclude_fields)

        # Merge config with kwargs
        # kwargs take precedence over config_params
        raw_params: Dict[str, Any] = {**config_params, **kwargs}

        # Extract reasoning flags before normalization — they are not regular
        # provider params and must bypass normalize_params() because they
        # contain nested/structured kwargs (e.g. {reasoning: {effort: "low"}}).
        reasoning_output = raw_params.pop("reasoning_output", False)
        # reasoning_config from kwargs overrides the one on ModelConfig
        reasoning_config = raw_params.pop("reasoning_config", config.reasoning_config)

        # Handle 'timeout' which is in config but might need to be 'request_timeout'
        # depending on provider policy, OR just standard normalization.
        # Below logic preserves existing behavior for timeout mapping
        if "request_timeout" not in raw_params:
            if "timeout" in raw_params:
                # Use timeout if provided (from config or kwargs)
                raw_params["request_timeout"] = raw_params.pop("timeout")
            elif hasattr(config, "timeout"):
                # Fallback to config timeout if still missing (though model_dump should have it)
                raw_params["request_timeout"] = config.timeout

        # Normalize parameters for the specific provider
        try:
            model_params = normalize_params(
                config.provider, raw_params, model_name=config.name
            )
        except ValueError as e:
            logger.error(f"Parameter normalization failed: {e}")
            return None

        # Merge reasoning kwargs into model_params when reasoning is requested
        # and the model defines a reasoning_config.
        if reasoning_output and reasoning_config:
            model_params.update(reasoning_config)
            logger.debug(
                "Reasoning output enabled for '%s' — merged reasoning_config: %s",
                config.name,
                list(reasoning_config.keys()),
            )

        if config.provider == ModelProvider.OPENAI:
            return self._create_openai_model(config, model_params)

        elif config.provider in [ModelProvider.CUSTOM_OPENAI_COMPATIBLE]:
            return self._create_openai_compatible_model(config, model_params)

        elif config.provider == ModelProvider.GEMINI:
            return self._create_gemini_model(config, model_params)

        elif config.provider == ModelProvider.CLAUDE:
            # Claude (Anthropic) does not allow `temperature` when extended thinking
            # is enabled — it must be either omitted or set to exactly 1.
            # Remove it here so the default (1) is used by the API.
            if "thinking" in reasoning_config:
                removed_temp = model_params.pop("temperature", None)
                if removed_temp is not None:
                    logger.debug(
                        "Removed 'temperature' from Claude model params (extended thinking enabled)"
                    )

                # `max_tokens` must be greater than `thinking.budget_token
                thinking_budget = reasoning_config.get("thinking", {}).get(
                    "budget_tokens"
                )
                max_tokens = model_params.get("max_tokens")
                if thinking_budget and max_tokens and max_tokens <= thinking_budget:
                    # If max_tokens is not greater than thinking budget, increase it to be so.
                    model_params["max_tokens"] = thinking_budget + 1000
                    logger.debug(
                        "Adjusted 'max_tokens' to be greater than 'thinking.budget_tokens' for Claude model"
                    )

            return self._create_claude_model(config, model_params)

        elif config.provider == ModelProvider.DEEPINFRA:
            return self._create_deepinfra_model(config, model_params)

        elif config.provider == ModelProvider.OLLAMA:
            return self._create_ollama_model(config, model_params)

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

    def _create_deepinfra_model(
        self, config: ModelConfig, params: Dict[str, Any]
    ) -> Optional[ChatDeepInfra]:
        """Create DeepInfra model instance via the dedicated LangChain integration.

        Uses ChatDeepInfra from langchain_community instead of the generic
        OpenAI-compatible wrapper, giving access to DeepInfra-specific features
        like native tool calling and proper token tracking.
        """
        if not config.api_key:
            logger.error("DeepInfra API token (DEEPINFRA_API_TOKEN) not provided")
            return None
        try:
            return ChatDeepInfra(
                model=config.name,
                deepinfra_api_token=config.api_key,
                **params,
            )
        except Exception as e:
            logger.error(f"Failed to create DeepInfra model: {str(e)}")
            return None

    def _create_ollama_model(
        self, config: ModelConfig, params: Dict[str, Any]
    ) -> Optional[ChatOllama]:
        """Create Ollama chat model instance backed by a local/remote Ollama server."""

        try:
            if config.base_url:
                return ChatOllama(model=config.name, base_url=config.base_url, **params)
            return ChatOllama(model=config.name, **params)
        except Exception as e:
            logger.error(f"Failed to create Ollama model: {str(e)}")
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

        # Forward reasoning_output from agent config when not explicitly set
        if "reasoning_output" not in kwargs:
            agent_config = self.config.get_specific_agent_config(agent_name)
            if agent_config and getattr(agent_config, "reasoning_output", False):
                kwargs["reasoning_output"] = True

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

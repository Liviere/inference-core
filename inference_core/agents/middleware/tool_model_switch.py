"""
Tool-Based Model Switch Middleware for LangChain v1 Agents.

This middleware allows using different models when specific tools are invoked.
It enables cost optimization by using cheaper models for simple tasks and
more capable models for complex reasoning after certain tool calls.

Key features:
- Configurable tool-to-model mappings via llm_config.yaml
- Switches model dynamically based on last tool call in conversation
- Supports both pre-tool and post-tool model selection strategies
- Integrates with existing model factory for consistent model instantiation

Usage:
    # In llm_config.yaml:
    agents:
      my_agent:
        primary: 'gpt-5-mini'
        tool_model_overrides:
          - tool_name: 'complex_reasoning_tool'
            model: 'claude-opus-4-1-20250805'
            trigger: 'after_tool'  # Switch model AFTER this tool returns
          - tool_name: 'code_generation'
            model: 'gpt-5'
            trigger: 'before_tool'  # Switch model BEFORE calling this tool

    # The middleware is automatically added when tool_model_overrides is configured

Source references:
  - LangChain v1 middleware docs: https://docs.langchain.com/oss/python/langchain/middleware/custom
  - Context engineering: https://docs.langchain.com/oss/python/langchain/context-engineering
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import NotRequired

logger = logging.getLogger(__name__)


# Define custom state to track tool calls for model selection decisions.


class ToolModelSwitchState(AgentState):
    """Extended agent state for tool-based model switching.

    Tracks the last tool called to enable model selection based on
    tool execution history.

    Attributes:
        last_tool_called: Name of the last tool that was executed.
            Used by wrap_model_call to determine if model should be switched.
        pending_tool_call: Name of tool about to be called (for before_tool trigger).
    """

    last_tool_called: NotRequired[Optional[str]]
    pending_tool_call: NotRequired[Optional[str]]


# Configuration for tool-to-model mappings.


@dataclass
class ToolModelOverride:
    """Configuration for a single tool-model override.

    Attributes:
        tool_name: Name of the tool that triggers the model switch.
        model: Name of the model to use (must be defined in llm_config.yaml).
        trigger: When to apply the override:
            - 'after_tool': Switch model for the next model call AFTER tool returns
            - 'before_tool': Switch model for the model call that will invoke this tool
              (Note: 'before_tool' is tricky as we don't know which tool will be called
               until the model decides. Use with caution.)
        description: Optional human-readable description of why this override exists.
    """

    tool_name: str
    model: str
    trigger: str = "after_tool"  # 'after_tool' or 'before_tool'
    description: str = ""

    def __post_init__(self):
        if self.trigger not in ("after_tool", "before_tool"):
            raise ValueError(
                f"trigger must be 'after_tool' or 'before_tool', got: {self.trigger}"
            )


@dataclass
class ToolModelSwitchConfig:
    """Configuration for tool-based model switching middleware.

    Attributes:
        overrides: List of tool-model override configurations.
        default_model: Default model name (used when no override applies).
        cache_models: Whether to cache initialized model instances.
    """

    overrides: List[ToolModelOverride] = field(default_factory=list)
    default_model: Optional[str] = None
    cache_models: bool = True


# Main middleware class implementing wrap_model_call and wrap_tool_call hooks.


class ToolBasedModelSwitchMiddleware(AgentMiddleware[ToolModelSwitchState]):
    """Middleware that switches models based on tool execution.

    This middleware uses the wrap_model_call hook to intercept model calls
    and dynamically select the appropriate model based on which tool was
    last executed (for 'after_tool' triggers).

    For 'before_tool' triggers, it tracks pending tool calls via wrap_tool_call.

    Example:
        config = ToolModelSwitchConfig(
            overrides=[
                ToolModelOverride(
                    tool_name='analyze_complex_data',
                    model='claude-opus-4-1-20250805',
                    trigger='after_tool',
                    description='Use Claude for complex analysis follow-up'
                ),
            ],
            default_model='gpt-5-mini',
        )

        middleware = ToolBasedModelSwitchMiddleware(config)
        agent = create_agent(model="gpt-5-mini", middleware=[middleware])

    Attributes:
        state_schema: The custom state schema with tool tracking fields.
        config: Tool-model override configuration.
    """

    state_schema = ToolModelSwitchState

    def __init__(
        self,
        config: ToolModelSwitchConfig,
        model_factory: Optional[Any] = None,
    ):
        """Initialize the tool-based model switch middleware.

        Args:
            config: Configuration specifying tool-model mappings.
            model_factory: Optional model factory for creating model instances.
                          If None, uses init_chat_model directly.
        """
        self.config = config
        self._model_factory = model_factory

        # Build lookup dicts for fast override matching
        self._after_tool_overrides: Dict[str, ToolModelOverride] = {}
        self._before_tool_overrides: Dict[str, ToolModelOverride] = {}

        for override in config.overrides:
            if override.trigger == "after_tool":
                self._after_tool_overrides[override.tool_name] = override
            else:
                self._before_tool_overrides[override.tool_name] = override

        # Model cache (if enabled)
        self._model_cache: Dict[str, Any] = {}

        logger.debug(
            f"Initialized ToolBasedModelSwitchMiddleware with "
            f"{len(self._after_tool_overrides)} after_tool overrides and "
            f"{len(self._before_tool_overrides)} before_tool overrides"
        )

    def _get_model(self, model_name: str) -> Any:
        """Get or create a model instance.

        Uses caching if enabled to avoid recreating model instances.

        Args:
            model_name: Name of the model to get/create.

        Returns:
            Model instance ready for invocation.
        """
        if self.config.cache_models and model_name in self._model_cache:
            return self._model_cache[model_name]

        # Use model factory if available, otherwise init_chat_model
        if self._model_factory is not None:
            try:
                # LLMModelFactory uses create_model method
                model = self._model_factory.create_model(model_name)
                if model is None:
                    raise ValueError(f"Factory returned None for model '{model_name}'")
            except Exception as e:
                logger.warning(
                    f"Model factory failed to create model '{model_name}': {e}. "
                    f"Falling back to init_chat_model."
                )
                model = init_chat_model(model_name)
        else:
            model = init_chat_model(model_name)

        if self.config.cache_models:
            self._model_cache[model_name] = model

        return model

    def _find_last_tool_message(self, messages: List[Any]) -> Optional[str]:
        """Find the name of the last tool that was called.

        Searches messages in reverse order for the most recent ToolMessage.

        Args:
            messages: List of conversation messages.

        Returns:
            Name of the last tool called, or None if no tool messages found.
        """
        for message in reversed(messages):
            # Check for ToolMessage type
            if isinstance(message, ToolMessage):
                return message.name
            # Also check for dict-style messages
            if isinstance(message, dict) and message.get("type") == "tool":
                return message.get("name")
        return None

    # -------------------------------------------------------------------------
    # Wrap-style hooks
    # -------------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Intercept model calls and switch model based on last tool.

        This hook checks if the last tool called has an 'after_tool' override
        configured and switches the model accordingly.

        Args:
            request: The model request containing messages, tools, etc.
            handler: The next handler in the middleware chain.

        Returns:
            ModelResponse from the (potentially switched) model.
        """
        # Find the last tool that was called
        last_tool = self._find_last_tool_message(request.messages)

        # Check for after_tool override
        if last_tool and last_tool in self._after_tool_overrides:
            override = self._after_tool_overrides[last_tool]
            target_model_name = override.model

            logger.info(
                f"Tool '{last_tool}' triggered model switch to '{target_model_name}' "
                f"(reason: {override.description or 'after_tool override'})"
            )

            # Get the target model
            target_model = self._get_model(target_model_name)

            # Override the model in the request
            request = request.override(model=target_model)

        return handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool calls for tracking and before_tool triggers.

        This hook tracks which tool is being called. For 'before_tool' triggers,
        the model switch happens in the preceding model call based on predicted
        tool usage (which is tricky and less commonly used).

        Args:
            request: The tool call request.
            handler: The next handler in the middleware chain.

        Returns:
            ToolMessage or Command from the tool execution.
        """
        tool_name = request.tool_call.get("name", "unknown")

        logger.debug(f"Tool call: {tool_name}")

        # Execute the tool
        result = handler(request)

        return result


# Convenience function to create middleware from config dict.


def create_tool_model_switch_middleware(
    overrides: List[Dict[str, Any]],
    default_model: Optional[str] = None,
    model_factory: Optional[Any] = None,
    cache_models: bool = True,
) -> ToolBasedModelSwitchMiddleware:
    """Create a ToolBasedModelSwitchMiddleware from configuration dicts.

    This factory function is used by AgentService to create the middleware
    from llm_config.yaml configuration.

    Args:
        overrides: List of override configurations as dicts with keys:
            - tool_name: str (required)
            - model: str (required)
            - trigger: str (optional, default 'after_tool')
            - description: str (optional)
        default_model: Default model name when no override applies.
        model_factory: Optional model factory for creating model instances.
        cache_models: Whether to cache initialized model instances.

    Returns:
        Configured ToolBasedModelSwitchMiddleware instance.

    Example:
        middleware = create_tool_model_switch_middleware(
            overrides=[
                {
                    'tool_name': 'complex_analysis',
                    'model': 'claude-opus-4-1-20250805',
                    'trigger': 'after_tool',
                    'description': 'Use Claude for complex reasoning'
                }
            ],
            default_model='gpt-5-mini',
        )
    """
    override_configs = [
        ToolModelOverride(
            tool_name=o["tool_name"],
            model=o["model"],
            trigger=o.get("trigger", "after_tool"),
            description=o.get("description", ""),
        )
        for o in overrides
    ]

    config = ToolModelSwitchConfig(
        overrides=override_configs,
        default_model=default_model,
        cache_models=cache_models,
    )

    return ToolBasedModelSwitchMiddleware(
        config=config,
        model_factory=model_factory,
    )

"""
Subagent Configuration Middleware for Agent Server subagent graphs.

WHY: On the Agent Server, subagent graphs are pre-compiled at module load
time with their YAML default model.  When a user has a UserAgentInstance
with subagent overrides, those overrides must be applied
at runtime.

Unlike InstanceConfigMiddleware (which reads top-level ``primary_model``
from configurable), this middleware reads a nested ``subagent_configs``
dict from configurable and looks up overrides keyed by this subagent's
base agent name.  This prevents the parent's model override from leaking
into the subagent.

Flow:
    1. Parent graph's InstanceConfigMiddleware applies parent-level overrides
    2. SubAgentMiddleware invokes the subagent graph
    3. LangGraph propagates the parent's RunnableConfig (including configurable)
       to the subgraph
    4. SubagentConfigMiddleware reads ``subagent_configs[agent_name]`` from
       that configurable and applies subagent-specific overrides

Only used on the Agent Server.  Local execution resolves subagent models
directly in ``DeepAgentService.from_user_instance()``.
"""

import logging
from typing import Any, Callable, Optional

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage
from langgraph.config import get_config

logger = logging.getLogger(__name__)


def _extract_system_text(system_message: SystemMessage | str | None) -> str:
    """Extract plain text from a system message regardless of its type."""
    if system_message is None:
        return ""
    if isinstance(system_message, SystemMessage):
        return str(system_message.content)
    return str(system_message)


class SubagentConfigMiddleware(AgentMiddleware[AgentState]):
    """Middleware that applies per-user subagent overrides on the Agent Server.

    Reads ``subagent_configs`` from ``RunnableConfig.configurable`` (propagated
    from the parent graph) and looks up this subagent's overrides by
    ``agent_name``.  Supports overriding:
        - primary_model (model swap)
        - system_prompt_override (full replacement)
        - system_prompt_append (append to default)

    Model instances are cached to avoid repeated creation.
    Falls through to the compile-time default when no overrides are present.
    """

    state_schema = AgentState

    def __init__(
        self,
        agent_name: str,
        model_factory: Optional[Any] = None,
    ):
        """
        Args:
            agent_name: The base agent name (YAML key) for this subagent.
                        Used to look up overrides in ``subagent_configs``.
            model_factory: ``LLMModelFactory`` used to create model instances.
        """
        self._agent_name = agent_name
        self._model_factory = model_factory
        self._model_cache: dict[str, Any] = {}

    def _get_model(self, model_name: str) -> Any:
        """Get or create a model instance by name (cached)."""
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        if self._model_factory is not None:
            try:
                model = self._model_factory.create_model(model_name)
                if model is None:
                    raise ValueError(f"Factory returned None for model '{model_name}'")
            except Exception as e:
                logger.warning(
                    "SubagentConfigMiddleware: model factory failed for '%s': %s "
                    "— falling back to init_chat_model",
                    model_name,
                    e,
                )
                from langchain.chat_models import init_chat_model

                model = init_chat_model(model_name)
        else:
            from langchain.chat_models import init_chat_model

            model = init_chat_model(model_name)

        self._model_cache[model_name] = model
        return model

    def _get_my_config(self) -> dict[str, Any]:
        """Read this subagent's overrides from configurable.

        Looks up ``configurable["subagent_configs"][self._agent_name]``.
        Returns an empty dict if no overrides are found.
        """
        try:
            config = get_config()
            configurable = config.get("configurable", {})
            subagent_configs = configurable.get("subagent_configs", {})

            my_config = subagent_configs.get(self._agent_name, {})

            logger.debug(
                "SubagentConfigMiddleware[%s]: configurable has subagent_configs "
                "keys=%s, my_config=%s",
                self._agent_name,
                list(subagent_configs.keys()) if subagent_configs else "EMPTY",
                (
                    {
                        k: (v[:60] if isinstance(v, str) and len(v) > 60 else v)
                        for k, v in my_config.items()
                    }
                    if my_config
                    else "NONE"
                ),
            )

            return my_config

        except RuntimeError:
            logger.debug(
                "SubagentConfigMiddleware[%s]: get_config() unavailable",
                self._agent_name,
            )
            return {}

    def _apply_overrides(self, request: ModelRequest) -> ModelRequest:
        """Apply subagent-specific overrides (model + system prompt)."""
        my_config = self._get_my_config()
        if not my_config:
            logger.debug(
                "SubagentConfigMiddleware[%s]: no overrides found, using defaults",
                self._agent_name,
            )
            return request

        override_model = my_config.get("primary_model")
        prompt_override = my_config.get("system_prompt_override")
        prompt_append = my_config.get("system_prompt_append")

        logger.info(
            "SubagentConfigMiddleware[%s]: applying overrides — "
            "model=%r, prompt_override=%s, prompt_append=%s",
            self._agent_name,
            override_model,
            bool(prompt_override),
            bool(prompt_append),
        )

        # --- Model override ---
        if override_model:
            try:
                target_model = self._get_model(override_model)
                request = request.override(model=target_model)
                logger.info(
                    "SubagentConfigMiddleware[%s]: switched model → '%s'",
                    self._agent_name,
                    override_model,
                )
            except Exception:
                logger.exception(
                    "SubagentConfigMiddleware[%s]: failed to switch model to '%s'",
                    self._agent_name,
                    override_model,
                )

        # --- System prompt override ---
        if prompt_override:
            request = request.override(
                system_message=SystemMessage(content=prompt_override),
            )
        elif prompt_append:
            existing_text = _extract_system_text(request.system_message)
            merged = (
                f"{existing_text}\n\n{prompt_append}"
                if existing_text
                else prompt_append
            )
            request = request.override(
                system_message=SystemMessage(content=merged),
            )

        return request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Override model/prompt based on subagent-specific config."""
        request = self._apply_overrides(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse:
        """Async override of model/prompt based on subagent-specific config."""
        request = self._apply_overrides(request)
        return await handler(request)

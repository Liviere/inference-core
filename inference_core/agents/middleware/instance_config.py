"""
Instance Configuration Middleware for LangChain v1 Agents on Agent Server.

WHY: Agent Server graphs are pre-compiled at module load time with a fixed
model from YAML config.  When a user has a UserAgentInstance with overrides
(e.g. primary_model, system_prompt), those overrides must be applied at
runtime.  This middleware reads instance-level overrides from
``runtime.configurable`` and dynamically swaps the model and/or system
prompt for each invocation.

This runs ONLY on the Agent Server — local execution handles overrides
directly in ``AgentService.__init__`` / ``from_user_instance()``.

Key features:
    - Reads ``primary_model`` from configurable and swaps model in wrap_model_call
    - Reads ``system_prompt_override`` / ``system_prompt_append`` and modifies
      the system message in wrap_model_call
    - Model instances are cached per model name to avoid repeated creation
    - Falls through to the default model/prompt when no overrides are present

Source references:
    - LangChain v1 middleware: request.override(model=..., system_message=...)
    - Existing pattern: ToolBasedModelSwitchMiddleware, MemoryMiddleware
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
from langgraph.runtime import Runtime

from inference_core.agents.middleware._runtime_context import populate_from_configurable

logger = logging.getLogger(__name__)


def _extract_system_text(system_message: SystemMessage | str | None) -> str:
    """Extract plain text from a system message regardless of its type.

    WHY: ``request.system_message`` may be a ``SystemMessage`` object (from
    ``create_agent``) or a raw ``str`` (if another middleware already
    corrupted it).  We need plain text to merge with ``prompt_append``.
    """
    if system_message is None:
        return ""
    if isinstance(system_message, SystemMessage):
        return str(system_message.content)
    return str(system_message)


class InstanceConfigMiddleware(AgentMiddleware[AgentState]):
    """Middleware that applies per-user instance overrides on the Agent Server.

    Reads ``primary_model``, ``system_prompt_override``, and
    ``system_prompt_append`` from ``runtime.configurable`` (forwarded by
    ``agent_server_client._build_config()``) and overrides the pre-compiled
    graph's model and system prompt for the current invocation.

    Model instances are cached to avoid repeated creation across
    multiple model calls within a single agent run.
    """

    state_schema = AgentState

    def __init__(self, model_factory: Optional[Any] = None):
        """
        Args:
            model_factory: ``LLMModelFactory`` used to create model instances.
                           When None, falls back to ``init_chat_model``.
        """
        self._model_factory = model_factory
        self._model_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Model creation (cached)
    # ------------------------------------------------------------------

    def _get_model(self, model_name: str) -> Any:
        """Get or create a model instance by name.

        Uses the same factory that ``graph_builder`` used to create the
        default model, ensuring consistent provider resolution.
        """
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        if self._model_factory is not None:
            try:
                model = self._model_factory.create_model(model_name)
                if model is None:
                    raise ValueError(f"Factory returned None for model '{model_name}'")
            except Exception as e:
                logger.warning(
                    "Model factory failed for '%s': %s — falling back to "
                    "init_chat_model",
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

    def _get_configurable(self) -> dict[str, Any]:
        """Read configurable dict from LangGraph's RunnableConfig.

        WHY: Runtime does not carry configurable — it lives in RunnableConfig
        accessible via get_config().  This is the only reliable way to read
        per-run config (user_id, primary_model, etc.) across all middleware
        hooks, since node-style and wrap-style hooks may run in different
        async tasks (breaking contextvar propagation).
        """
        try:
            config = get_config()
            return config.get("configurable", {})
        except RuntimeError:
            return {}

    # ------------------------------------------------------------------
    # Node-style hooks
    # ------------------------------------------------------------------

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Populate context vars from RunnableConfig.configurable.

        WHY: CostTrackingMiddleware and other downstream middleware read
        per-request identifiers (user_id, session_id) from context vars.
        We populate them here as a best-effort — but InstanceConfigMiddleware
        itself reads configurable directly via get_config() in _apply_overrides
        to avoid contextvar isolation between async tasks.
        """
        if configurable := self._get_configurable():
            populate_from_configurable(configurable)
        return None

    # ------------------------------------------------------------------
    # Wrap-style hooks — model + prompt override
    # ------------------------------------------------------------------

    def _apply_overrides(self, request: ModelRequest) -> ModelRequest:
        """Apply instance overrides (model + system prompt) to a request.

        WHY: Reads configurable directly via get_config() instead of relying
        on context vars, because LangGraph runs node-style hooks (before_agent)
        and wrap-style hooks (awrap_model_call) in different async tasks —
        context vars set in one are invisible in the other.
        """
        configurable = self._get_configurable()
        override_model_name = configurable.get("primary_model")
        prompt_override = configurable.get("system_prompt_override")
        prompt_append = configurable.get("system_prompt_append")

        logger.debug(
            "InstanceConfigMiddleware._apply_overrides: configurable keys=%s, "
            "primary_model=%r, instance_name=%r, prompt_override=%r, prompt_append=%r, "
            "subagent_configs=%s",
            list(configurable.keys()),
            override_model_name,
            configurable.get("instance_name"),
            prompt_override[:80] if prompt_override else None,
            prompt_append[:80] if prompt_append else None,
            (
                list(configurable["subagent_configs"].keys())
                if configurable.get("subagent_configs")
                else "NONE"
            ),
        )

        # --- Model override ---
        if override_model_name:
            try:
                target_model = self._get_model(override_model_name)
                request = request.override(model=target_model)
                logger.debug(
                    "InstanceConfigMiddleware: switched model → '%s'",
                    override_model_name,
                )
            except Exception:
                logger.exception(
                    "InstanceConfigMiddleware: failed to switch model to '%s'",
                    override_model_name,
                )

        # --- System prompt override ---
        if prompt_override:
            request = request.override(
                system_message=SystemMessage(content=prompt_override),
            )
            logger.debug("InstanceConfigMiddleware: applied system_prompt_override")
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
            logger.debug("InstanceConfigMiddleware: appended to system_prompt")

        return request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Override model and/or system prompt based on instance config."""
        request = self._apply_overrides(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse:
        """Async override of model and/or system prompt."""
        request = self._apply_overrides(request)
        return await handler(request)

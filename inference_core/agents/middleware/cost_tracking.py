"""
Cost Tracking Middleware for LangChain v1 Agents.

This middleware provides usage and cost tracking for agent executions using
the new LangChain v1 middleware API. It hooks into agent lifecycle events
to capture token usage from model calls and persist cost data to the database.

Key features:
- Logs each model step individually (per-model granularity)
- Calculates costs using pricing configuration (input/output/extras)
- Supports context tier multipliers
- Persists to LLMRequestLog via existing UsageSession infrastructure

Usage:
    from langchain.agents import create_agent
    from inference_core.agents.middleware import CostTrackingMiddleware

    middleware = CostTrackingMiddleware(
        pricing_config=pricing_config,
        user_id=user_uuid,
    )

    agent = create_agent(
        model="gpt-4o",
        tools=[...],
        middleware=[middleware],
    )

Source references:
  - LangChain v1 middleware docs: https://docs.langchain.com/oss/python/langchain/middleware/custom
  - Custom state schema pattern from LangChain middleware documentation
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.errors import GraphInterrupt
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import NotRequired

from inference_core.agents.middleware._runtime_context import (
    get_instance_id,
    get_instance_name,
    get_request_id,
    get_session_id,
    get_user_id,
    populate_from_configurable,
)
from inference_core.llm.config import PricingConfig, UsageLoggingConfig, get_llm_config
from inference_core.llm.usage_logging import PricingCalculator, UsageSession
from inference_core.services._cancel import AgentCancelled

logger = logging.getLogger(__name__)


class CostTrackingState(AgentState):
    """Extended agent state for cost tracking middleware.

    This state schema extends the base AgentState with fields for tracking
    accumulated usage and cost data across agent execution.

    Attributes:
        usage_session_id: Unique identifier for the usage tracking session
        accumulated_input_tokens: Running total of input tokens across model calls
        accumulated_output_tokens: Running total of output tokens across model calls
        accumulated_total_tokens: Running total of all tokens
        accumulated_extra_tokens: Additional token types (reasoning, cache, etc.)
        tool_call_count: Number of tool calls made during the run
        tool_call_latencies: List of tool call durations in milliseconds
        model_call_count: Number of model calls made during the run
        model_call_latencies: List of model call durations in milliseconds
        last_request_log_id: UUID of the most recently created LLMRequestLog
        last_request_cost_usd: Cost (USD) of the most recent model call
        last_request_model_name: Model name for the most recent call
    """

    usage_session_id: NotRequired[str]
    accumulated_input_tokens: NotRequired[int]
    accumulated_output_tokens: NotRequired[int]
    accumulated_total_tokens: NotRequired[int]
    accumulated_extra_tokens: NotRequired[Dict[str, int]]
    tool_call_count: NotRequired[int]
    tool_call_latencies: NotRequired[List[float]]
    model_call_count: NotRequired[int]
    model_call_latencies: NotRequired[List[float]]
    last_request_log_id: NotRequired[str]
    last_request_cost_usd: NotRequired[float]
    last_request_model_name: NotRequired[str]


@dataclass
class _MiddlewareContext:
    """Internal context for tracking data between middleware hooks.

    The ``persisted_*`` fields are populated by ``_persist_from_response()``
    (called inside ``wrap_model_call``) so that ``after_model`` can update
    the graph state without re-persisting to the database.  This is critical
    because ``after_model`` runs as a **separate LangGraph node** and may be
    skipped when the agent is cancelled between the model node and the
    after_model node.
    """

    model_call_start_time: Optional[float] = None

    # Populated by _persist_from_response() inside wrap_model_call
    persisted_log_id: Optional[str] = None
    persisted_cost_usd: Optional[float] = None
    persisted_model_name: Optional[str] = None
    persisted_fragment: Optional[Dict[str, Any]] = None
    persisted_extra_tokens: Optional[Dict[str, int]] = None
    persisted_is_estimated: bool = False


class CostTrackingMiddleware(AgentMiddleware[CostTrackingState]):
    """Middleware for tracking LLM usage and costs during agent execution.

    This middleware captures token usage from model calls and calculates costs
    using the configured pricing. It persists usage logs to the database via
    the existing UsageSession infrastructure.

    The middleware uses the following hooks:
    - before_agent: Reset per-run context and initialize counters
    - wrap_model_call: Capture start time for each model step
    - wrap_tool_call: Lightweight tool boundary logging
    - after_agent: No-op because per-step logging happens in after_model

    Attributes:
        state_schema: The custom state schema with usage tracking fields
        pricing_config: Pricing configuration for cost calculations
        user_id: Optional user ID for attribution
        session_id: Optional session ID for grouping requests
        request_id: Optional correlation ID (e.g., Celery task ID)
        logging_config: Usage logging configuration
        task_type: Task type for logging (default: "agent")
        request_mode: Request mode for logging (default: "sync")
    """

    state_schema = CostTrackingState

    def __init__(
        self,
        pricing_config: Optional[PricingConfig] = None,
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        logging_config: Optional[UsageLoggingConfig] = None,
        task_type: str = "agent",
        request_mode: str = "sync",
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        instance_id: Optional[uuid.UUID] = None,
        instance_name: Optional[str] = None,
    ):
        """Initialize the cost tracking middleware.

        Args:
            pricing_config: Pricing configuration for cost calculations.
                           If None, attempts to load from llm_config.
            user_id: Optional user ID for attribution in logs.
            session_id: Optional session ID for grouping related requests.
            request_id: Optional correlation ID (e.g., Celery task ID).
            logging_config: Usage logging configuration. If None, uses defaults.
            task_type: Task type for logging (default: "agent").
            request_mode: Request mode for logging (default: "sync").
            provider: LLM provider name (e.g., "openai"). Auto-detected if None.
            model_name: Model name. Auto-detected from runtime if None.
            instance_id: Optional UserAgentInstance UUID for per-instance attribution.
            instance_name: Optional UserAgentInstance name for per-instance attribution.
        """
        self.pricing_config = pricing_config
        self.user_id = user_id
        self.session_id = session_id
        self.request_id = request_id
        self.logging_config = logging_config or UsageLoggingConfig()
        self.task_type = task_type
        self.request_mode = request_mode
        self._provider = provider
        self._model_name = model_name
        self.instance_id = instance_id
        self.instance_name = instance_name
        self.cancel_check: Callable[[], bool] | None = None

        # Per-invocation context (reset on each before_agent)
        self._ctx: Optional[_MiddlewareContext] = None

    def _init_context(self):
        """Initialize the internal middleware context."""
        self._ctx = _MiddlewareContext()

        return {
            "usage_session_id": str(uuid.uuid4()),
            "accumulated_input_tokens": 0,
            "accumulated_output_tokens": 0,
            "accumulated_total_tokens": 0,
            "accumulated_extra_tokens": {},
            "tool_call_count": 0,
            "tool_call_latencies": [],
            "model_call_count": 0,
            "model_call_latencies": [],
        }

    # -------------------------------------------------------------------------
    # Node-style hooks
    # -------------------------------------------------------------------------

    def before_agent(
        self, state: CostTrackingState, runtime: Runtime
    ) -> Dict[str, Any] | None:
        """Initialize per-run context and state counters.

        This hook prepares in-memory context and state accumulators so
        downstream hooks can persist per-step usage cleanly.

        Args:
            state: Current agent state
            runtime: Agent runtime context

        Returns:
            State updates with initialized tracking fields
        """
        # Create fresh context for this invocation
        return self._init_context()

    def before_agent(
        self, state: CostTrackingState, runtime: Runtime
    ) -> Dict[str, Any] | None:
        """Re-Initialize context and state counters.

        Also populates per-request context vars from RunnableConfig.configurable
        so that wrap-style hooks (which don't receive ``runtime``) can
        resolve user_id / session_id in a concurrency-safe way.
        """
        # Populate task-local context vars from Agent Server configurable
        try:
            from langgraph.config import get_config

            config = get_config()
            configurable = config.get("configurable")
        except RuntimeError:
            configurable = None

        if isinstance(configurable, dict) and configurable:
            populate_from_configurable(configurable)

        if not self._ctx:
            return self._init_context()

    def after_model(
        self, state: CostTrackingState, runtime: Runtime
    ) -> Dict[str, Any] | None:
        """Propagate usage data from the model response into graph state.

        The actual DB persistence (``LLMRequestLog``) now happens inside
        ``wrap_model_call`` via ``_persist_from_response()``, which runs in
        the **same LangGraph node** as the model call itself.  This
        guarantees that the log entry is written even when the agent is
        cancelled between the model node and this ``after_model`` node.

        This hook reads the pre-persisted data from ``_ctx`` and only
        accumulates token counters + exposes ``last_request_log_id`` etc.
        to the shared state (consumed by downstream middleware such as
        ``CreditBillingMiddleware``).

        If ``_ctx`` has no persisted data (edge-case safety net), the hook
        falls back to the legacy path and persists from ``state.messages``.

        Args:
            state: Current agent state with messages
            runtime: Agent runtime context

        Returns:
            State updates with accumulated token counts
        """
        if not self._ctx:
            self._ctx = _MiddlewareContext()

        updates: Dict[str, Any] = {}

        # Increment model call count
        current_count = state.get("model_call_count", 0)
        updates["model_call_count"] = current_count + 1

        try:
            # ----- Fast path: data already persisted by wrap_model_call -----
            if self._ctx.persisted_log_id is not None:
                fragment = self._ctx.persisted_fragment or {}
                extra_tokens = self._ctx.persisted_extra_tokens or {}

                if fragment:
                    updates["accumulated_input_tokens"] = state.get(
                        "accumulated_input_tokens", 0
                    ) + fragment.get("input_tokens", 0)
                    updates["accumulated_output_tokens"] = state.get(
                        "accumulated_output_tokens", 0
                    ) + fragment.get("output_tokens", 0)
                    updates["accumulated_total_tokens"] = state.get(
                        "accumulated_total_tokens", 0
                    ) + fragment.get("total_tokens", 0)
                    updates["accumulated_extra_tokens"] = self._merge_extra_tokens(
                        state.get("accumulated_extra_tokens", {}), extra_tokens
                    )

                updates["last_request_log_id"] = self._ctx.persisted_log_id
                if self._ctx.persisted_cost_usd is not None:
                    updates["last_request_cost_usd"] = self._ctx.persisted_cost_usd
                if self._ctx.persisted_model_name:
                    updates["last_request_model_name"] = self._ctx.persisted_model_name

                return updates if updates else None

            # ----- Fallback: persist from state (legacy / safety net) -----
            messages = state.get("messages", [])
            if not messages:
                return updates

            last_message = messages[-1]

            # Extract usage_metadata from AIMessage
            # LangChain v1 AIMessage has usage_metadata attribute
            usage_metadata = getattr(last_message, "usage_metadata", None)

            response_metadata = getattr(last_message, "response_metadata", None) or {}
            model_name = response_metadata.get(
                "model_name", self._model_name or "unknown"
            )

            # Determine usage fragment: prefer real metadata, fall back to estimation
            if usage_metadata and isinstance(usage_metadata, dict):
                fragment, extra_tokens = self._extract_usage_fragment(usage_metadata)
                is_estimated = False
            else:
                # Fallback: estimate tokens from message content when the provider
                # doesn't return usage in streaming responses (e.g. DeepInfra Nemotron)
                fragment = self._estimate_token_usage(messages)
                extra_tokens = {}
                is_estimated = True
                if fragment:
                    logger.info(
                        "usage_metadata unavailable for model=%s; "
                        "falling back to estimated tokens: %s",
                        model_name,
                        fragment,
                    )

            if fragment:
                # Keep simple accumulators in state for compatibility
                updates["accumulated_input_tokens"] = state.get(
                    "accumulated_input_tokens", 0
                ) + fragment.get("input_tokens", 0)
                updates["accumulated_output_tokens"] = state.get(
                    "accumulated_output_tokens", 0
                ) + fragment.get("output_tokens", 0)
                updates["accumulated_total_tokens"] = state.get(
                    "accumulated_total_tokens", 0
                ) + fragment.get("total_tokens", 0)
                updates["accumulated_extra_tokens"] = self._merge_extra_tokens(
                    state.get("accumulated_extra_tokens", {}), extra_tokens
                )

                provider = self._provider or self._detect_provider(model_name)
                pricing_config = self._get_pricing_config(model_name)

                session = UsageSession(
                    task_type=self.task_type,
                    request_mode=self.request_mode,
                    model_name=model_name,
                    provider=provider,
                    pricing_config=pricing_config,
                    user_id=self.user_id or get_user_id(),
                    session_id=self.session_id or get_session_id(),
                    request_id=self.request_id or get_request_id(),
                    logging_config=self.logging_config,
                    instance_id=self.instance_id or get_instance_id(),
                    instance_name=self.instance_name or get_instance_name(),
                )

                # Align latency window with actual model call if recorded
                if self._ctx.model_call_start_time:
                    session.start_time = self._ctx.model_call_start_time

                log_id = session.finalize_sync(
                    success=True,
                    error=None,
                    final_usage=fragment,
                    streamed=False,
                    partial=False,
                    details={"estimated": True} if is_estimated else None,
                )

                if log_id and pricing_config:
                    cost_result = PricingCalculator.compute_cost(
                        fragment, pricing_config
                    )
                    updates["last_request_log_id"] = str(log_id)
                    updates["last_request_cost_usd"] = cost_result.get(
                        "cost_total_usd", 0.0
                    )
                    updates["last_request_model_name"] = model_name
        except Exception as e:
            logger.warning(
                f"Error extracting usage from model response: {e}", exc_info=True
            )
        finally:
            if self._ctx:
                self._ctx.model_call_start_time = None
                # Reset persisted data so next model call starts clean
                self._ctx.persisted_log_id = None
                self._ctx.persisted_cost_usd = None
                self._ctx.persisted_model_name = None
                self._ctx.persisted_fragment = None
                self._ctx.persisted_extra_tokens = None
                self._ctx.persisted_is_estimated = False

        return updates if updates else None

    def after_agent(
        self, state: CostTrackingState, runtime: Runtime
    ) -> Dict[str, Any] | None:
        """Finalize and persist usage log after agent completion.

        Per-step logging is already handled in after_model, so this hook
        currently performs no additional persistence.

        Args:
            state: Final agent state
            runtime: Agent runtime context

        Returns:
            None (no state updates needed)
        """
        # Per-step logging already persisted in after_model; nothing to finalize.
        return None

    # -------------------------------------------------------------------------
    # Wrap-style hooks
    # -------------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Wrap model calls to measure latency.

        This hook captures the call start time so after_model can compute
        per-step latency when persisting usage.  Also checks the optional
        ``cancel_check`` before delegating to the LLM.

        Args:
            request: The model request
            handler: The handler to call the model

        Returns:
            The model response

        Raises:
            AgentCancelled: If the cancel_check callback returns ``True``.
        """
        if self.cancel_check:
            try:
                if self.cancel_check():
                    raise AgentCancelled("Agent execution cancelled by user")
            except AgentCancelled:
                raise
            except Exception:
                pass

        if self._ctx:
            self._ctx.model_call_start_time = time.monotonic()

        response = handler(request)

        # Persist LLMRequestLog immediately — inside the same LangGraph node
        # as the model call.  This guarantees the log entry exists even when
        # after_model is skipped due to AgentCancelled between graph nodes.
        self._persist_from_response(response, request)

        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse:
        """Async version of wrap_model_call for use with astream."""
        if self.cancel_check:
            try:
                if self.cancel_check():
                    raise AgentCancelled("Agent execution cancelled by user")
            except AgentCancelled:
                raise
            except Exception:
                pass

        if self._ctx:
            self._ctx.model_call_start_time = time.monotonic()

        response = await handler(request)

        # Use the async persist path to avoid spawning a new event loop via
        # finalize_sync / run_async_safely (which causes MissingGreenlet when
        # the Agent Server runs this inside a ThreadPoolExecutor).
        await self._apersist_from_response(response, request)

        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Wrap tool calls for lightweight logging.

        Tool invocations are not persisted to usage logs here, but we keep
        debug visibility around tool execution boundaries.  Also checks
        the optional ``cancel_check`` before executing the tool.

        Args:
            request: The tool call request
            handler: The handler to execute the tool

        Returns:
            The tool response (ToolMessage or Command)

        Raises:
            AgentCancelled: If the cancel_check callback returns ``True``.
        """
        if self.cancel_check:
            try:
                if self.cancel_check():
                    raise AgentCancelled("Agent execution cancelled by user")
            except AgentCancelled:
                raise
            except Exception:
                pass

        tool_name = (
            request.tool_call.get("name", "<unknown>")
            if hasattr(request, "tool_call")
            else "<unknown>"
        )

        logger.debug(f"Tool call start: {tool_name}")

        try:
            result = handler(request)
            logger.debug(f"Tool call end: {tool_name}")
            return result

        except GraphInterrupt as gi:
            logger.debug(f"Tool call interrupted by GraphInterrupt ({tool_name}): {gi}")
            raise

        except Exception as e:
            logger.warning(f"Tool call error ({tool_name}): {e}")
            tool_call_id = (
                request.tool_call.get("id", "") if hasattr(request, "tool_call") else ""
            )
            return ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_call_id)
        # Tool timing is not persisted; no context tracking needed here.

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> ToolMessage | Command:
        """Async version of wrap_tool_call for use with astream."""
        tool_name = (
            request.tool_call.get("name", "<unknown>")
            if hasattr(request, "tool_call")
            else "<unknown>"
        )
        logger.debug(f"Tool call start (async): {tool_name}")
        try:
            result = await handler(request)
            logger.debug(f"Tool call end (async): {tool_name}")
            return result

        except GraphInterrupt as gi:
            logger.debug(f"Tool call interrupted by GraphInterrupt ({tool_name}): {gi}")
            raise

        except Exception as e:
            logger.warning(f"Tool call error (async, {tool_name}): {e}")
            tool_call_id = (
                request.tool_call.get("id", "") if hasattr(request, "tool_call") else ""
            )
            return ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_call_id)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _build_usage_session(
        self, response: ModelResponse, request: ModelRequest
    ) -> tuple | None:
        """Prepare a UsageSession and extract usage data from a model response.

        Shared preparation logic for both sync (_persist_from_response) and
        async (_apersist_from_response) persist paths, keeping both callers DRY.

        Returns:
            (session, fragment, extra_tokens, is_estimated, model_name,
             pricing_config) or None if there is nothing to persist.
        """
        if not response.result:
            return None

        ai_message = response.result[-1]
        usage_metadata = getattr(ai_message, "usage_metadata", None)
        response_metadata = getattr(ai_message, "response_metadata", None) or {}
        model_name = response_metadata.get("model_name", self._model_name or "unknown")

        if usage_metadata and isinstance(usage_metadata, dict):
            fragment, extra_tokens = self._extract_usage_fragment(usage_metadata)
            is_estimated = False
        else:
            # Build full message list for estimation (request input + response output)
            all_messages: list = []
            if request.system_message:
                all_messages.append(request.system_message)
            all_messages.extend(request.messages)
            all_messages.extend(response.result)
            fragment = self._estimate_token_usage(all_messages)
            extra_tokens = {}
            is_estimated = True
            if fragment:
                logger.info(
                    "usage_metadata unavailable for model=%s; "
                    "falling back to estimated tokens (wrap_model_call): %s",
                    model_name,
                    fragment,
                )

        if not fragment:
            return None

        provider = self._provider or self._detect_provider(model_name)
        pricing_config = self._get_pricing_config(model_name)

        # Resolve per-request IDs: prefer instance attrs (local),
        # fall back to task-local context vars (Agent Server).
        effective_user_id = self.user_id or get_user_id()
        effective_session_id = self.session_id or get_session_id()
        effective_request_id = self.request_id or get_request_id()
        effective_instance_id = self.instance_id or get_instance_id()
        effective_instance_name = self.instance_name or get_instance_name()

        session = UsageSession(
            task_type=self.task_type,
            request_mode=self.request_mode,
            model_name=model_name,
            provider=provider,
            pricing_config=pricing_config,
            user_id=effective_user_id,
            session_id=effective_session_id,
            request_id=effective_request_id,
            logging_config=self.logging_config,
            instance_id=effective_instance_id,
            instance_name=effective_instance_name,
        )

        if self._ctx and self._ctx.model_call_start_time:
            session.start_time = self._ctx.model_call_start_time

        return session, fragment, extra_tokens, is_estimated, model_name, pricing_config

    def _apply_persist_result(
        self,
        log_id,
        fragment: Dict[str, Any],
        extra_tokens: Dict[str, int],
        is_estimated: bool,
        model_name: str,
        pricing_config,
    ) -> None:
        """Store persist results in middleware context after finalize.

        Shared epilogue for both sync and async persist paths.
        """
        if log_id and self._ctx:
            self._ctx.persisted_log_id = str(log_id)
            self._ctx.persisted_fragment = fragment
            self._ctx.persisted_extra_tokens = extra_tokens
            self._ctx.persisted_is_estimated = is_estimated
            self._ctx.persisted_model_name = model_name

            if pricing_config:
                cost_result = PricingCalculator.compute_cost(fragment, pricing_config)
                self._ctx.persisted_cost_usd = cost_result.get("cost_total_usd", 0.0)

    def _persist_from_response(
        self, response: ModelResponse, request: ModelRequest
    ) -> None:
        """Extract usage from a ModelResponse and persist an LLMRequestLog.

        Called from ``wrap_model_call`` (sync path) so the DB write happens
        inside the same LangGraph node as the model call. Results are stored
        in ``self._ctx.persisted_*`` for ``after_model`` to propagate into
        graph state without a second DB write.

        For the async path (``awrap_model_call``) use
        ``_apersist_from_response`` instead to avoid spawning a new event
        loop via ``finalize_sync`` / ``run_async_safely``.

        This method is intentionally non-fatal: any exception is logged
        and swallowed so the agent flow is never interrupted by billing.
        """
        if not self._ctx:
            return

        try:
            result = self._build_usage_session(response, request)
            if result is None:
                return

            (
                session,
                fragment,
                extra_tokens,
                is_estimated,
                model_name,
                pricing_config,
            ) = result

            log_id = session.finalize_sync(
                success=True,
                error=None,
                final_usage=fragment,
                streamed=False,
                partial=False,
                details={"estimated": True} if is_estimated else None,
            )

            self._apply_persist_result(
                log_id, fragment, extra_tokens, is_estimated, model_name, pricing_config
            )
        except Exception as e:
            logger.warning(
                "Failed to persist usage from wrap_model_call: %s",
                e,
                exc_info=True,
            )

    async def _apersist_from_response(
        self, response: ModelResponse, request: ModelRequest
    ) -> None:
        """Async variant of _persist_from_response for use in awrap_model_call.

        Directly awaits ``session.finalize()`` on the current event loop
        instead of calling ``finalize_sync()`` (which spawns a new thread +
        event loop via ``run_async_safely``).  This prevents MissingGreenlet
        errors that occur when the Agent Server runs middleware code in a
        ThreadPoolExecutor where the event loop differs from the one the DB
        engine was originally created on.

        This method is intentionally non-fatal: any exception is logged
        and swallowed so the agent flow is never interrupted by billing.
        """
        if not self._ctx:
            return

        try:
            result = self._build_usage_session(response, request)
            if result is None:
                return

            (
                session,
                fragment,
                extra_tokens,
                is_estimated,
                model_name,
                pricing_config,
            ) = result

            log_id = await session.finalize(
                success=True,
                error=None,
                final_usage=fragment,
                streamed=False,
                partial=False,
                details={"estimated": True} if is_estimated else None,
            )

            self._apply_persist_result(
                log_id, fragment, extra_tokens, is_estimated, model_name, pricing_config
            )
        except Exception as e:
            logger.warning(
                "Failed to persist usage from awrap_model_call: %s",
                e,
                exc_info=True,
            )

    @staticmethod
    def _detect_provider(model_name: str) -> str:
        """Detect provider from model name.

        Args:
            model_name: The model name

        Returns:
            Detected provider string
        """
        model_lower = model_name.lower()

        if any(x in model_lower for x in ["gpt", "o1", "o3", "davinci", "turbo"]):
            return "openai"
        elif any(x in model_lower for x in ["claude", "anthropic"]):
            return "anthropic"
        elif any(x in model_lower for x in ["gemini", "palm", "bard"]):
            return "google"
        elif any(x in model_lower for x in ["llama", "mistral", "mixtral"]):
            return "meta"
        elif "deepseek" in model_lower:
            return "deepseek"
        else:
            return "unknown"

    def _get_pricing_config(self, model_name: str) -> Optional[PricingConfig]:
        """Fetch pricing config for a given model.

        Prefers llm_config per-model pricing; falls back to middleware default.
        """

        try:
            llm_config = get_llm_config()
            model_cfg = llm_config.models.get(model_name)
            if model_cfg and model_cfg.pricing:
                return model_cfg.pricing
        except Exception as e:
            logger.debug(f"Could not load pricing config for {model_name}: {e}")

        if self.pricing_config:
            return self.pricing_config

        return None

    @staticmethod
    def _extract_usage_fragment(
        usage_metadata: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, int]]:
        """Extract a per-step usage fragment from LangChain metadata."""

        input_tokens = usage_metadata.get("input_tokens", 0) or 0
        output_tokens = usage_metadata.get("output_tokens", 0) or 0
        total_tokens = usage_metadata.get("total_tokens", input_tokens + output_tokens)

        fragment: Dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

        extra_tokens: Dict[str, int] = {}

        input_details = usage_metadata.get("input_token_details", {}) or {}
        output_details = usage_metadata.get("output_token_details", {}) or {}

        for detail_key, val in input_details.items():
            if isinstance(val, (int, float)) and val > 0:
                token_key = f"{detail_key}_tokens"
                fragment[token_key] = val
                extra_tokens[token_key] = extra_tokens.get(token_key, 0) + int(val)

        for detail_key, val in output_details.items():
            if isinstance(val, (int, float)) and val > 0:
                token_key = f"{detail_key}_tokens"
                fragment[token_key] = val
                extra_tokens[token_key] = extra_tokens.get(token_key, 0) + int(val)

        for key, value in usage_metadata.items():
            if key.endswith("_tokens") and key not in {
                "input_tokens",
                "output_tokens",
                "total_tokens",
            }:
                if isinstance(value, (int, float)) and value > 0:
                    fragment[key] = value
                    extra_tokens[key] = extra_tokens.get(key, 0) + int(value)

        return fragment, extra_tokens

    @staticmethod
    def _estimate_token_usage(
        messages: List[Any],
    ) -> Optional[Dict[str, int]]:
        """Estimate token usage from message content when usage_metadata is unavailable.

        Some providers (e.g. DeepInfra for certain models) don't return usage
        data in streaming responses despite ``stream_options``. This method
        provides a reasonable fallback using tiktoken (cl100k_base) or, if
        tiktoken is unavailable, a character-based heuristic (~4 chars/token).

        Args:
            messages: Full state message list; the last entry is the AI response,
                      everything before it is counted as input.

        Returns:
            Usage fragment dict or None if estimation yields zero tokens.
        """
        if not messages:
            return None

        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = None

        def _count(text: str) -> int:
            if enc:
                return len(enc.encode(text))
            # Rough heuristic: ~4 characters per token
            return max(1, len(text) // 4)

        input_tokens = 0
        for msg in messages[:-1]:
            content = getattr(msg, "content", "") or ""
            if isinstance(content, list):
                # Multi-modal messages may have list content
                content = " ".join(
                    str(c.get("text", "")) if isinstance(c, dict) else str(c)
                    for c in content
                )
            input_tokens += _count(str(content))
            # Account for message framing overhead (~4 tokens per message)
            input_tokens += 4

        last_message = messages[-1]
        output_content = getattr(last_message, "content", "") or ""
        if isinstance(output_content, list):
            output_content = " ".join(
                str(c.get("text", "")) if isinstance(c, dict) else str(c)
                for c in output_content
            )
        output_tokens = _count(str(output_content))

        if input_tokens == 0 and output_tokens == 0:
            return None

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    @staticmethod
    def _merge_extra_tokens(
        accumulated: Dict[str, int], current: Dict[str, int]
    ) -> Dict[str, int]:
        """Merge accumulated and current extra token counters."""

        merged = dict(accumulated)
        for key, value in current.items():
            merged[key] = merged.get(key, 0) + value
        return merged


# Convenience function for creating middleware with common defaults
def create_cost_tracking_middleware(
    user_id: Optional[uuid.UUID] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    task_type: str = "agent",
    **kwargs,
) -> CostTrackingMiddleware:
    """Factory function for creating CostTrackingMiddleware with sensible defaults.

    This function loads pricing configuration from llm_config.yaml if available
    and creates a middleware instance with the provided options.

    Args:
        user_id: Optional user ID for attribution
        session_id: Optional session ID for grouping requests
        request_id: Optional correlation ID
        task_type: Task type for logging (default: "agent")
        **kwargs: Additional arguments passed to CostTrackingMiddleware

    Returns:
        Configured CostTrackingMiddleware instance
    """
    # Try to load logging config from global config
    logging_config = None
    try:
        llm_config = get_llm_config()
        logging_config = llm_config.usage_logging
    except Exception:
        pass

    return CostTrackingMiddleware(
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        logging_config=logging_config,
        task_type=task_type,
        **kwargs,
    )


__all__ = [
    "CostTrackingMiddleware",
    "CostTrackingState",
    "create_cost_tracking_middleware",
]

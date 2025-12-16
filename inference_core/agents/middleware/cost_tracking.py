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

from __future__ import annotations

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
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import NotRequired

from inference_core.llm.config import PricingConfig, UsageLoggingConfig, get_llm_config
from inference_core.llm.usage_logging import UsageSession

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


@dataclass
class _MiddlewareContext:
    """Internal context for tracking data between middleware hooks."""

    model_call_start_time: Optional[float] = None


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

        # Per-invocation context (reset on each before_agent)
        self._ctx: Optional[_MiddlewareContext] = None

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
        self._ctx = _MiddlewareContext()

        # Return initial state updates
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

    def after_model(
        self, state: CostTrackingState, runtime: Runtime
    ) -> Dict[str, Any] | None:
        """Capture usage metadata from the model response.

        This hook extracts token usage from the last message in state
        (which should be the AIMessage from the model) and immediately
        persists a per-step UsageSession so each model call is logged
        separately (important when models differ per step).

        Args:
            state: Current agent state with messages
            runtime: Agent runtime context

        Returns:
            State updates with accumulated token counts
        """
        if not self._ctx:
            return None

        updates: Dict[str, Any] = {}

        # Increment model call count
        current_count = state.get("model_call_count", 0)
        updates["model_call_count"] = current_count + 1

        try:
            # Get the last message (should be AIMessage from model)
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

            if usage_metadata and isinstance(usage_metadata, dict):
                # Extract usage for this step only
                fragment, extra_tokens = self._extract_usage_fragment(usage_metadata)

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

                # Persist this single step immediately (per-model granularity)
                provider = self._provider or self._detect_provider(model_name)
                pricing_config = self._get_pricing_config(model_name)

                session = UsageSession(
                    task_type=self.task_type,
                    request_mode=self.request_mode,
                    model_name=model_name,
                    provider=provider,
                    pricing_config=pricing_config,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    request_id=self.request_id,
                    logging_config=self.logging_config,
                )

                # Align latency window with actual model call if recorded
                if self._ctx.model_call_start_time:
                    session.start_time = self._ctx.model_call_start_time

                session.finalize_sync(
                    success=True,
                    error=None,
                    final_usage=fragment,
                    streamed=False,
                    partial=False,
                    details=None,
                )
        except Exception as e:
            logger.warning(f"Error extracting usage from model response: {e}")
        finally:
            if self._ctx:
                self._ctx.model_call_start_time = None

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
        per-step latency when persisting usage.

        Args:
            request: The model request
            handler: The handler to call the model

        Returns:
            The model response
        """
        if self._ctx:
            self._ctx.model_call_start_time = time.monotonic()

        response = handler(request)
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Wrap tool calls for lightweight logging.

        Tool invocations are not persisted to usage logs here, but we keep
        debug visibility around tool execution boundaries.

        Args:
            request: The tool call request
            handler: The handler to execute the tool

        Returns:
            The tool response (ToolMessage or Command)
        """
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
        except Exception as e:
            logger.warning(f"Tool call error ({tool_name}): {e}")
            return {"error": str(e)}
        # Tool timing is not persisted; no context tracking needed here.

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

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

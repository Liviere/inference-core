"""Custom LangChain callback handlers for usage & cost logging.

The goal is to capture provider-specific usage metadata emitted by LangChain
LLM/ChatModel wrappers (OpenAI, Anthropic, Google, etc.) and accumulate it in
an existing `UsageSession` without needing to manually pass usage metadata
through higher-level chain abstractions.

We hook into `on_llm_end` which receives an `LLMResult` containing
`llm_output`. For most modern LangChain providers this includes one of:

    llm_output = {
        "token_usage": { ... },        # OpenAI, Anthropic, etc.
        "model_name": "gpt-4.1-mini",
        ...
    }

Some providers may use alternative keys (e.g. `usage`, `usage_metadata`,
`response_metadata`). We attempt to extract the first present mapping of token
counts. All numeric values are accumulated into the attached UsageSession.

This handler is intentionally lightweight: it never commits to the database;
that is handled by `UsageSession.finalize()` invoked by the service layer
after chain completion (success or error). This preserves existing error
handling semantics (`fail_open`).

Source references:
  - LangChain Runnables callbacks config propagation (2025-09 snapshot)
  - OpenAI / Anthropic chat model `llm_output.token_usage` pattern
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler

from .config import PricingConfig
from .usage_logging import UsageNormalizer, UsageSession

logger = logging.getLogger(__name__)


class LLMUsageCallbackHandler(UsageMetadataCallbackHandler):
    """Hybrid usage callback accumulating provider token usage into UsageSession.

    Combines two mechanisms:
      1. Built-in `UsageMetadataCallbackHandler` (AIMessage.usage_metadata)
      2. Fallback to `llm_output.token_usage` / `usage` / `usage_metadata` / `response_metadata`

    To avoid double-counting when the same model is invoked multiple times
    within a single session, we only accumulate deltas (differences) relative to
    the last recorded values per model.
    """

    def __init__(
        self,
        usage_session: UsageSession,
        pricing_config: Optional[PricingConfig] = None,
    ):
        super().__init__()
        self.usage_session = usage_session
        self.pricing_config = pricing_config
        # We store the last known cumulative usage from usage_metadata (per model)
        self._last_model_totals: Dict[str, Dict[str, int]] = {}

    # --- LLM lifecycle hooks -------------------------------------------------
    def on_llm_end(self, response, **kwargs):  # type: ignore[override]
        # First allow the base class to update self.usage_metadata
        try:
            super().on_llm_end(response, **kwargs)
        except Exception as e:  # pragma: no cover
            logger.debug(f"Base usage metadata handler error: {e}")

        try:
            # 1. Delta from usage_metadata (if available)
            for model_name, meta in self.usage_metadata.items():
                input_tokens = meta.get("input_tokens", 0) or 0
                output_tokens = meta.get("output_tokens", 0) or 0
                total_tokens = meta.get("total_tokens", input_tokens + output_tokens)

                prev = self._last_model_totals.get(model_name, {})
                delta_input = max(0, input_tokens - prev.get("input_tokens", 0))
                delta_output = max(0, output_tokens - prev.get("output_tokens", 0))
                delta_total = max(0, total_tokens - prev.get("total_tokens", 0))

                if delta_input or delta_output or delta_total:
                    fragment = {}
                    if delta_input:
                        fragment["input_tokens"] = delta_input
                    if delta_output:
                        fragment["output_tokens"] = delta_output
                    # total_tokens does not need to be accumulated separately - it's calculated in finalize, but keep it if the provider supplied it
                    if delta_total and "total_tokens" in meta:
                        fragment["total_tokens"] = delta_total

                    # Details (cache_read, cache_creation, reasoning, audio, etc.)
                    input_details = meta.get("input_token_details", {}) or {}
                    output_details = meta.get("output_token_details", {}) or {}
                    for detail_key, val in input_details.items():
                        # map to <detail>_tokens if it makes sense (numeric value)
                        if isinstance(val, (int, float)) and val > 0:
                            fragment[f"{detail_key}_tokens"] = (
                                fragment.get(f"{detail_key}_tokens", 0) + val
                            )
                    for detail_key, val in output_details.items():
                        if isinstance(val, (int, float)) and val > 0:
                            fragment[f"{detail_key}_tokens"] = (
                                fragment.get(f"{detail_key}_tokens", 0) + val
                            )

                    if fragment:
                        self.usage_session.accumulate(fragment)

                    # Update the last seen values
                    self._last_model_totals[model_name] = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    }

            # 2. Fallback to llm_output.* if present (do not double-count if usage_metadata already covered it)
            llm_output = getattr(response, "llm_output", None)
            if isinstance(llm_output, dict):
                usage_block = (
                    llm_output.get("token_usage")
                    or llm_output.get("usage")
                    or llm_output.get("usage_metadata")
                    or llm_output.get("response_metadata")
                )
                if isinstance(usage_block, dict):
                    fragment: Dict[str, Any] = {}
                    for k, v in usage_block.items():
                        if isinstance(v, (int, float)):
                            # If usage_metadata already contained input/output data, avoid duplication - check if k looks like *tokens
                            fragment[k] = v
                    if fragment:
                        self.usage_session.accumulate(fragment)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Usage callback accumulation error: {e}")

    def on_llm_error(self, error: Exception, **kwargs):  # type: ignore[override]
        try:
            self.usage_session.partial = True
        except Exception:  # pragma: no cover
            pass

    # --- Utility -------------------------------------------------------------
    def get_accumulated(self) -> Dict[str, Any]:
        return dict(self.usage_session.accumulated_usage)


__all__ = ["LLMUsageCallbackHandler"]


class ToolUsageCallbackHandler(BaseCallbackHandler):
    """Callback handler that logs and captures tool invocations.

    Captures:
    - on_tool_start: tool name and input string
    - on_tool_end: output string
    - on_tool_error: error message

    Stores a list of events in chronological order so the caller can inspect
    which MCP tools were used during an agent run.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []

    # LangChain core uses serialized schema for tools; handle both legacy and new signatures
    def on_tool_start(self, serialized, input_str: str | None = None, **kwargs):  # type: ignore[override]
        try:
            # `serialized` may be a dict like {"name": "browser_navigate", ...}
            name = None
            if isinstance(serialized, dict):
                name = (
                    serialized.get("name")
                    or serialized.get("tool")
                    or serialized.get("id")
                )
            event = {"event": "start", "tool": name or "<unknown>", "input": input_str}
            logger.info(f"Tool start: {event['tool']} input={input_str!r}")
            self.events.append(event)
        except Exception as e:  # pragma: no cover
            logger.debug(f"ToolUsageCallbackHandler.on_tool_start error: {e}")

    def on_tool_end(self, output: str | None = None, **kwargs):  # type: ignore[override]
        try:
            # Attach output to the last 'start' if exists; else push standalone end
            payload = output or ""
            # Truncate very large payloads for logs
            short = payload if len(payload) <= 2000 else payload[:2000] + "…"
            # Try to annotate the most recent start
            for item in reversed(self.events):
                if item.get("event") == "start" and "output" not in item:
                    item["output"] = short
                    item["event"] = "finish"
                    logger.info("Tool end: %s", item.get("tool"))
                    break
            else:
                self.events.append({"event": "finish", "output": short})
        except Exception as e:  # pragma: no cover
            logger.debug(f"ToolUsageCallbackHandler.on_tool_end error: {e}")

    def on_tool_error(self, error: Exception, **kwargs):  # type: ignore[override]
        try:
            err_text = str(error)
            self.events.append({"event": "error", "error": err_text})
            logger.warning("Tool error: %s", err_text)
        except Exception:  # pragma: no cover
            pass

    def get_events(self) -> list[dict]:
        return list(self.events)


__all__.append("ToolUsageCallbackHandler")

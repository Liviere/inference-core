"""Loop guard that breaks out of repeated identical failing tool calls.

Some models / provider integrations can get stuck issuing the *same* tool call
with the *same* (often empty or malformed) arguments, getting the *same*
validation error back, and retrying verbatim — burning the entire tool-call run
limit without making progress (observed with the DeepInfra streaming tool-call
bug fixed elsewhere, but it can happen with any weak model).

``RepeatedToolErrorGuardMiddleware`` is a small, provider-agnostic safety net:
before each model call it inspects the trailing message history and, if the same
tool was called with identical arguments and errored ``threshold`` times in a
row, it short-circuits the agent loop (``jump_to="end"``) with an explanatory
message instead of letting the model try the identical call yet again.

It is intentionally fail-soft: any unexpected error in the scan is swallowed and
the loop proceeds normally (the existing tool-call limit remains the backstop).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 3


def _canonical_args(args: Any) -> str:
    """Stable signature for tool-call arguments (handles the ``{}`` case)."""
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except Exception:
        return repr(args)


class RepeatedToolErrorGuardMiddleware(AgentMiddleware):
    """Stop the agent loop on repeated identical failing tool calls.

    Args:
        threshold: Number of consecutive identical erroring tool-call turns that
            triggers the break. Defaults to 3 (allow three attempts, then stop).
    """

    def __init__(self, threshold: int = _DEFAULT_THRESHOLD) -> None:
        super().__init__()
        self.threshold = max(1, threshold)

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: dict[str, Any], runtime: Any
    ) -> dict[str, Any] | None:
        try:
            return self._evaluate(state)
        except Exception:  # fail-soft — never break the loop ourselves
            logger.debug("RepeatedToolErrorGuard scan failed", exc_info=True)
            return None

    def _evaluate(self, state: dict[str, Any]) -> dict[str, Any] | None:
        messages = state.get("messages") or []

        # Group messages into turns: each AIMessage carrying tool_calls, paired
        # with the ToolMessages that answered it (keyed by tool_call_id).
        turns: list[tuple[AIMessage, dict[str, ToolMessage]]] = []
        current_ai: AIMessage | None = None
        current_results: dict[str, ToolMessage] = {}
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                if current_ai is not None:
                    turns.append((current_ai, current_results))
                current_ai = msg
                current_results = {}
            elif isinstance(msg, ToolMessage) and current_ai is not None:
                if msg.tool_call_id:
                    current_results[msg.tool_call_id] = msg
        if current_ai is not None:
            turns.append((current_ai, current_results))

        # Signature of the *erroring* tool calls in a turn. ``None`` when the
        # turn produced no errors (so it cannot extend a failing streak).
        def error_signature(
            ai: AIMessage, results: dict[str, ToolMessage]
        ) -> tuple | None:
            errored: list[tuple[str, str]] = []
            for tc in ai.tool_calls:
                result = results.get(tc.get("id"))
                if result is not None and getattr(result, "status", None) == "error":
                    errored.append(
                        (tc.get("name") or "", _canonical_args(tc.get("args")))
                    )
            return tuple(sorted(errored)) if errored else None

        if not turns:
            return None

        last_sig = error_signature(*turns[-1])
        if last_sig is None:
            return None

        streak = 0
        for ai, results in reversed(turns):
            if error_signature(ai, results) == last_sig:
                streak += 1
            else:
                break

        if streak < self.threshold:
            return None

        tool_names = ", ".join(sorted({name for name, _ in last_sig})) or "a tool"
        logger.warning(
            "RepeatedToolErrorGuard: breaking loop after %d identical failing "
            "calls to %s",
            streak,
            tool_names,
        )
        message = (
            f"I was unable to complete this step: the call to {tool_names} kept "
            f"failing with the same error {streak} times in a row. I'm stopping "
            "here to avoid an unproductive loop. Please review the request or try "
            "again with adjusted instructions."
        )
        return {"jump_to": "end", "messages": [AIMessage(content=message)]}

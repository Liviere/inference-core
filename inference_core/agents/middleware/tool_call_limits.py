"""Factory for building ToolCallLimitMiddleware instances from YAML config.

Usage::

    from inference_core.agents.middleware.tool_call_limits import (
        build_tool_call_limit_middleware,
        generate_tool_call_limits_instructions,
    )

    middleware = build_tool_call_limit_middleware(agent_config.tool_call_limits)
    # Returns a list of ToolCallLimitMiddleware instances (may be empty).

    instructions = generate_tool_call_limits_instructions(agent_config.tool_call_limits)
    # Returns a system prompt block (str or None).

A caller may pass ``uncounted``, a predicate on a tool's result: a per-tool
limit then gives back the calls whose result it accepts (a call that ran out
of time, say), so they do not use up the tool's limit. The global limit still
counts every call. Without it, every call counts, as in LangChain.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage, ToolMessage

if TYPE_CHECKING:
    from inference_core.llm.config import ToolCallLimitsConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# System prompt instructions
# ------------------------------------------------------------------

_TOOL_CALL_LIMITS_INSTRUCTIONS = """\
## Tool-Call Limit Policy

Your tool usage is subject to two independent kinds of call limits:

- **Run limit** — caps how many times you can call a tool (or all tools) \
within a single invocation (i.e. one user message → one assistant response). \
This counter **resets** every time the user sends a new message. \
A run-limit block is therefore **temporary** — if the user asks you to \
continue, you SHOULD attempt to use tools again because the limit has reset.
- **Thread limit** — caps total tool calls across the entire conversation \
thread. This counter does **not** reset between messages. A thread-limit \
block is **permanent** for this conversation.

The limit-exceeded message you receive does not specify which kind of limit \
was hit. **Assume it is a run limit** (the more common case) unless you have \
strong evidence that you have been blocked across multiple consecutive \
invocations for the same tool.

### When you hit a limit

1. **Summarize progress** — Briefly list what you have already accomplished \
and what information you have gathered so far.
2. **Report blockers** — If there are unresolved problems or missing data \
that prevented you from completing the task, describe them clearly.
3. **Plan next steps** — Propose concrete follow-up actions the user can \
request in a follow-up message to finish the task without repeating work \
already done.

Do NOT keep retrying the blocked tool within the **same** response. \
Use the information you already have to produce the best possible answer.

### When the user asks you to continue

If the user explicitly asks you to continue or retry after a limit message, \
**do not refuse**. The run limit has reset with the new invocation, so you \
should resume tool usage normally. Only refuse if you are certain the \
thread limit for that tool has been exhausted."""


def generate_tool_call_limits_instructions(
    config: "ToolCallLimitsConfig | None",
) -> str | None:
    """Generate system prompt instructions for tool-call limits.

    Returns the instruction block when *config* defines at least one limit,
    or ``None`` when no limits are active (so callers can skip concatenation).
    The output includes the static policy text followed by a dynamic table
    showing the actual configured limits.
    """
    if config is None:
        return None
    if config.global_limit is None and not config.per_tool:
        return None

    parts = [_TOOL_CALL_LIMITS_INSTRUCTIONS, "", "### Your current limits", ""]

    if config.global_limit:
        gl = config.global_limit
        parts.append("**Global (all tools combined):**")
        if gl.run_limit is not None:
            parts.append(f"- run limit: {gl.run_limit} calls per invocation")
        if gl.thread_limit is not None:
            parts.append(f"- thread limit: {gl.thread_limit} calls per conversation")
        parts.append("")

    if config.per_tool:
        parts.append("**Per-tool overrides:**")
        for entry in config.per_tool:
            limits = []
            if entry.run_limit is not None:
                limits.append(f"run={entry.run_limit}")
            if entry.thread_limit is not None:
                limits.append(f"thread={entry.thread_limit}")
            parts.append(f"- `{entry.tool_name}`: {', '.join(limits)}")
        parts.append("")

    return "\n".join(parts).rstrip()


UncountedResult = Callable[[ToolMessage], bool]


def _last_batch_results(messages: list[Any]) -> list[ToolMessage]:
    """The results of the tool calls of the model's message before the last.

    They are the ``ToolMessage``s between the last two ``AIMessage``s: each
    model call's ``after_model`` sees one batch of results, once.
    """
    seen_last_ai = False
    results: list[ToolMessage] = []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            if seen_last_ai:
                break
            seen_last_ai = True
            continue
        if seen_last_ai and isinstance(message, ToolMessage):
            results.append(message)
    return results


class UncountingToolCallLimitMiddleware(ToolCallLimitMiddleware):
    """A per-tool ``ToolCallLimitMiddleware`` that gives some calls back.

    LangChain counts a call when the model asks for it, before it runs. This
    one, before it counts the model's new calls, gives back the calls of the
    previous batch whose result ``uncounted`` accepts, to both counts (never
    below zero). It keeps nothing of its own: parallel calls of one batch
    write no state when they end.
    """

    def __init__(self, *, uncounted: UncountedResult, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.uncounted = uncounted

    @property
    def name(self) -> str:
        # The same name as LangChain's: one limit of a tool, whichever counts.
        if self.tool_name:
            return f"{ToolCallLimitMiddleware.__name__}[{self.tool_name}]"
        return ToolCallLimitMiddleware.__name__

    def _given_back(self, messages: list[Any]) -> int:
        given_back = 0
        for result in _last_batch_results(messages):
            if self.tool_name is not None and result.name != self.tool_name:
                continue
            try:
                if self.uncounted(result):
                    given_back += 1
            except Exception:
                logger.warning(
                    "ToolCallLimit: the uncounted check failed for tool %s",
                    result.name,
                    exc_info=True,
                )
        return given_back

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        given_back = self._given_back(state.get("messages", []))
        if not given_back:
            return super().after_model(state, runtime)

        count_key = self.tool_name or "__all__"
        thread_counts = dict(state.get("thread_tool_call_count", {}))
        run_counts = dict(state.get("run_tool_call_count", {}))
        thread_counts[count_key] = max(
            0, thread_counts.get(count_key, 0) - given_back
        )
        run_counts[count_key] = max(0, run_counts.get(count_key, 0) - given_back)
        logger.debug(
            "ToolCallLimit: tool=%s gave back %d call(s)", self.tool_name, given_back
        )

        update = super().after_model(
            {
                **state,
                "thread_tool_call_count": thread_counts,
                "run_tool_call_count": run_counts,
            },
            runtime,
        )
        if update is None:
            return {
                "thread_tool_call_count": thread_counts,
                "run_tool_call_count": run_counts,
            }
        return update


def build_tool_call_limit_middleware(
    config: "ToolCallLimitsConfig | None",
    *,
    uncounted: UncountedResult | None = None,
) -> list[ToolCallLimitMiddleware]:
    """Build ``ToolCallLimitMiddleware`` instances from a config block.

    Args:
        config: Parsed ``ToolCallLimitsConfig`` from the agent's YAML.
            ``None`` means no limits — returns an empty list.
        uncounted: Optional predicate on a tool's result. When given, a
            per-tool limit does not count the calls whose result it accepts
            (``UncountingToolCallLimitMiddleware``); the global limit counts
            every call either way. ``None`` (the default) counts every call.

    Returns:
        Ordered list of middleware: global limiter first, then per-tool.
    """
    if config is None:
        return []

    instances: list[ToolCallLimitMiddleware] = []

    # --- Global limit (tool_name=None) ---
    if config.global_limit is not None:
        gl = config.global_limit
        instances.append(
            ToolCallLimitMiddleware(
                tool_name=None,
                run_limit=gl.run_limit,
                thread_limit=gl.thread_limit,
                exit_behavior=gl.exit_behavior,
            )
        )
        logger.debug(
            "ToolCallLimit: global run_limit=%s thread_limit=%s exit=%s",
            gl.run_limit,
            gl.thread_limit,
            gl.exit_behavior,
        )

    # --- Per-tool limits ---
    for entry in config.per_tool:
        limit_params = {
            "tool_name": entry.tool_name,
            "run_limit": entry.run_limit,
            "thread_limit": entry.thread_limit,
            "exit_behavior": entry.exit_behavior,
        }
        instances.append(
            UncountingToolCallLimitMiddleware(uncounted=uncounted, **limit_params)
            if uncounted is not None
            else ToolCallLimitMiddleware(**limit_params)
        )
        logger.debug(
            "ToolCallLimit: tool=%s run_limit=%s thread_limit=%s exit=%s",
            entry.tool_name,
            entry.run_limit,
            entry.thread_limit,
            entry.exit_behavior,
        )

    return instances

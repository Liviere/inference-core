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
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain.agents.middleware import ToolCallLimitMiddleware

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


def build_tool_call_limit_middleware(
    config: "ToolCallLimitsConfig | None",
) -> list[ToolCallLimitMiddleware]:
    """Build ``ToolCallLimitMiddleware`` instances from a config block.

    Args:
        config: Parsed ``ToolCallLimitsConfig`` from the agent's YAML.
            ``None`` means no limits — returns an empty list.

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
        instances.append(
            ToolCallLimitMiddleware(
                tool_name=entry.tool_name,
                run_limit=entry.run_limit,
                thread_limit=entry.thread_limit,
                exit_behavior=entry.exit_behavior,
            )
        )
        logger.debug(
            "ToolCallLimit: tool=%s run_limit=%s thread_limit=%s exit=%s",
            entry.tool_name,
            entry.run_limit,
            entry.thread_limit,
            entry.exit_behavior,
        )

    return instances

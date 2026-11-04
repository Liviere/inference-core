"""Which ``task`` call, if any, the code running now is a subagent's run of.

``BriefedSubAgentMiddleware`` sets it around each ``task`` call it lets
through, for the call's whole run: the subagent's graph runs inside the tool,
and LangGraph hands each node it runs a copy of the context it was started
in, so the subagent's model calls see it and the parent's do not. Two
subagents running at the same time each see their own. A run resumed after an
interrupt runs the ``task`` call again, and with it this.

It says nothing to anyone by itself; a middleware that wants to know whose a
model call is reads it (``CostTrackingMiddleware.log_subagent``).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class SubagentTask:
    """A ``task`` call that runs a subagent."""

    subagent_type: str
    tool_call_id: str


_RUNNING: ContextVar[Optional[SubagentTask]] = ContextVar(
    "inference_core_subagent_task", default=None
)


def running_subagent_task() -> Optional[SubagentTask]:
    """The innermost ``task`` call whose subagent is running here, if any."""
    return _RUNNING.get()


@contextmanager
def subagent_task_running(task: Optional[SubagentTask]) -> Iterator[None]:
    """``task`` as the running one for the block; ``None`` changes nothing."""
    if task is None:
        yield
        return
    token = _RUNNING.set(task)
    try:
        yield
    finally:
        _RUNNING.reset(token)


__all__ = ["SubagentTask", "running_subagent_task", "subagent_task_running"]

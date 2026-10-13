"""The whole brief of a delegated task reaches the subagent.

deepagents' ``task`` tool takes the brief as ``description``: that text is the
subagent's only message. A model that learned another agent framework's tool
of the same name writes a short title there and puts the brief under
``prompt``, an argument this tool does not have and whose schema drops it
without a word. The subagent is then started on a title of a few words. A
researcher guesses its job from it, and a writer that was to be handed
findings is handed none and answers with nothing.

So a ``task`` call that carries a ``prompt`` gets its brief put where the tool
reads it, before the tool runs. A call written the tool's own way is not
touched.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest

# The tool deepagents' SubAgentMiddleware adds, and the argument it reads the
# subagent's task from.
TASK_TOOL_NAME = "task"
_BRIEF = "description"
# Where a model that knows another ``task`` tool writes it.
_MISNAMED_BRIEF = "prompt"


def whole_brief(args: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """The arguments of a ``task`` call with the brief where the tool reads it.

    ``None`` when there is nothing to move. What stood under ``description``
    is kept in front of the brief unless the brief already says it: it is
    then the title of the task, a line the subagent can use.
    """
    misnamed = args.get(_MISNAMED_BRIEF)
    if not isinstance(misnamed, str) or not misnamed.strip():
        return None
    brief = misnamed.strip()
    title = args.get(_BRIEF)
    title = title.strip() if isinstance(title, str) else ""
    if title and title not in brief:
        brief = f"{title}\n\n{brief}"
    moved = {key: value for key, value in args.items() if key != _MISNAMED_BRIEF}
    moved[_BRIEF] = brief
    return moved


def _with_whole_brief(request: ToolCallRequest) -> ToolCallRequest:
    call = request.tool_call
    if call.get("name") != TASK_TOOL_NAME:
        return request
    moved = whole_brief(call.get("args") or {})
    if moved is None:
        return request
    return request.override(tool_call={**call, "args": moved})


class TaskBriefMiddleware(AgentMiddleware):
    """Give a subagent the brief its parent wrote, whichever argument it is in."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        return handler(_with_whole_brief(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        return await handler(_with_whole_brief(request))


__all__ = ["TASK_TOOL_NAME", "TaskBriefMiddleware", "whole_brief"]

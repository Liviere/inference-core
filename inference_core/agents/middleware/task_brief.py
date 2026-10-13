"""The whole brief of a delegated task reaches the subagent.

deepagents' ``task`` tool has two arguments. It takes the brief as
``description``, and that text is the subagent's only message. A model that
learned another agent framework's tool of the same name writes a short title
there and puts the brief under an argument of its own choosing: ``prompt`` in
one run, ``content`` in the next. The tool's schema drops what it does not
know without a word, so the subagent is started on a title of a few words. A
researcher guesses its job from it, and a writer that was to be handed
findings is handed none and answers with nothing.

So before the tool runs, the text a ``task`` call carries beside its title is
put where the tool reads it. A call written the tool's own way is not touched.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest

# The tool deepagents' SubAgentMiddleware adds.
TASK_TOOL_NAME = "task"
# The argument it reads the subagent's task from, and the two it has.
_BRIEF = "description"
_READ = frozenset({_BRIEF, "subagent_type"})

# A text of one word that is no longer than this picks something (a model, a
# mode): it is not what a subagent is to be told.
_LONGEST_WORD_THAT_IS_NO_BRIEF = 40


def _is_brief(text: str, title: str) -> bool:
    """Whether *text*, found under an argument the tool does not read, is the task.

    It says more than the title does, and it is a sentence rather than a
    single word such as ``lite``.
    """
    if len(text) <= len(title):
        return False
    return len(text.split()) > 1 or len(text) > _LONGEST_WORD_THAT_IS_NO_BRIEF


def whole_brief(args: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """The arguments of a ``task`` call with the brief where the tool reads it.

    ``None`` when there is nothing to move. What stood under ``description``
    is kept in front of the brief unless the brief already says it: it is
    then the title of the task, a line the subagent can use.
    """
    title = args.get(_BRIEF)
    title = title.strip() if isinstance(title, str) else ""
    misplaced = {
        key: value.strip()
        for key, value in args.items()
        if key not in _READ and isinstance(value, str) and _is_brief(value, title)
    }
    if not misplaced:
        return None
    parts = list(misplaced.values())
    if title and not any(title in part for part in parts):
        parts.insert(0, title)
    moved = {key: value for key, value in args.items() if key not in misplaced}
    moved[_BRIEF] = "\n\n".join(parts)
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

"""A delegated task has a label and a brief, each under an argument of its own.

deepagents' ``task`` tool has two arguments and takes the subagent's task from
``description``: that text is the subagent's only message. A model that
learned another agent framework's tool of the same name reads ``description``
as a short label. It writes a title there and the brief under ``prompt``,
under ``content``, or not at all, and the subagent is started on a few words:
a researcher guesses its job from them, and a writer that was to be handed
findings is handed none and answers with nothing.

So the tool is given the layout such a model expects. ``description`` is the
label, ``prompt`` is the brief and is required, and the subagent is handed
``prompt``. A call that carries its brief somewhere else is read for what it
means before the tool runs, and a call that carries none is refused in words
that say what to write: a required argument alone is satisfied by copying the
title into it.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Mapping

from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# The tool deepagents' SubAgentMiddleware adds, and the arguments it is given
# here. deepagents' own tool has the first and the last of them.
TASK_TOOL_NAME = "task"
_LABEL = "description"
_BRIEF = "prompt"
_AGENT = "subagent_type"
_OWN = frozenset({_LABEL, _BRIEF, _AGENT})
_INNER = frozenset({_LABEL, _AGENT})

# A label is one line of a few words. The schema asks for three to eight; a
# text of one line is still taken for a label up to this many.
_MOST_WORDS_IN_A_LABEL = 12
_WORDS_IN_A_DERIVED_LABEL = 8

# A text of one word that is no longer than this picks something (a model, a
# mode): it is not what a subagent is to be told.
_LONGEST_WORD_THAT_IS_NO_BRIEF = 40

# How a call came to its brief.
AS_WRITTEN = "as_written"
LABELLED = "labelled"
JOINED = "joined"
MOVED = "moved"
FROM_LABEL = "from_label"
MISSING = "missing"

_ARGUMENTS_NOTE = (
    "\n\n## Arguments\n\n"
    "`prompt` is the only thing the subagent reads: write the whole brief "
    "there. `description` is a label of a few words that the subagent never "
    "sees. A call without `prompt` is refused."
)

NO_BRIEF = (
    "`task` was called without a brief. `description` is only a short label: "
    "the subagent never sees it, and it sees nothing of this conversation. "
    "Call `task` again with the same `subagent_type` and write everything the "
    "subagent needs in `prompt`: the goal, the facts and earlier results to "
    "work from, every limit, and what to return."
)


class TaskCall(BaseModel):
    """The arguments of ``task`` as the parent's model is shown them."""

    description: str = Field(
        description=(
            "A short label for this task, 3 to 8 words, e.g. "
            '"Summarise the release notes". A title for logs only: the '
            "subagent never sees it."
        )
    )
    prompt: str = Field(
        description=(
            "The whole brief, and the subagent's only message. The subagent "
            "sees nothing of this conversation, so write here everything it "
            "needs: the goal, the facts and earlier results to work from, "
            "every limit the user set, and exactly what to return."
        )
    )
    subagent_type: str = Field(
        description=(
            "The type of subagent to use. Must be one of the available agent "
            "types listed in the tool description."
        )
    )


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _says_more_than_a_label(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return len(lines) > 1 or len(text.split()) > _MOST_WORDS_IN_A_LABEL


def _label_of(text: str) -> str:
    """A label for a task that came without one: how its brief begins."""
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    words = first.split()
    if len(words) <= _WORDS_IN_A_DERIVED_LABEL:
        return first
    return " ".join(words[:_WORDS_IN_A_DERIVED_LABEL]) + "…"


def _is_brief(text: str, label: str) -> bool:
    """Whether *text*, found under an argument the tool does not have, is the task.

    It says more than the label does, and it is a sentence rather than a
    single word such as ``lite``.
    """
    if len(text) <= len(label):
        return False
    return len(text.split()) > 1 or len(text) > _LONGEST_WORD_THAT_IS_NO_BRIEF


def settle(args: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """The arguments of a ``task`` call with its brief under ``prompt``, and how.

    A call written the tool's way is returned as it is. Otherwise the brief is
    looked for where models put it:

    - under ``description`` with a pointer to it under ``prompt`` (a model
      that goes by the calls it sees in an older conversation): the two are
      joined;
    - under an argument the tool does not have (``content``,
      ``instructions``): moved;
    - under ``description`` alone, the way deepagents' own tool takes it and
      every call made before this layout was written: taken from there.

    A call with a label and nothing else has no brief, and none is made up.
    """
    settled = dict(args)
    label = _text(args.get(_LABEL))
    written = args.get(_BRIEF)
    brief = _text(written)

    if written is not None and not isinstance(written, str):
        # Not a text at all: the tool's own validation says so.
        return settled, AS_WRITTEN

    if brief:
        how = AS_WRITTEN
        if (
            _says_more_than_a_label(label)
            and len(label) > len(brief)
            and label not in brief
        ):
            brief = f"{label}\n\n{brief}"
            settled[_BRIEF] = brief
            label = ""
            how = JOINED
        if not label:
            settled[_LABEL] = _label_of(brief)
            how = LABELLED if how == AS_WRITTEN else how
        return settled, how

    misplaced = {
        key: value.strip()
        for key, value in args.items()
        if key not in _OWN and isinstance(value, str) and _is_brief(value, label)
    }
    if misplaced:
        for key in misplaced:
            del settled[key]
        brief = "\n\n".join(misplaced.values())
        settled[_BRIEF] = brief
        if not label:
            settled[_LABEL] = _label_of(brief)
        return settled, MOVED

    if _says_more_than_a_label(label):
        settled[_BRIEF] = label
        settled[_LABEL] = _label_of(label)
        return settled, FROM_LABEL

    settled.pop(_BRIEF, None)
    return settled, MISSING


def briefed(args: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments of a ``task`` call with its brief where the tool reads it.

    A result without ``prompt`` is a call that carried no brief.
    """
    return settle(args)[0]


def with_prompt(inner: BaseTool) -> BaseTool:
    """deepagents' ``task`` tool under the arguments of :class:`TaskCall`.

    The subagent is handed ``prompt``; the label is for whoever reads the
    call. Raises when *inner* is not the tool this was written against, so a
    deepagents that builds its tool another way is noticed when an agent is
    built and not when a subagent is started on the wrong text.
    """
    schema = getattr(inner, "args_schema", None)
    fields = frozenset(getattr(schema, "model_fields", None) or ())
    run = getattr(inner, "func", None)
    arun = getattr(inner, "coroutine", None)
    if inner.name != TASK_TOOL_NAME or fields != _INNER or run is None or arun is None:
        raise RuntimeError(
            "deepagents' subagent tool is not what inference-core wraps: "
            f"expected a tool {TASK_TOOL_NAME!r} with the arguments "
            f"{sorted(_INNER)} and a sync and an async body, got "
            f"{inner.name!r} with {sorted(fields)}. The wrapper in "
            "inference_core/agents/middleware/task_brief.py has to follow it."
        )

    def task(
        description: str, prompt: str, subagent_type: str, runtime: ToolRuntime
    ) -> Any:
        return run(description=prompt, subagent_type=subagent_type, runtime=runtime)

    async def atask(
        description: str, prompt: str, subagent_type: str, runtime: ToolRuntime
    ) -> Any:
        return await arun(
            description=prompt, subagent_type=subagent_type, runtime=runtime
        )

    return StructuredTool.from_function(
        name=TASK_TOOL_NAME,
        func=task,
        coroutine=atask,
        description=inner.description + _ARGUMENTS_NOTE,
        infer_schema=False,
        args_schema=TaskCall,
        return_direct=inner.return_direct,
        response_format=inner.response_format,
        metadata=inner.metadata,
        tags=inner.tags,
    )


def handed_back(result: Any, keys: frozenset[str]) -> Any:
    """What of a finished subagent's state the parent is given: its answer.

    deepagents hands the parent the subagent's whole state but a few keys.
    The rest is what the subagent's middleware kept about its own run
    (counters, the last request it made, what it recalled). Given to the
    parent, it overwrites the parent's record of its own run; and when two
    subagents finish in the same step, both write a key that takes one value
    per step and the run fails. So only ``messages`` and *keys* (state that
    is the subagent's work, merged by a reducer) go back.
    """
    if not isinstance(result, Command) or not isinstance(result.update, dict):
        return result
    update = {
        key: value
        for key, value in result.update.items()
        if key == "messages" or key in keys
    }
    if len(update) == len(result.update):
        return result
    return Command(
        graph=result.graph, update=update, resume=result.resume, goto=result.goto
    )


# State a subagent hands back besides its messages: the files it wrote, which
# deepagents merges with a reducer.
RETURNED_STATE_KEYS = frozenset({"files"})


class BriefedSubAgentMiddleware(SubAgentMiddleware):
    """``SubAgentMiddleware`` whose ``task`` takes a label and a required brief.

    Takes what ``SubAgentMiddleware`` takes, and ``returned_state_keys``: the
    state, besides messages, a subagent hands back to the parent (see
    :func:`handed_back`). The schema and the reading of a call are one class
    on purpose: a tool that asks for ``prompt`` and a call that is not read
    for where its brief is would refuse every call written the way
    deepagents' own tool takes it.
    """

    def __init__(
        self,
        *,
        returned_state_keys: frozenset[str] = RETURNED_STATE_KEYS,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.returned_state_keys = frozenset(returned_state_keys)
        self._brief_the_tool()

    def _brief_the_tool(self) -> None:
        [inner] = self.tools
        self.tools = [with_prompt(inner)]

    @property
    def private_state_keys(self) -> frozenset[str]:
        return SubAgentMiddleware.private_state_keys.fget(self)

    @private_state_keys.setter
    def private_state_keys(self, value: frozenset[str]) -> None:
        # The parent builds its tool anew here, with its own arguments.
        SubAgentMiddleware.private_state_keys.fset(self, value)
        self._brief_the_tool()

    def _settled(self, request: ToolCallRequest) -> ToolCallRequest | ToolMessage:
        call = request.tool_call
        if call.get("name") != TASK_TOOL_NAME:
            return request
        args, how = settle(call.get("args") or {})
        if how == AS_WRITTEN:
            return request
        agent = args.get(_AGENT)
        if how == MISSING:
            logger.warning(
                "task call without a brief refused (subagent_type=%r)", agent
            )
            return ToolMessage(
                content=NO_BRIEF,
                name=TASK_TOOL_NAME,
                tool_call_id=call.get("id"),
                status="error",
            )
        logger.info("task call's brief %s (subagent_type=%r)", how, agent)
        return request.override(tool_call={**call, "args": args})

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        settled = self._settled(request)
        if isinstance(settled, ToolMessage):
            return settled
        return self._answer(settled, handler(settled))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        settled = self._settled(request)
        if isinstance(settled, ToolMessage):
            return settled
        return self._answer(settled, await handler(settled))

    def _answer(self, request: ToolCallRequest, result: Any) -> Any:
        if request.tool_call.get("name") != TASK_TOOL_NAME:
            return result
        return handed_back(result, self.returned_state_keys)


__all__ = [
    "NO_BRIEF",
    "RETURNED_STATE_KEYS",
    "TASK_TOOL_NAME",
    "BriefedSubAgentMiddleware",
    "TaskCall",
    "briefed",
    "handed_back",
    "settle",
    "with_prompt",
]

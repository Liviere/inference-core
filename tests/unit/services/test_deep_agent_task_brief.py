"""A delegated task has a label and a brief, and the subagent is handed the brief.

deepagents' ``task`` tool takes the subagent's task from ``description``. A
model that knows another framework's tool of that name reads ``description``
as a label: it wrote the brief under ``prompt`` in one run, under ``content``
in the next, and in two more not at all, so the subagent was started on a few
words. The tool is given the layout such a model expects (a label, a required
``prompt``), a call that carries its brief elsewhere is read for what it
means, and a call without one is refused. The last tests run a real graph
with stand-ins for the two models, since what a tool call is made of by the
time the tool runs, and what the model is shown of a tool, is the agent
framework's to decide.
"""

import uuid
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from inference_core.agents.middleware import BriefedSubAgentMiddleware
from inference_core.agents.middleware.task_brief import (
    AS_WRITTEN,
    FROM_LABEL,
    JOINED,
    LABELLED,
    MISSING,
    MOVED,
    NO_BRIEF,
    briefed,
    settle,
    with_prompt,
)
from inference_core.services.agents_service import DeepAgentService

FINDINGS = (
    "Write three points in Polish from these findings:\n"
    "1. Version 158 ships JPEG-XL (https://example.org/158).\n"
    "2. An enterprise edition was announced (https://example.org/enterprise)."
)


# --------------------------------------------------------------------------- #
# Where a call's brief is
# --------------------------------------------------------------------------- #


def test_a_call_written_the_tools_way_is_left_alone():
    call = {
        "description": "Write the summary",
        "prompt": FINDINGS,
        "subagent_type": "writer",
        "model": "lite",
    }

    assert settle(call) == (call, AS_WRITTEN)


def test_a_short_brief_is_a_brief_when_it_is_written_as_one():
    """Nothing is asked of a text under ``prompt`` but that it is there."""
    call = {
        "description": "Rain tomorrow",
        "prompt": "Will it rain in Krakow tomorrow?",
        "subagent_type": "weather",
    }

    assert settle(call) == (call, AS_WRITTEN)


@pytest.mark.parametrize(
    "argument", ["content", "task", "instructions", "anything_else"]
)
def test_a_brief_under_an_argument_the_tool_does_not_have_is_moved(argument):
    """The same model wrote ``prompt`` in one run and ``content`` in the next."""
    args, how = settle(
        {
            "description": "Write the summary",
            argument: FINDINGS,
            "subagent_type": "writer",
            "model": "lite",
        }
    )

    assert how == MOVED
    # The label is not put in front of it: the subagent is handed the brief.
    assert args == {
        "description": "Write the summary",
        "prompt": FINDINGS,
        "subagent_type": "writer",
        "model": "lite",
    }


def test_a_word_that_picks_something_is_not_taken_for_the_brief():
    args, how = settle(
        {"description": "Write", "model": "lite", "subagent_type": "writer"}
    )

    assert how == MISSING
    assert "prompt" not in args


def test_a_text_shorter_than_the_label_is_not_the_brief():
    args, how = settle(
        {
            "description": "Summarise the three release notes",
            "note": "be brief",
            "subagent_type": "writer",
        }
    )

    assert how == MISSING
    assert "prompt" not in args


def test_a_brief_under_description_alone_is_taken_from_there():
    """How deepagents' own tool takes it, and how every call made before this
    layout is written: a conversation that goes on must not start failing."""
    args, how = settle({"description": FINDINGS, "subagent_type": "writer"})

    assert how == FROM_LABEL
    assert args["prompt"] == FINDINGS
    assert args["description"] == "Write three points in Polish from these findings:"
    assert args["subagent_type"] == "writer"


def test_a_brief_under_description_with_a_pointer_under_prompt_is_joined():
    """A model that goes by the calls of an older conversation and by the new
    schema at once."""
    args, how = settle(
        {
            "description": FINDINGS,
            "prompt": "Do as described.",
            "subagent_type": "writer",
        }
    )

    assert how == JOINED
    assert args["prompt"] == f"{FINDINGS}\n\nDo as described."
    assert args["description"] == "Write three points in Polish from these findings:"


def test_a_brief_written_under_both_is_not_said_twice():
    call = {"description": FINDINGS, "prompt": FINDINGS, "subagent_type": "writer"}

    assert settle(call) == (call, AS_WRITTEN)


def test_a_label_and_nothing_else_is_no_brief():
    for call in (
        {"description": "Write the summary", "subagent_type": "writer"},
        {"description": "Write the summary", "prompt": "  ", "subagent_type": "writer"},
        {"description": "Write the summary", "prompt": None, "subagent_type": "writer"},
        {"subagent_type": "writer"},
    ):
        args, how = settle(call)
        assert how == MISSING, call
        assert "prompt" not in args, call
        assert "prompt" not in briefed(call), call


def test_where_a_label_ends_and_a_brief_begins():
    twelve = "one two three four five six seven eight nine ten eleven twelve"

    assert settle({"description": twelve, "subagent_type": "w"})[1] == MISSING
    assert settle({"description": f"{twelve} more", "subagent_type": "w"})[1] == (
        FROM_LABEL
    )
    # Two lines are a brief however short they are.
    assert (
        settle({"description": "Find it.\nSay where.", "subagent_type": "w"})[1]
        == FROM_LABEL
    )


def test_a_brief_without_a_label_is_given_one():
    """A call with a real brief does not fail over the label it lacks."""
    long_first_line = (
        "Go through every release note of the last quarter and list changes"
    )
    args, how = settle({"prompt": long_first_line, "subagent_type": "writer"})

    assert how == LABELLED
    assert args["prompt"] == long_first_line
    assert args["description"] == "Go through every release note of the last…"

    moved = briefed({"content": "Find it.\nSay where.", "subagent_type": "w"})
    assert moved == {
        "description": "Find it.",
        "prompt": "Find it.\nSay where.",
        "subagent_type": "w",
    }


def test_a_prompt_that_is_no_text_is_left_for_the_tool_to_refuse():
    call = {"description": "Summary", "prompt": ["a", "b"], "subagent_type": "w"}

    assert settle(call) == (call, AS_WRITTEN)


def test_only_the_task_tool_is_looked_at():
    middleware = object.__new__(BriefedSubAgentMiddleware)
    request = MagicMock()
    request.tool_call = {
        "name": "write_file",
        "args": {"path": "notes.md", "content": FINDINGS},
        "id": "call-1",
    }
    handler = MagicMock(return_value="written")

    assert middleware.wrap_tool_call(request, handler) == "written"
    handler.assert_called_once_with(request)
    request.override.assert_not_called()


def test_a_tool_that_is_not_the_one_this_was_written_against_is_refused():
    """deepagents building its tool another way is noticed when an agent is
    built, not when a subagent is started on the wrong text."""

    @tool
    def task(instructions: str, agent: str) -> str:
        """Start a subagent."""
        return instructions

    with pytest.raises(RuntimeError, match="not what inference-core wraps"):
        with_prompt(task)


# --------------------------------------------------------------------------- #
# A run
# --------------------------------------------------------------------------- #


class _Scripted(BaseChatModel):
    """Answers with its replies in order, and keeps what it was sent and the
    tools it was bound with."""

    replies: List[Any]
    seen: List[List[BaseMessage]] = Field(default_factory=list)
    bound: List[dict] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_Scripted":
        self.bound[:] = [convert_to_openai_tool(one) for one in tools]
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        reply = self.replies[len(self.seen) - 1]
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _deep_service(model: BaseChatModel, **overrides: Any) -> DeepAgentService:
    """A ``DeepAgentService`` on *model*, with no config, database or keys."""
    factory = MagicMock()
    factory.get_agent_model_name.return_value = "scripted"
    factory.get_model_for_agent.return_value = model
    factory.aclose = AsyncMock()
    factory.config.models = {}
    factory.config.get_model_params.return_value = {}
    factory.config.get_specific_agent_config.return_value = MagicMock(
        local_tool_providers=[],
        mcp_profile=None,
        allowed_tools=None,
        tool_model_overrides=None,
        description="A coordinator",
        skills=None,
        subagents=None,
        interrupt_on=None,
        memory_tools=None,
        memory_session_context_enabled=None,
        memory_tool_instructions_enabled=None,
        tool_call_limits=None,
        response_format=None,
        fallback=None,
        generation_params=None,
        reasoning_output=False,
    )
    arguments = {
        "agent_name": "coordinator",
        "enable_cost_tracking": False,
        "user_id": uuid.uuid4(),
        "session_id": "sess-1",
        "request_id": "req-1",
    }
    arguments.update(overrides)
    with (
        patch(
            "inference_core.services.agents_service.get_model_factory",
            return_value=factory,
        ),
        patch(
            "inference_core.services.agents_service.get_llm_config",
            return_value=MagicMock(),
        ),
        patch(
            "inference_core.services.agents_service.get_settings",
            return_value=MagicMock(
                database_url="sqlite+aiosqlite:///test.db",
                agent_memory_enabled=False,
                llm_emulation_enabled=False,
                agent_snapshot_capture_enabled=False,
                agent_snapshot_replay_enabled=False,
                vector_dim=8,
            ),
        ),
    ):
        return DeepAgentService(**arguments)


def _task_call(args: dict, call_id: str = "task-1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": "task", "args": args, "id": call_id}]
    )


def _coordinator_and_writer(*task_args: dict):
    """A coordinator that makes the given ``task`` calls one after another and
    then answers, and the writer it hands them to."""
    writer = _Scripted(
        replies=[AIMessage(content="Three points, in Polish.") for _ in task_args]
    )
    coordinator = _Scripted(
        replies=[
            *(_task_call(args, f"task-{n}") for n, args in enumerate(task_args, 1)),
            AIMessage(content="Here is the summary."),
        ]
    )
    service = _deep_service(
        coordinator,
        subagents=[
            {
                "name": "writer",
                "description": "Writes summaries.",
                "system_prompt": "You are the writer.",
                "model": writer,
                "tools": [],
            }
        ],
    )
    return service, coordinator, writer


def _task_of(writer: _Scripted, run: int = 0) -> str:
    """What the subagent was asked: its one human message."""
    [asked] = [m for m in writer.seen[run] if isinstance(m, HumanMessage)]
    return asked.text if hasattr(asked, "text") else str(asked.content)


@pytest.mark.asyncio
async def test_a_subagent_is_handed_the_prompt_and_not_the_label():
    service, _coordinator, writer = _coordinator_and_writer(
        {
            "description": "Write the summary",
            "prompt": FINDINGS,
            "subagent_type": "writer",
        }
    )
    agent = await service.create_agent(system_prompt="You coordinate.")

    state = await agent.ainvoke({"messages": [("user", "What is new?")]})

    assert state["messages"][-1].content == "Here is the summary."
    assert _task_of(writer) == FINDINGS


@pytest.mark.asyncio
async def test_the_prompt_is_handed_on_in_a_run_that_is_not_async_too():
    service, _coordinator, writer = _coordinator_and_writer(
        {"description": "Summary", "prompt": FINDINGS, "subagent_type": "writer"}
    )
    agent = await service.create_agent(system_prompt="You coordinate.")

    agent.invoke({"messages": [("user", "What is new?")]})

    assert _task_of(writer) == FINDINGS


@pytest.mark.asyncio
async def test_the_coordinators_model_is_shown_a_label_and_a_required_prompt():
    service, coordinator, _writer = _coordinator_and_writer(
        {"description": "Summary", "prompt": FINDINGS, "subagent_type": "writer"}
    )
    agent = await service.create_agent(system_prompt="You coordinate.")

    await agent.ainvoke({"messages": [("user", "What is new?")]})

    [task] = [
        t["function"] for t in coordinator.bound if t["function"]["name"] == "task"
    ]
    parameters = task["parameters"]
    assert sorted(parameters["properties"]) == [
        "description",
        "prompt",
        "subagent_type",
    ]
    assert sorted(parameters["required"]) == ["description", "prompt", "subagent_type"]
    assert "never sees it" in parameters["properties"]["description"]["description"]
    assert "only message" in parameters["properties"]["prompt"]["description"]
    # deepagents' own description, with its list of subagents, and what this
    # layout adds to it.
    assert "- writer: Writes summaries." in task["description"]
    assert task["description"].rstrip().endswith("A call without `prompt` is refused.")


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["content", "instructions"])
async def test_a_brief_under_another_argument_reaches_the_subagent(argument):
    service, _coordinator, writer = _coordinator_and_writer(
        {
            "description": "Write the summary",
            argument: FINDINGS,
            "subagent_type": "writer",
            "model": "lite",
        }
    )
    agent = await service.create_agent(system_prompt="You coordinate.")

    await agent.ainvoke({"messages": [("user", "What is new?")]})

    assert _task_of(writer) == FINDINGS


@pytest.mark.asyncio
async def test_a_call_written_for_deepagents_own_tool_reaches_the_subagent_whole():
    """Every call a conversation made before this layout looks like this, and
    a model goes by the calls it sees."""
    service, _coordinator, writer = _coordinator_and_writer(
        {"description": FINDINGS, "subagent_type": "writer"}
    )
    agent = await service.create_agent(system_prompt="You coordinate.")

    state = await agent.ainvoke({"messages": [("user", "What is new?")]})

    assert _task_of(writer) == FINDINGS
    assert state["messages"][-1].content == "Here is the summary."


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True])
async def test_a_title_alone_starts_no_subagent_and_is_answered_in_words(sync):
    """What used to start a writer on four words. The coordinator is told what
    is missing, and its next call, with the brief, goes through."""
    service, coordinator, writer = _coordinator_and_writer(
        {"description": "Write the summary", "subagent_type": "writer"},
        {
            "description": "Write the summary",
            "prompt": FINDINGS,
            "subagent_type": "writer",
        },
    )
    # The first call starts nobody, so the writer answers once.
    writer.replies = writer.replies[:1]
    agent = await service.create_agent(system_prompt="You coordinate.")

    question = {"messages": [("user", "What is new?")]}
    state = agent.invoke(question) if sync else await agent.ainvoke(question)

    refusal = coordinator.seen[1][-1]
    assert isinstance(refusal, ToolMessage)
    assert refusal.status == "error"
    assert refusal.tool_call_id == "task-1"
    assert refusal.content == NO_BRIEF
    assert "`prompt`" in refusal.content
    assert len(writer.seen) == 1
    assert _task_of(writer) == FINDINGS
    assert state["messages"][-1].content == "Here is the summary."


def test_rebuilding_the_tool_does_not_bring_the_old_arguments_back():
    """``SubAgentMiddleware`` builds its tool anew when its private state keys
    are set, with its own two arguments."""
    middleware = BriefedSubAgentMiddleware(
        backend=MagicMock(),
        subagents=[
            {
                "name": "writer",
                "description": "Writes summaries.",
                "system_prompt": "You are the writer.",
                "model": _Scripted(replies=[]),
                "tools": [],
            }
        ],
    )
    [before] = middleware.tools

    middleware.private_state_keys = frozenset({"scratch"})

    [after] = middleware.tools
    assert after is not before
    assert middleware.private_state_keys == frozenset({"scratch"})
    for built in (before, after):
        assert sorted(built.args_schema.model_fields) == [
            "description",
            "prompt",
            "subagent_type",
        ]

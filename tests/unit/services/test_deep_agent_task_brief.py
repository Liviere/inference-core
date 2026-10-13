"""The whole brief of a delegated task reaches the subagent.

deepagents' ``task`` tool reads the subagent's task from ``description``. A
model that knows another framework's tool of that name writes a title there
and the brief under an argument of its own (``prompt`` in one run, ``content``
in the next), which the tool's schema drops in silence: the subagent was
started on a few words. The last tests run a real graph with stand-ins for
the two models, since what a tool call is made of by the time the tool runs
is the agent framework's to decide.
"""

import uuid
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from inference_core.agents.middleware import TaskBriefMiddleware
from inference_core.agents.middleware.task_brief import whole_brief
from inference_core.services.agents_service import DeepAgentService

FINDINGS = (
    "Write three points in Polish from these findings:\n"
    "1. Version 158 ships JPEG-XL (https://example.org/158).\n"
    "2. An enterprise edition was announced (https://example.org/enterprise)."
)


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #


def test_a_brief_under_another_tools_argument_is_put_where_this_one_reads_it():
    moved = whole_brief(
        {
            "description": "Write the summary",
            "prompt": FINDINGS,
            "subagent_type": "writer",
            "model": "lite",
        }
    )

    # The title stays in front of it: a line the subagent can use.
    assert moved["description"] == f"Write the summary\n\n{FINDINGS}"
    assert "prompt" not in moved
    assert moved["subagent_type"] == "writer"


@pytest.mark.parametrize(
    "argument", ["prompt", "content", "task", "instructions", "anything_else"]
)
def test_the_brief_is_found_under_whichever_argument_the_model_chose(argument):
    """The same model wrote ``prompt`` in one run and ``content`` in the next."""
    moved = whole_brief(
        {
            "description": "Write the summary",
            argument: FINDINGS,
            "subagent_type": "writer",
        }
    )

    assert moved == {
        "description": f"Write the summary\n\n{FINDINGS}",
        "subagent_type": "writer",
    }


def test_a_word_that_picks_something_is_not_taken_for_the_brief():
    """``model: lite`` came with a call whose brief was where it belongs."""
    assert (
        whole_brief(
            {"description": FINDINGS, "model": "lite", "subagent_type": "writer"}
        )
        is None
    )
    # Not even where there is no title to be longer than.
    assert whole_brief({"model": "lite", "subagent_type": "writer"}) is None

    moved = whole_brief(
        {
            "description": "Summary",
            "content": FINDINGS,
            "model": "lite",
            "subagent_type": "writer",
        }
    )
    assert moved["description"] == f"Summary\n\n{FINDINGS}"
    # What was not the brief stays as it came: the tool's schema drops it.
    assert moved["model"] == "lite"
    assert "content" not in moved


def test_a_text_shorter_than_the_title_is_not_the_brief():
    assert (
        whole_brief(
            {
                "description": FINDINGS,
                "note": "two sources at most",
                "subagent_type": "writer",
            }
        )
        is None
    )


def test_a_title_the_brief_already_says_is_not_said_twice():
    moved = whole_brief(
        {
            "description": "three points",
            "prompt": "Write three points from the findings.",
            "subagent_type": "writer",
        }
    )

    assert moved["description"] == "Write three points from the findings."


def test_a_brief_alone_is_the_task():
    moved = whole_brief({"prompt": f"  {FINDINGS}  ", "subagent_type": "writer"})

    assert moved == {"description": FINDINGS, "subagent_type": "writer"}


@pytest.mark.parametrize(
    "args",
    [
        {"description": FINDINGS, "subagent_type": "writer"},
        {"description": FINDINGS, "prompt": "", "subagent_type": "writer"},
        {"description": FINDINGS, "prompt": "   ", "subagent_type": "writer"},
        {"description": FINDINGS, "prompt": None, "subagent_type": "writer"},
        {"description": FINDINGS, "prompt": {"not": "text"}, "subagent_type": "w"},
        {},
    ],
)
def test_a_call_written_the_tools_own_way_is_left_alone(args):
    assert whole_brief(args) is None


def test_only_the_task_tool_is_looked_at():
    seen = []
    request = MagicMock()
    request.tool_call = {
        "name": "write_file",
        "args": {"description": "notes", "prompt": "not a brief"},
        "id": "call-1",
    }

    TaskBriefMiddleware().wrap_tool_call(request, seen.append)

    assert seen == [request]
    request.override.assert_not_called()


# --------------------------------------------------------------------------- #
# A run
# --------------------------------------------------------------------------- #


class _Scripted(BaseChatModel):
    """Answers with its replies in order and keeps what it was sent."""

    replies: List[Any]
    seen: List[List[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_Scripted":
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


def _coordinator_and_writer(task_args: dict):
    writer = _Scripted(replies=[AIMessage(content="Three points, in Polish.")])
    coordinator = _Scripted(
        replies=[
            AIMessage(
                content="",
                tool_calls=[{"name": "task", "args": task_args, "id": "task-1"}],
            ),
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
    return service, writer


def _task_of(writer: _Scripted) -> str:
    """What the subagent was asked: its one human message."""
    [asked] = [m for m in writer.seen[0] if isinstance(m, HumanMessage)]
    return asked.text if hasattr(asked, "text") else str(asked.content)


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["prompt", "content"])
async def test_a_subagent_is_handed_the_brief_and_not_its_title(argument):
    """The call as a model that knows another ``task`` tool writes it."""
    service, writer = _coordinator_and_writer(
        {
            "description": "Write the summary",
            argument: FINDINGS,
            "subagent_type": "writer",
            "model": "lite",
        }
    )
    agent = await service.create_agent(system_prompt="You coordinate.")

    state = await agent.ainvoke({"messages": [("user", "What is new?")]})

    assert state["messages"][-1].content == "Here is the summary."
    assert _task_of(writer) == f"Write the summary\n\n{FINDINGS}"


@pytest.mark.asyncio
async def test_a_call_written_the_tools_own_way_reaches_the_subagent_as_it_is():
    service, writer = _coordinator_and_writer(
        {"description": FINDINGS, "subagent_type": "writer"}
    )
    agent = await service.create_agent(system_prompt="You coordinate.")

    await agent.ainvoke({"messages": [("user", "What is new?")]})

    assert _task_of(writer) == FINDINGS


@pytest.mark.asyncio
async def test_the_brief_is_handed_on_in_a_run_that_is_not_async_too():
    service, writer = _coordinator_and_writer(
        {"description": "Summary", "prompt": FINDINGS, "subagent_type": "writer"}
    )
    agent = await service.create_agent(system_prompt="You coordinate.")

    agent.invoke({"messages": [("user", "What is new?")]})

    assert _task_of(writer) == f"Summary\n\n{FINDINGS}"

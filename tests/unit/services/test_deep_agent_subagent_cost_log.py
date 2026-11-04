"""``CostTrackingMiddleware.log_subagent``: whose call a usage log row is.

A subagent shares its parent's cost tracking, so its calls were logged like
the parent's. With the option each row of a subagent's call says which
subagent made it and for which ``task`` call; the parent's rows say nothing.
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import Field

from inference_core.agents.middleware import CostTrackingMiddleware
from inference_core.agents.middleware.subagent_task import (
    SubagentTask,
    running_subagent_task,
    subagent_task_running,
)
from inference_core.llm.usage_logging import UsageSession
from inference_core.services.agents_service import DeepAgentService
from tests.unit.services.test_deep_agent_task_brief import FINDINGS, _deep_service

Row = Tuple[int, Dict[str, Any]]


class _Scripted(BaseChatModel):
    """Answers with its replies in order, whichever thread asks."""

    replies: List[Any]
    asked: List[int] = Field(default_factory=lambda: [0])

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
        reply = self.replies[self.asked[0]]
        self.asked[0] += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _reply(tokens: int, **kwargs: Any) -> AIMessage:
    """A model's message whose usage says which call it was."""
    return AIMessage(
        content=kwargs.pop("content", ""),
        usage_metadata={
            "input_tokens": tokens,
            "output_tokens": 1,
            "total_tokens": tokens + 1,
        },
        **kwargs,
    )


def _task(call_id: str, subagent: str) -> Dict[str, Any]:
    return {
        "name": "task",
        "args": {"description": "Part", "prompt": FINDINGS, "subagent_type": subagent},
        "id": call_id,
    }


@pytest.fixture
def logged():
    """``(input tokens, details)`` of each usage log row written."""
    rows: List[Row] = []

    def finalize_sync(session, **kwargs):
        rows.append((kwargs["final_usage"]["input_tokens"], kwargs.get("details") or {}))
        return uuid.uuid4()

    async def finalize(session, **kwargs):
        return finalize_sync(session, **kwargs)

    with patch.object(UsageSession, "finalize_sync", finalize_sync), patch.object(
        UsageSession, "finalize", finalize
    ):
        yield rows


@pytest.fixture
def per_message(monkeypatch):
    # Two subagents at once on one instance: see ``persisted_per_message``.
    monkeypatch.setattr(CostTrackingMiddleware, "persisted_per_message", True)


def _whose(rows: List[Row]) -> Dict[int, Tuple[Optional[str], Optional[str]]]:
    return {
        tokens: (details.get("subagent"), details.get("task_call_id"))
        for tokens, details in rows
    }


async def _team_of_two(sync: bool) -> None:
    """A coordinator that hands two parts to two subagents in one message."""
    coordinator = _Scripted(
        replies=[
            _reply(10, tool_calls=[_task("task-1", "writer"), _task("task-2", "reader")]),
            _reply(20, content="Here it is."),
        ]
    )
    service = _deep_service(
        coordinator,
        middleware=[CostTrackingMiddleware(pricing_config=None)],
        subagents=[
            {
                "name": "writer",
                "description": "Writes.",
                "system_prompt": "You write.",
                "model": _Scripted(replies=[_reply(100, content="Written.")]),
                "tools": [],
            },
            {
                "name": "reader",
                "description": "Reads.",
                "system_prompt": "You read.",
                "model": _Scripted(replies=[_reply(300, content="Read.")]),
                "tools": [],
            },
        ],
    )
    agent = await service.create_agent(system_prompt="You coordinate.")
    if sync:
        agent.invoke({"messages": [("user", "Go.")]})
    else:
        await agent.ainvoke({"messages": [("user", "Go.")]})


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_by_default_a_row_does_not_say_whose_call_it_is(logged, per_message, sync):
    await _team_of_two(sync)

    assert sorted(tokens for tokens, _ in logged) == [10, 20, 100, 300]
    assert all("subagent" not in d and "task_call_id" not in d for _, d in logged)


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_with_the_option_each_subagents_row_says_whose_it_is(
    logged, per_message, monkeypatch, sync
):
    monkeypatch.setattr(CostTrackingMiddleware, "log_subagent", True)

    await _team_of_two(sync)

    assert sorted(tokens for tokens, _ in logged) == [10, 20, 100, 300]
    assert _whose(logged) == {
        10: (None, None),
        20: (None, None),
        100: ("writer", "task-1"),
        300: ("reader", "task-2"),
    }


@tool
def send(text: str) -> str:
    """Send a text."""
    return "sent"


def _approving_team(sender: List[AIMessage], coordinator: List[AIMessage], saver):
    service = _deep_service(
        _Scripted(replies=coordinator),
        middleware=[CostTrackingMiddleware(pricing_config=None)],
        subagents=[
            {
                "name": "sender",
                "description": "Sends.",
                "system_prompt": "You send.",
                "model": _Scripted(replies=sender),
                "tools": [send],
                "interrupt_on": {"send": True},
            }
        ],
    )
    service.checkpointer = saver
    return service


@pytest.mark.asyncio
async def test_a_subagent_resumed_after_its_approval_still_says_whose_its_calls_are(
    logged, monkeypatch
):
    monkeypatch.setattr(CostTrackingMiddleware, "log_subagent", True)
    monkeypatch.setattr(
        DeepAgentService, "subagent_approvals_ahead_of_cost_tracking", True
    )
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "approved"}}
    first = _approving_team(
        [_reply(100, tool_calls=[{"name": "send", "args": {"text": "x"}, "id": "c"}])],
        [_reply(10, tool_calls=[_task("task-1", "sender")])],
        saver,
    )
    stopped = await (await first.create_agent(system_prompt="Coordinate.")).ainvoke(
        {"messages": [("user", "Send it.")]}, config
    )
    assert stopped["__interrupt__"]

    # Another worker: a service and a cost tracking of its own.
    second = _approving_team(
        [_reply(200, content="Sent.")], [_reply(20, content="Done.")], saver
    )
    await (await second.create_agent(system_prompt="Coordinate.")).ainvoke(
        Command(resume={"decisions": [{"type": "approve"}]}), config
    )

    assert sorted(tokens for tokens, _ in logged) == [10, 20, 100, 200]
    assert _whose(logged) == {
        10: (None, None),
        20: (None, None),
        100: ("sender", "task-1"),
        200: ("sender", "task-1"),
    }


def test_the_running_task_is_the_innermost_and_ends_with_its_block():
    outer = SubagentTask("coordinator_of_its_own", "t-1")
    inner = SubagentTask("writer", "t-2")

    assert running_subagent_task() is None
    with subagent_task_running(outer):
        with subagent_task_running(inner):
            assert running_subagent_task() == inner
        assert running_subagent_task() == outer
        with subagent_task_running(None):
            assert running_subagent_task() == outer
    assert running_subagent_task() is None

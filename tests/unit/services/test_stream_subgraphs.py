"""What the stream callbacks get of a subagent's run.

A subagent started by ``task`` runs as a graph of its own inside the tool. By
default the stream is read without ``subgraphs`` and its thinking, text and
tools reach no callback: a client sees the ``task`` call and its result. With
``stream_subgraphs=True`` they do, each event with its namespace and the id
of the ``task`` call it belongs to, which the tool announces before the
subagent starts (LangGraph names a nested graph by its task, not by the tool
call). The run's own result, messages and cost stay the coordinator's.

The tests run a real graph with stand-ins for the models, since what a
nested event looks like is the agent framework's to decide.
"""

import json
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool

from inference_core.agents.middleware import BriefedSubAgentMiddleware
from inference_core.agents.middleware.task_brief import (
    SUBAGENT_RUN,
    announced_subagent_run,
)
from inference_core.services.agents_service import RunAccumulator
from tests.unit.services.test_agents_service import _make_agent_service
from tests.unit.services.test_deep_agent_task_brief import (
    FINDINGS,
    _deep_service,
    _Scripted,
)


class _Streaming(_Scripted):
    """A scripted model that streams each reply as one chunk, as a provider's
    model does, so the stream's ``messages`` mode gets its tokens."""

    def _stream(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        [generation] = self._generate(messages, stop, run_manager, **kwargs).generations
        reply = generation.message
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=reply.content,
                tool_call_chunks=[
                    {
                        "name": call["name"],
                        "args": json.dumps(call["args"]),
                        "id": call["id"],
                        "index": index,
                    }
                    for index, call in enumerate(reply.tool_calls)
                ],
            )
        )


@tool
def look(q: str) -> str:
    """Look something up."""
    return f"found {q}"


def _task(call_id: str, subagent: str) -> dict[str, Any]:
    return {
        "name": "task",
        "args": {
            "description": f"Part for the {subagent}",
            "prompt": FINDINGS,
            "subagent_type": subagent,
        },
        "id": call_id,
    }


def _subagent(name: str, look_id: str, answer: str) -> tuple[dict, _Scripted]:
    """A subagent that looks something up once and then answers."""
    model = _Streaming(
        replies=[
            AIMessage(
                content="",
                tool_calls=[{"name": "look", "args": {"q": name}, "id": look_id}],
            ),
            AIMessage(content=answer),
        ]
    )
    spec = {
        "name": name,
        "description": f"The {name}.",
        "system_prompt": f"You are the {name}.",
        "model": model,
        "tools": [look],
    }
    return spec, model


def _team(**overrides: Any):
    """A coordinator that hands two tasks out at once, to two subagents."""
    writer, _ = _subagent("writer", "w-look", "Written.")
    reader, _ = _subagent("reader", "r-look", "Read.")
    coordinator = _Streaming(
        replies=[
            AIMessage(
                content="",
                tool_calls=[_task("task-1", "writer"), _task("task-2", "reader")],
            ),
            AIMessage(content="Here is the summary."),
        ]
    )
    return _deep_service(coordinator, subagents=[writer, reader], **overrides)


class _Recorder:
    """The three callbacks, keeping what they were given."""

    def __init__(self) -> None:
        self.steps: list[tuple[str, Any]] = []
        self.tokens: list[tuple[str, dict]] = []
        self.custom: list[Any] = []

    def on_step(self, name: str, data: Any) -> None:
        self.steps.append((name, data))

    def on_token(self, text: str, meta: dict) -> None:
        self.tokens.append((text, meta))

    def on_custom(self, data: Any) -> None:
        self.custom.append(data)

    def nested_steps(self) -> list[tuple[str, Any]]:
        return [
            (name, data)
            for name, data in self.steps
            if isinstance(data, dict) and "ns" in data
        ]

    def tool_calls_by_parent(self) -> dict[str, list[str]]:
        """The ids of the tool calls the subagents made, by their task call."""
        made: dict[str, list[str]] = {}
        for _, data in self.nested_steps():
            for message in data.get("messages", []):
                for call in getattr(message, "tool_calls", None) or []:
                    made.setdefault(data["parent_tool_call_id"], []).append(call["id"])
        return made


async def _run(service, recorder: _Recorder, *, sync: bool = False) -> RunAccumulator:
    """Stream the service's agent and process the chunks as the run loops do."""
    agent = await service.create_agent(system_prompt="You coordinate.")
    accumulator = RunAccumulator()
    stream_args = (
        {"messages": [("user", "What is new?")]},
        {"configurable": {"thread_id": "t-1"}},
    )
    stream_kwargs = {
        "stream_mode": ["updates", "messages", "custom"],
        "version": "v2",
        **service.subgraph_stream_kwargs(),
    }

    def process(chunk: dict) -> None:
        service._process_stream_chunk(
            chunk,
            accumulator,
            on_step=recorder.on_step,
            on_token=recorder.on_token,
            on_custom=recorder.on_custom,
            cancel_check=None,
        )

    if sync:
        for chunk in agent.stream(*stream_args, **stream_kwargs):
            process(chunk)
    else:
        async for chunk in agent.astream(*stream_args, **stream_kwargs):
            process(chunk)
    return accumulator


# --------------------------------------------------------------------------- #
# Off by default
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_by_default_no_callback_gets_a_subagents_events():
    service = _team()
    recorder = _Recorder()

    accumulator = await _run(service, recorder)

    assert service.stream_subgraphs is False
    assert service.subgraph_stream_kwargs() == {}
    assert recorder.nested_steps() == []
    assert all("ns" not in meta for _, meta in recorder.tokens)
    assert "Written." not in [text for text, _ in recorder.tokens]
    # The task tool announces nothing.
    assert recorder.custom == []
    assert accumulator.subagent_calls == {}


def test_the_middleware_announces_nothing_unless_asked():
    middleware = BriefedSubAgentMiddleware(
        backend=MagicMock(),
        subagents=[
            {
                "name": "writer",
                "description": "Writes.",
                "system_prompt": "You write.",
                "model": _Scripted(replies=[]),
                "tools": [],
            }
        ],
    )

    assert middleware.announce_runs is False


# --------------------------------------------------------------------------- #
# With stream_subgraphs
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True])
async def test_each_subagents_steps_name_the_task_call_they_belong_to(sync):
    service = _team(stream_subgraphs=True)
    recorder = _Recorder()

    await _run(service, recorder, sync=sync)

    assert service.subgraph_stream_kwargs() == {"subgraphs": True}
    # Two tasks at once: the events of each subagent name its own call.
    assert recorder.tool_calls_by_parent() == {
        "task-1": ["w-look"],
        "task-2": ["r-look"],
    }
    for _, data in recorder.nested_steps():
        assert data["ns"][0].startswith("tools:")
    # The announcements are the stream's own business.
    assert all(SUBAGENT_RUN not in (c or {}) for c in recorder.custom)


@pytest.mark.asyncio
async def test_a_subagents_text_names_its_call_and_the_coordinators_does_not():
    service = _team(stream_subgraphs=True)
    recorder = _Recorder()

    await _run(service, recorder)

    by_text = {text: meta for text, meta in recorder.tokens}
    assert by_text["Written."]["parent_tool_call_id"] == "task-1"
    assert by_text["Read."]["parent_tool_call_id"] == "task-2"
    assert by_text["Written."]["type"] == "text"
    # The coordinator's answer is a token of the run itself, as it always was.
    assert "ns" not in by_text["Here is the summary."]
    assert "parent_tool_call_id" not in by_text["Here is the summary."]


@pytest.mark.asyncio
async def test_the_runs_result_and_cost_are_the_coordinators_own():
    service = _team(stream_subgraphs=True)
    recorder = _Recorder()

    accumulator = await _run(service, recorder)

    result = accumulator.extract_result()["messages"]
    assert result[-1].content == "Here is the summary."
    contents = [getattr(m, "content", None) for m in result]
    # The subagents' answers arrive as the task calls' results only.
    assert contents.count("Written.") == 1
    assert contents.count("Read.") == 1
    made = [call["id"] for m in result for call in getattr(m, "tool_calls", None) or []]
    assert made == ["task-1", "task-2"]
    # One model step for each of the coordinator's two calls, one tools step
    # for each task: none of the subagents' four.
    names = [step["name"] for step in accumulator.steps]
    assert names.count("model") == 2
    assert names.count("tools") == 2
    assert set(accumulator.subagent_calls.values()) == {"task-1", "task-2"}


# --------------------------------------------------------------------------- #
# The pieces
# --------------------------------------------------------------------------- #


def test_an_announcement_is_read_and_anything_else_is_not():
    run = {SUBAGENT_RUN: {"tool_call_id": "task-1", "checkpoint_ns": "tools:abc"}}

    assert announced_subagent_run(run) == ("tools:abc", "task-1")
    assert announced_subagent_run({"lifecycle": "finalizing"}) is None
    assert announced_subagent_run({SUBAGENT_RUN: {"tool_call_id": "task-1"}}) is None
    assert announced_subagent_run("text") is None


def test_an_event_belongs_to_the_innermost_announced_call():
    accumulator = RunAccumulator(
        subagent_calls={"tools:a": "outer", "tools:a|tools:b": "inner"}
    )

    assert accumulator.parent_tool_call_id(("tools:a",)) == "outer"
    assert accumulator.parent_tool_call_id(("tools:a", "tools:b")) == "inner"
    # A graph nested in one the stream knows of goes with that one.
    assert accumulator.parent_tool_call_id(("tools:a", "tools:c")) == "outer"
    assert accumulator.parent_tool_call_id(("tools:z",)) is None


def test_a_subagents_interrupt_is_not_handed_on():
    """The run's own graph reports the interrupt as well."""
    service = _make_agent_service(stream_subgraphs=True)
    recorder = _Recorder()
    accumulator = RunAccumulator()

    service._process_stream_chunk(
        {"type": "updates", "ns": ("tools:a",), "data": {"__interrupt__": ("x",)}},
        accumulator,
        on_step=recorder.on_step,
        on_token=recorder.on_token,
        on_custom=recorder.on_custom,
        cancel_check=None,
    )

    assert recorder.steps == []
    assert accumulator.steps == []
    assert accumulator.all_messages == []


def test_a_subagents_custom_event_is_handed_on_with_where_it_comes_from():
    service = _make_agent_service(stream_subgraphs=True)
    recorder = _Recorder()
    accumulator = RunAccumulator(subagent_calls={"tools:a": "task-1"})

    service._process_stream_chunk(
        {"type": "custom", "ns": ("tools:a",), "data": {"lifecycle": "finalizing"}},
        accumulator,
        on_step=recorder.on_step,
        on_token=recorder.on_token,
        on_custom=recorder.on_custom,
        cancel_check=None,
    )

    assert recorder.custom == [
        {
            "lifecycle": "finalizing",
            "ns": ("tools:a",),
            "parent_tool_call_id": "task-1",
        }
    ]


@pytest.mark.parametrize("on", [False, True])
def test_the_run_loop_reads_nested_graphs_only_when_asked(on):
    service = _make_agent_service(stream_subgraphs=on)
    agent = MagicMock()
    agent.stream.return_value = [
        {"type": "updates", "ns": (), "data": {"model": {"messages": []}}}
    ]
    service.agent = agent

    with patch(
        "inference_core.services.agents_service.get_settings",
        return_value=MagicMock(
            llm_emulation_enabled=False,
            agent_snapshot_capture_enabled=False,
            agent_snapshot_replay_enabled=False,
            vector_dim=8,
        ),
    ):
        service.run_agent_steps("Hello")

    kwargs = agent.stream.call_args.kwargs
    assert kwargs.get("subgraphs", False) is on

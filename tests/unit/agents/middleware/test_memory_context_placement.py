"""Where the recalled memories go, and what the search by meaning is made with.

``memory_context_placement="message"`` keeps the system prompt the same in
every turn: each turn's memories go into the person's message they were
recalled for, and every later request carries the same bytes there. A
provider's prompt cache and Claude's thinking blocks (bound to the
conversation before them) then stay valid from one turn to the next.

``MemoryWritePolicy.recall_query`` lets an application leave out of the
search what the run must not send to the embedding provider.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from inference_core.agents.middleware.memory import (
    MEMORY_CONTEXT_NOTE,
    MEMORY_CONTEXT_TURNS,
    MemoryMiddleware,
    MemoryState,
)
from inference_core.services.agent_memory_service import (
    MemoryWritePolicy,
    RecalledContext,
)


class _Recorder(BaseChatModel):
    """Answers every call and keeps what each call was sent."""

    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "recorder"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Noted."))]
        )


def _service(texts: list[str], policy: Any = None) -> MagicMock:
    service = MagicMock()
    service.spaces = ()
    service.write_policy = policy
    service.recall_context = AsyncMock(
        side_effect=[RecalledContext(text=t) for t in texts]
    )
    service.recall_memories = AsyncMock(return_value=[])
    return service


def _agent(service, placement: str):
    model = _Recorder(calls=[])
    middleware = MemoryMiddleware(
        memory_service=service,
        user_id="user-1",
        postrun_analysis=False,
        memory_context_placement=placement,
    )
    agent = create_agent(
        model=model,
        system_prompt="You help with office work.",
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    return agent, model


def _dump(messages) -> list[str]:
    """Each message as the bytes of its type and content."""
    return [json.dumps([m.type, m.content], sort_keys=True) for m in messages]


def _two_turns(placement: str):
    service = _service(["- Prefers short answers.", "- Lives in Krakow."])
    agent, model = _agent(service, placement)
    config = {"configurable": {"thread_id": "t1"}}
    agent.invoke({"messages": [HumanMessage("Hello")]}, config)
    agent.invoke({"messages": [HumanMessage("What is the weather?")]}, config)
    return agent, model, config


def test_by_default_the_memories_rebuild_the_system_prompt_each_turn():
    _, model, _ = _two_turns("system_prompt")

    first, second = model.calls
    assert first[0].type == "system" and second[0].type == "system"
    assert "Prefers short answers" in str(first[0].content)
    assert "Lives in Krakow" in str(second[0].content)
    assert first[0].content != second[0].content
    # The person's messages go as they were written.
    assert first[1].content == "Hello"


def test_in_the_message_the_system_prompt_stays_the_same():
    _, model, _ = _two_turns("message")

    first, second = model.calls
    assert first[0].content == second[0].content
    assert "Prefers short answers" not in str(first[0].content)


def test_in_the_message_each_turn_carries_its_own_memories():
    _, model, _ = _two_turns("message")

    first, second = model.calls
    turn_one = first[1]
    assert isinstance(turn_one, HumanMessage)
    memory, text = turn_one.content
    assert memory["text"].startswith("<user_memory_context>\n")
    assert MEMORY_CONTEXT_NOTE in memory["text"]
    assert "Prefers short answers" in memory["text"]
    assert text == {"type": "text", "text": "Hello"}
    # The second turn's memories are in the second turn's message only.
    assert "Lives in Krakow" in second[3].content[0]["text"]
    assert "Lives in Krakow" not in json.dumps(second[1].content)


def test_in_the_message_the_second_request_starts_with_the_first():
    """Byte for byte: the conversation sent only grows."""
    _, model, _ = _two_turns("message")

    first, second = model.calls
    assert _dump(second[: len(first)]) == _dump(first)


def test_the_thread_keeps_the_memories_by_the_persons_message():
    agent, _, config = _two_turns("message")

    state = agent.get_state(config).values
    turns = state[MEMORY_CONTEXT_TURNS]
    humans = [m for m in state["messages"] if m.type == "human"]
    assert turns == {
        humans[0].id: "- Prefers short answers.",
        humans[1].id: "- Lives in Krakow.",
    }
    # A dict, never a bare str: the checkpointer stores a dict through its
    # serde (encrypted with an encrypting one), a str inline in its JSON.
    assert isinstance(turns, dict)
    # The messages of the thread are as the person wrote them.
    assert humans[0].content == "Hello"


def test_the_memories_are_not_part_of_the_agents_output():
    service = _service(["- Prefers short answers."])
    agent, _ = _agent(service, "message")

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]}, {"configurable": {"thread_id": "t"}}
    )

    assert MEMORY_CONTEXT_TURNS not in result


def test_a_turn_without_memories_leaves_the_message_alone():
    service = _service([""])
    agent, model = _agent(service, "message")

    agent.invoke(
        {"messages": [HumanMessage("Hello")]}, {"configurable": {"thread_id": "t"}}
    )

    assert model.calls[0][1].content == "Hello"


def test_turns_of_messages_no_longer_in_the_thread_are_dropped():
    middleware = MemoryMiddleware(
        memory_service=_service([]), memory_context_placement="message"
    )
    kept = HumanMessage("Now", id="h2")
    state = MemoryState(
        messages=[kept], **{MEMORY_CONTEXT_TURNS: {"h1": "old", "h2": "before"}}
    )

    update = middleware._with_turns_update(state, "new", {})

    assert update[MEMORY_CONTEXT_TURNS] == {"h2": "new"}


def test_by_default_no_turns_are_kept():
    middleware = MemoryMiddleware(memory_service=_service([]))
    state = MemoryState(messages=[HumanMessage("Now", id="h1")])

    assert middleware._with_turns_update(state, "new", {"x": 1}) == {"x": 1}


def test_a_subagents_copy_keeps_the_placement():
    middleware = MemoryMiddleware(
        memory_service=_service([]), memory_context_placement="message"
    )

    assert middleware.copy_for_subagent().memory_context_placement == "message"


def test_an_unknown_placement_is_refused():
    with pytest.raises(ValueError):
        MemoryMiddleware(memory_service=_service([]), memory_context_placement="tail")


# ----------------------------------------------------------------------------
# What the search by meaning is made with (MemoryWritePolicy.recall_query)
# ----------------------------------------------------------------------------


class _Policy(MemoryWritePolicy):
    def __init__(self, answer):
        self.answer = answer
        self.asked: list[tuple] = []

    def recall_query(self, text, purpose, messages=()):
        self.asked.append((text, purpose, len(messages)))
        return self.answer(text) if callable(self.answer) else self.answer


def _recall_query_of(policy) -> Any:
    service = _service(["- A note."], policy=policy)
    agent, _ = _agent(service, "system_prompt")
    agent.invoke(
        {"messages": [HumanMessage("Forwarded mail: the invoice is overdue")]},
        {"configurable": {"thread_id": "t"}},
    )
    return service.recall_context.await_args.kwargs["query"]


def test_the_default_policy_searches_with_the_persons_message():
    assert (
        _recall_query_of(MemoryWritePolicy())
        == "Forwarded mail: the invoice is overdue"
    )


def test_a_policy_shortens_the_query():
    policy = _Policy(lambda text: text.split(":")[0])

    assert _recall_query_of(policy) == "Forwarded mail"
    assert policy.asked == [
        ("Forwarded mail: the invoice is overdue", "recall", 1)
    ]


def test_a_policy_can_leave_out_the_search_by_meaning():
    assert _recall_query_of(_Policy(None)) is None


def test_a_policy_that_fails_sends_nothing():
    def fail(text):
        raise RuntimeError("down")

    assert _recall_query_of(_Policy(fail)) is None


async def test_the_analysis_searches_with_the_policys_query():
    policy = _Policy("the persons words")
    service = _service([], policy=policy)
    middleware = MemoryMiddleware(memory_service=service, user_id="u1")
    model = MagicMock()
    model.bind_tools.return_value.ainvoke = AsyncMock(
        return_value=MagicMock(tool_calls=[])
    )
    state = MemoryState(
        messages=[HumanMessage("Forwarded mail: overdue"), AIMessage("Noted.")]
    )

    await middleware._analyse_via_tool_call(state, model, None, "u1", False)

    assert policy.asked[0][1:] == ("analysis", 2)
    assert service.recall_memories.await_args.kwargs["query"] == "the persons words"


async def test_without_a_query_the_analysis_makes_no_search_by_meaning():
    service = _service([], policy=_Policy(None))
    middleware = MemoryMiddleware(memory_service=service, user_id="u1")
    model = MagicMock()
    model.bind_tools.return_value.ainvoke = AsyncMock(
        return_value=MagicMock(tool_calls=[])
    )
    state = MemoryState(
        messages=[HumanMessage("Forwarded mail: overdue"), AIMessage("Noted.")]
    )

    await middleware._analyse_via_tool_call(state, model, None, "u1", False)

    service.recall_memories.assert_not_awaited()


def test_a_summary_that_replaces_the_messages_drops_their_turns():
    """A summary replaces the older messages (``RemoveMessage`` of all of
    them, then the summary and the kept ones): no turn of a message that is
    gone is kept."""
    middleware = MemoryMiddleware(
        memory_service=_service([]), memory_context_placement="message"
    )
    summary = HumanMessage("Summary of the conversation", id="s1")
    latest = HumanMessage("Latest", id="h9")
    state = MemoryState(
        messages=[summary, latest], **{MEMORY_CONTEXT_TURNS: {"h1": "a", "h2": "b"}}
    )

    update = middleware._with_turns_update(state, "c", {})

    assert update[MEMORY_CONTEXT_TURNS] == {"h9": "c"}

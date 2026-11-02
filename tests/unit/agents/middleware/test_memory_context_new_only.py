"""``memory_context_new_only``: a turn's memories leave out those an earlier
turn of the thread shows.

With ``memory_context_placement="message"`` each turn's recalled memories
stay before its message in every later request. Without the option a turn
shows all it recalled, so the memories that suit the conversation are sent
once for every turn. With it a turn shows only the memories no earlier turn
still in the thread shows in the same version (``memory_context_turn_entries``
keeps which, by the entry's key in the store and the time it was last
written).
"""

from __future__ import annotations

from typing import Any, List

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.store.memory import InMemoryStore

from inference_core.agents.middleware.memory import (
    MEMORY_CONTEXT_TURN_ENTRIES,
    MEMORY_CONTEXT_TURNS,
    MemoryMiddleware,
)
from inference_core.services.agent_memory_service import (
    AgentMemoryStoreService,
    MemoryStoreDocument,
)

_TOPICS = ("tea", "krakow", "python", "report", "invoice", "holiday")


def _embed(texts: List[str]) -> List[List[float]]:
    """One dimension per topic word, and a constant one."""
    return [
        [1.0 if topic in text.lower() else 0.0 for topic in _TOPICS] + [0.1]
        for text in texts
    ]


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


class _Setup:
    def __init__(self, new_only: bool, placement: str = "message", extra=()):
        self.store = InMemoryStore(
            index={"embed": _embed, "dims": len(_TOPICS) + 1, "fields": ["content"]}
        )
        self.service = AgentMemoryStoreService(
            store=self.store, agent_name="testbot", scope="dedicated"
        )
        self.model = _Recorder(calls=[])
        self.middleware = MemoryMiddleware(
            memory_service=self.service,
            user_id="u1",
            postrun_analysis=False,
            memory_context_placement=placement,
            memory_context_new_only=new_only,
        )
        self.agent = create_agent(
            model=self.model,
            system_prompt="You help with office work.",
            middleware=[self.middleware, *extra],
            checkpointer=InMemorySaver(),
        )
        self.config = {"configurable": {"thread_id": "t1"}}

    def save(self, content: str, memory_type: str = "facts") -> str:
        from inference_core.celery.async_utils import run_async_safely

        return run_async_safely(self.service.save_memory("u1", content, memory_type))

    def update(self, memory_id: str, content: str) -> None:
        from inference_core.celery.async_utils import run_async_safely

        run_async_safely(self.service.update_memory("u1", memory_id, content))

    def turn(self, text: str, **message: Any) -> None:
        self.agent.invoke(
            {"messages": [HumanMessage(text, **message)]}, config=self.config
        )

    def state(self) -> dict:
        return self.agent.get_state(self.config).values

    def humans(self) -> list:
        return [m for m in self.state()["messages"] if m.type == "human"]

    def block_of(self, message_id: str) -> str:
        return self.state().get(MEMORY_CONTEXT_TURNS, {}).get(message_id, "")


def _memory_blocks(request: list) -> list[str]:
    """The memory blocks of a request, in order."""
    blocks = []
    for message in request:
        if not isinstance(message, HumanMessage) or isinstance(message.content, str):
            continue
        for block in message.content:
            if isinstance(block, dict) and "<user_memory_context>" in block.get(
                "text", ""
            ):
                blocks.append(block["text"])
    return blocks


def test_by_default_every_turn_shows_all_it_recalled():
    setup = _Setup(new_only=False)
    setup.save("Likes green tea.")
    setup.turn("Do I like tea?")
    setup.turn("Which tea again?")

    first, second = setup.humans()
    assert "green tea" in setup.block_of(first.id)
    assert "green tea" in setup.block_of(second.id)
    assert MEMORY_CONTEXT_TURN_ENTRIES not in setup.state()


def test_off_the_recall_and_the_state_are_as_without_the_option():
    """Off, the service is asked as before (no ``leave_out``) and the state
    has exactly the keys it had."""
    calls = []
    setup = _Setup(new_only=False)
    original = setup.service.recall_context

    async def recording(*args, **kwargs):
        calls.append(kwargs)
        return await original(*args, **kwargs)

    setup.service.recall_context = recording
    setup.save("Likes green tea.")
    setup.turn("Do I like tea?")
    setup.turn("Which tea again?")

    assert calls and all(
        set(kwargs) == {"user_id", "query", "include_types", "include_categories"}
        for kwargs in calls
    )
    # The checkpoint has the channels it had before the option existed.
    saved = setup.agent.checkpointer.get_tuple(setup.config)
    channels = set(saved.checkpoint["channel_values"])
    assert MEMORY_CONTEXT_TURN_ENTRIES not in channels
    assert MEMORY_CONTEXT_TURNS in channels
    assert MEMORY_CONTEXT_TURN_ENTRIES not in setup.state()


def test_with_the_system_prompt_the_option_does_nothing():
    setup = _Setup(new_only=True, placement="system_prompt")
    setup.save("Likes green tea.")
    setup.turn("Do I like tea?")
    setup.turn("Which tea again?")

    for request in setup.model.calls:
        assert "green tea" in request[0].text
    assert MEMORY_CONTEXT_TURN_ENTRIES not in setup.state()
    assert MEMORY_CONTEXT_TURNS not in setup.state()


def test_a_turn_with_the_same_memories_has_no_block():
    setup = _Setup(new_only=True)
    setup.save("Likes green tea.")
    setup.turn("Do I like tea?")
    setup.turn("Which tea again?")

    first, second = setup.humans()
    assert "green tea" in setup.block_of(first.id)
    assert second.id not in setup.state()[MEMORY_CONTEXT_TURNS]
    entries = setup.state()[MEMORY_CONTEXT_TURN_ENTRIES]
    assert list(entries) == [first.id]
    # The second request carries the memory once, in the first turn.
    assert len(_memory_blocks(setup.model.calls[1])) == 1


def test_the_older_turns_are_sent_as_they_were():
    """The second request starts with the first one's bytes: the prompt
    cache stays valid."""
    setup = _Setup(new_only=True)
    setup.save("Likes green tea.")
    setup.turn("Do I like tea?")
    setup.save("Lives in Krakow.")
    setup.turn("Where do I live, Krakow?")

    first_request, second_request = setup.model.calls
    assert second_request[: len(first_request)] == first_request


def test_a_new_memory_is_the_only_one_in_its_turn():
    setup = _Setup(new_only=True)
    setup.save("Likes green tea.")
    setup.turn("Do I like tea?")
    setup.save("Lives in Krakow.")
    setup.turn("Where do I live, Krakow?")

    _, second = setup.humans()
    block = setup.block_of(second.id)
    assert "Krakow" in block
    assert "green tea" not in block
    assert len(setup.state()[MEMORY_CONTEXT_TURN_ENTRIES][second.id]) == 1


def test_a_memory_written_since_is_shown_again():
    setup = _Setup(new_only=True)
    memory_id = setup.save("Likes green tea.")
    setup.turn("Do I like tea?")
    setup.update(memory_id, "Likes black tea now.")
    setup.turn("Which tea again?")

    first, second = setup.humans()
    assert "green tea" in setup.block_of(first.id)
    assert "black tea" in setup.block_of(second.id)


def test_a_turn_run_again_is_shown_what_it_recalled():
    """The same message again (a reply given anew) is not left out against
    itself."""
    setup = _Setup(new_only=True)
    setup.save("Likes green tea.")
    setup.turn("Do I like tea?", id="h1")
    assert "green tea" in setup.block_of("h1")
    setup.turn("Do I like tea?", id="h1")

    assert "green tea" in setup.block_of("h1")
    assert len(_memory_blocks(setup.model.calls[-1])) == 1


def test_a_memory_shown_only_in_a_summarised_turn_comes_back():
    setup = _Setup(new_only=True)
    setup.save("Likes green tea.")
    setup.turn("Do I like tea?")
    setup.turn("Which tea again?")
    # A summary replaces the first turn: its message (and its memories) go.
    kept = setup.humans()[1:]
    setup.agent.update_state(
        setup.config,
        {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                HumanMessage("Summary: we talked about tea.", id="summary"),
                *kept,
            ]
        },
    )
    setup.turn("And tea once more?")

    third = setup.humans()[-1]
    assert "green tea" in setup.block_of(third.id)
    assert set(setup.state()[MEMORY_CONTEXT_TURN_ENTRIES]) == {third.id}


class _SummaryInTheRun(AgentMiddleware):
    """Takes the first turn away before the model call of a later turn, as a
    summary in the same run does."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = False

    def before_model(self, state, runtime):
        if not self.armed:
            return None
        self.armed = False
        messages = list(state["messages"])
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                HumanMessage("Summary: we talked about tea.", id="summary"),
                *messages[2:],
            ]
        }


def test_a_summary_in_the_run_gives_the_turn_all_it_recalled():
    summary = _SummaryInTheRun()
    setup = _Setup(new_only=True, extra=[summary])
    setup.save("Likes green tea.")
    setup.turn("Do I like tea?")
    summary.armed = True
    setup.turn("Which tea again?")

    second = setup.humans()[-1]
    # The request after the summary has the memory, in the second turn.
    blocks = _memory_blocks(setup.model.calls[-1])
    assert len(blocks) == 1 and "green tea" in blocks[0]
    # And keeps it, so the next run sends the same bytes.
    assert "green tea" in setup.block_of(second.id)
    assert set(setup.state()[MEMORY_CONTEXT_TURN_ENTRIES]) == {second.id}

    setup.turn("Thanks.")
    after = setup.model.calls[-1]
    assert after[: len(setup.model.calls[-2])] == setup.model.calls[-2]


def test_a_subagents_copy_keeps_the_option():
    setup = _Setup(new_only=True)
    assert setup.middleware.copy_for_subagent().memory_context_new_only is True


def test_the_entries_are_stored_through_an_encrypting_serde(monkeypatch):
    pytest.importorskip("langgraph.checkpoint.serde.encrypted")
    from cryptography.fernet import Fernet

    from inference_core.services import content_cipher as cc
    from inference_core.services.checkpoint_cipher import build_checkpoint_serializer

    monkeypatch.setenv("CHECKPOINT_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("ENCRYPT_CHECKPOINTS_WRITES", "true")
    cc.reset_content_cipher()
    try:
        serde = build_checkpoint_serializer()
        assert serde is not None
        setup = _Setup(new_only=True)
        saver = InMemorySaver(serde=serde)
        setup.agent = create_agent(
            model=setup.model,
            system_prompt="You help with office work.",
            middleware=[setup.middleware],
            checkpointer=saver,
        )
        setup.save("Likes green tea.")
        setup.turn("Do I like tea?")

        entries = setup.state()[MEMORY_CONTEXT_TURN_ENTRIES]
        (shown,) = entries.values()
        (key,) = shown
        blobs = [
            value
            for (_, _, channel, _), value in saver.blobs.items()
            if channel == MEMORY_CONTEXT_TURN_ENTRIES
        ]
        assert blobs
        for kind, data in blobs:
            assert "fernet" in kind
            assert key.encode() not in data
    finally:
        cc.reset_content_cipher()


# -- The service ---------------------------------------------------------------


def _doc(**kwargs: Any) -> MemoryStoreDocument:
    base = dict(
        id="m1", value={"content": "Likes tea.", "memory_type": "facts"}, metadata={}
    )
    base.update(kwargs)
    return MemoryStoreDocument(**base)


def test_an_entry_is_known_by_its_namespace_and_key():
    assert _doc(namespace=("memories", "u1", "semantic")).entry_key == (
        "memories/u1/semantic/m1"
    )


def test_an_entrys_version_is_when_it_was_last_written():
    from datetime import datetime, timezone

    at = datetime(2026, 11, 2, 10, 0, tzinfo=timezone.utc)
    assert _doc(updated_at=at).entry_version == at.isoformat()


def test_without_a_write_time_the_version_follows_the_text():
    one = _doc().entry_version
    assert one.startswith("sha256:")
    assert _doc().entry_version == one
    other = _doc(value={"content": "Likes coffee.", "memory_type": "facts"})
    assert other.entry_version != one


@pytest.mark.asyncio
async def test_the_recall_leaves_out_what_was_shown_and_nothing_takes_its_place():
    setup = _Setup(new_only=True)
    await setup.service.save_memory("u1", "Likes green tea.", "facts")
    await setup.service.save_memory("u1", "Lives in Krakow.", "facts")
    everything = await setup.service.recall_context("u1", query="tea krakow")
    tea = next(d for d in everything.documents if "tea" in d.content)

    recalled = await setup.service.recall_context(
        "u1", query="tea krakow", leave_out={tea.entry_key: tea.entry_version}
    )

    assert [d.content for d in recalled.documents] == ["Lives in Krakow."]
    assert [d.entry_key for d in recalled.left_out] == [tea.entry_key]
    assert "green tea" not in recalled.text and "Krakow" in recalled.text
    assert "green tea" in recalled.full_text and "Krakow" in recalled.full_text


@pytest.mark.asyncio
async def test_another_version_is_not_left_out():
    setup = _Setup(new_only=True)
    await setup.service.save_memory("u1", "Likes green tea.", "facts")
    (tea,) = (await setup.service.recall_context("u1", query="tea")).documents

    recalled = await setup.service.recall_context(
        "u1", query="tea", leave_out={tea.entry_key: "an older version"}
    )

    assert [d.entry_key for d in recalled.documents] == [tea.entry_key]
    assert recalled.left_out == () and recalled.full_text == ""


@pytest.mark.asyncio
async def test_all_left_out_gives_no_text():
    setup = _Setup(new_only=True)
    await setup.service.save_memory("u1", "Likes green tea.", "facts")
    (tea,) = (await setup.service.recall_context("u1", query="tea")).documents

    recalled = await setup.service.recall_context(
        "u1", query="tea", leave_out={tea.entry_key: tea.entry_version}
    )

    assert recalled.text == "" and recalled.documents == ()
    assert len(recalled.left_out) == 1 and "green tea" in recalled.full_text


# -- What it saves -------------------------------------------------------------

_MEASURE_TURNS = (
    "What tea do I like?",
    "And in Krakow, which tea shop?",
    "Help me with the Python report.",
    "The report again, tea break first.",
    "Is the invoice for the holiday paid?",
)


def _measured(new_only: bool) -> list[tuple[int, int]]:
    """For each turn: the characters of its own block, and of all the blocks
    its last request carries."""
    setup = _Setup(new_only=new_only)
    kinds = ("facts", "preferences", "context", "instructions", "goals")
    for n in range(50):
        topic = _TOPICS[n % len(_TOPICS)]
        setup.save(
            f"Note {n} about {topic}: " + "a detail worth keeping. " * 8,
            kinds[n % len(kinds)],
        )
    sizes = []
    for text in _MEASURE_TURNS:
        setup.turn(text)
        blocks = _memory_blocks(setup.model.calls[-1])
        own = setup.block_of(setup.humans()[-1].id)
        sizes.append((len(own), sum(len(b) for b in blocks)))
    return sizes


def test_each_memory_is_sent_once_and_the_requests_grow_less():
    before = _measured(new_only=False)
    after = _measured(new_only=True)

    print(
        "\nturn | own block before / after | all blocks before / after (tokens ~ chars/4)"
    )
    for turn, ((own_b, all_b), (own_a, all_a)) in enumerate(zip(before, after), 1):
        print(f"{turn} | {own_b // 4} / {own_a // 4} | {all_b // 4} / {all_a // 4}")
    assert before[0] == after[0]
    assert all(a[1] <= b[1] for a, b in zip(after, before))
    assert after[-1][1] < before[-1][1]

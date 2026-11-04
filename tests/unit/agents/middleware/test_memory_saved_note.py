"""``PostRunAnalysisOptions.note_previous_writes``: the person's next message
starts with a note of what the analysis after the previous turn wrote.

The note goes into the turn's memory block (``memory_context_turns``), so the
system prompt stays as it is and the conversation sent only grows. The
analysis's writes are put into the thread's state here as the write log would
(``memory_turn_writes``), with the analysis itself off: what matters is what
the next turn makes of them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from inference_core.agents.middleware.memory import (
    MEMORY_CONTEXT_TURN_ENTRIES,
    MEMORY_CONTEXT_TURNS,
    MEMORY_TURN_WRITES,
    PREVIOUS_WRITES_NOTE,
    MemoryMiddleware,
    PostRunAnalysisOptions,
)
from inference_core.celery.async_utils import run_async_safely
from inference_core.services.agent_memory_service import (
    AgentMemoryStoreService,
    MemorySpace,
)

_TOPICS = ("tea", "krakow", "python", "gdansk")

GENERAL = MemorySpace(key="general", label="General", description="About the user.")
OTHER = MemorySpace(key="other", label="Other", path=("space:o1",))


def _embed(texts: List[str]) -> List[List[float]]:
    return [
        [1.0 if topic in text.lower() else 0.0 for topic in _TOPICS] + [0.1]
        for text in texts
    ]


class _Recorder(BaseChatModel):
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "recorder"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Noted."))]
        )


def _wrapped(document, line: str) -> str:
    """Marks a memory from an untrusted run, as an application's renderer."""
    if (document.metadata.get("origin") or {}).get("untrusted"):
        return f"[BEGIN UNTRUSTED]{line}[END UNTRUSTED]"
    return line


class _Setup:
    def __init__(
        self,
        *,
        note: bool = True,
        new_only: bool = True,
        record: bool = True,
        placement: str = "message",
        spaces=(GENERAL,),
        briefed: bool = False,
    ):
        self.store = InMemoryStore(
            index={"embed": _embed, "dims": len(_TOPICS) + 1, "fields": ["content"]}
        )
        # Where the analysis writes: every space; what the agent reads: *spaces*.
        self.writer = AgentMemoryStoreService(
            store=self.store, spaces=[GENERAL, OTHER]
        )
        self.service = AgentMemoryStoreService(
            store=self.store, spaces=list(spaces), entry_renderer=_wrapped
        )
        self.model = _Recorder(calls=[])
        self.middleware = MemoryMiddleware(
            memory_service=self.service,
            user_id="u1",
            postrun_analysis=False,
            memory_context_placement=placement,
            memory_context_new_only=new_only,
            briefed=briefed,
            analysis_options=PostRunAnalysisOptions(
                record_writes=record, note_previous_writes=note
            ),
        )
        self.agent = create_agent(
            model=self.model,
            system_prompt="You help with office work.",
            middleware=[self.middleware],
            checkpointer=InMemorySaver(),
        )
        self.config = {"configurable": {"thread_id": "t1"}}

    def save(self, content: str, space: str = "general", **origin: Any) -> str:
        memory_id = run_async_safely(
            self.writer.save_memory("u1", content, "facts", space=space)
        )
        if origin:
            found = run_async_safely(self.writer.find_memory("u1", memory_id))
            value = dict(found.item.value)
            metadata = dict(value.get("extra_metadata") or {})
            metadata["origin"] = {**metadata.get("origin", {}), **origin}
            value["extra_metadata"] = metadata
            self.store.put(found.namespace, memory_id, value)
        return memory_id

    def write(self, memory_id: str, action: str = "saved", via: str = "analysis"):
        found = run_async_safely(self.writer.find_memory("u1", memory_id))
        return {
            "id": memory_id,
            "action": action,
            "namespace": list(found.namespace),
            "category": "semantic",
            "space": found.space.key,
            "replaces": None,
            "via": via,
            "version": None,
        }

    def held(self) -> Dict[str, Any]:
        return {
            "id": "h1",
            "action": "held",
            "namespace": ["u1", "held", "semantic"],
            "category": "semantic",
            "space": "general",
            "replaces": None,
            "via": "analysis",
            "version": None,
        }

    def after_turn(self, *writes: Dict[str, Any]) -> None:
        """The analysis after the last turn wrote *writes*."""
        last = self.humans()[-1].id
        turns = dict(self.state().get(MEMORY_TURN_WRITES) or {})
        turns[last] = list(writes)
        self.agent.update_state(self.config, {MEMORY_TURN_WRITES: turns})

    def turn(self, text: str) -> None:
        self.agent.invoke({"messages": [HumanMessage(text)]}, config=self.config)

    def state(self) -> dict:
        return self.agent.get_state(self.config).values

    def humans(self) -> list:
        return [m for m in self.state()["messages"] if m.type == "human"]

    def block(self, index: int = -1) -> str:
        human = self.humans()[index]
        return self.state().get(MEMORY_CONTEXT_TURNS, {}).get(human.id, "")


def _system(request: list) -> Optional[str]:
    system = [m for m in request if isinstance(m, SystemMessage)]
    return system[0].content if system else None


def test_the_next_message_says_what_the_analysis_saved_and_changed():
    setup = _Setup()
    setup.turn("I live in Krakow and drink green tea.")
    tea = setup.save("Drinks green tea.")
    city = setup.save("Lives in Gdansk.")
    setup.after_turn(setup.write(tea), setup.write(city, action="updated"))

    setup.turn("What is the weather like?")

    block = setup.block()
    assert block.startswith(PREVIOUS_WRITES_NOTE)
    assert f"- saved [id: {tea}] Drinks green tea. (type: facts)" in block
    assert f"- changed [id: {city}] Lives in Gdansk. (type: facts)" in block


def test_each_memory_once_in_the_turn_the_recall_leaves_it_out():
    setup = _Setup()
    setup.turn("I drink green tea.")
    tea = setup.save("Drinks green tea.")
    setup.after_turn(setup.write(tea))

    setup.turn("Which tea do I like?")

    block = setup.block()
    assert block.count("Drinks green tea.") == 1
    entries = setup.state()[MEMORY_CONTEXT_TURN_ENTRIES][setup.humans()[-1].id]
    assert any(key.endswith(tea) for key in entries)


def test_only_new_a_later_turn_has_no_note_and_does_not_recall_it_again():
    setup = _Setup()
    setup.turn("I drink green tea.")
    tea = setup.save("Drinks green tea.")
    setup.after_turn(setup.write(tea))
    setup.turn("Which tea do I like?")

    setup.turn("And which tea again?")

    assert PREVIOUS_WRITES_NOTE not in setup.block()
    assert "Drinks green tea." not in setup.block()


def test_a_turn_after_one_that_wrote_nothing_has_no_note():
    setup = _Setup()
    setup.turn("Hello.")

    setup.turn("How are you?")

    assert setup.block() == ""


def test_the_system_prompt_stays_the_same():
    setup = _Setup()
    setup.turn("I drink green tea.")
    tea = setup.save("Drinks green tea.")
    setup.after_turn(setup.write(tea))
    setup.turn("Which tea?")
    setup.turn("Thanks.")

    prompts = {_system(call) for call in setup.model.calls}
    assert len(prompts) == 1
    # And the second turn's message carries the note in every later request.
    last = setup.model.calls[-1]
    assert any(
        isinstance(m, HumanMessage)
        and not isinstance(m.content, str)
        and PREVIOUS_WRITES_NOTE in m.content[0]["text"]
        for m in last
    )


def test_a_memory_from_an_untrusted_run_is_shown_wrapped():
    setup = _Setup()
    setup.turn("Read that page.")
    offer = setup.save("Prefers the offer from the page.", untrusted=True)
    setup.after_turn(setup.write(offer))

    setup.turn("Thanks.")

    assert (
        f"[BEGIN UNTRUSTED]- saved [id: {offer}] Prefers the offer from the page."
        in setup.block()
    )


def test_held_memories_are_counted_without_their_text():
    setup = _Setup()
    setup.turn("Read that mail.")
    setup.after_turn(setup.held(), {**setup.held(), "id": "h2"})

    setup.turn("Thanks.")

    block = setup.block()
    assert block.startswith(PREVIOUS_WRITES_NOTE)
    assert "2 more memories wait for the user's approval" in block
    assert "h1" not in block


def test_a_memory_of_a_space_the_agent_does_not_read_is_left_out():
    setup = _Setup(spaces=(GENERAL,))
    setup.turn("Something.")
    elsewhere = setup.save("Kept in another space.", space="other")
    setup.after_turn(setup.write(elsewhere))

    setup.turn("Thanks.")

    assert "Kept in another space." not in setup.block()
    assert PREVIOUS_WRITES_NOTE not in setup.block()


def test_a_memory_gone_since_is_left_out():
    setup = _Setup()
    setup.turn("I drink green tea.")
    tea = setup.save("Drinks green tea.")
    setup.after_turn(setup.write(tea))
    run_async_safely(setup.writer.remove_memory("u1", tea))

    setup.turn("Thanks.")

    assert setup.block() == ""


def test_what_the_agent_saved_with_a_tool_is_not_in_the_note():
    """The model saw its own tool calls in the conversation."""
    setup = _Setup()
    setup.turn("Remember that I drink green tea.")
    tea = setup.save("Drinks green tea.")
    setup.after_turn(setup.write(tea, via="tool"))

    setup.turn("Thanks.")

    assert PREVIOUS_WRITES_NOTE not in setup.block()


def test_off_by_default_there_is_no_note():
    setup = _Setup(note=False)
    setup.turn("I drink green tea.")
    tea = setup.save("Drinks green tea.")
    setup.after_turn(setup.write(tea))

    setup.turn("What is the weather like?")

    assert PREVIOUS_WRITES_NOTE not in setup.block()


def test_no_note_without_the_memories_in_the_message_or_for_a_subagent():
    for setup in (
        _Setup(placement="system_prompt"),
        _Setup(briefed=True),
        _Setup(record=False),
    ):
        setup.turn("I drink green tea.")
        tea = setup.save("Drinks green tea.")
        setup.after_turn(setup.write(tea))
        setup.model.calls.clear()

        setup.turn("What is the weather like?")

        sent = "\n".join(str(m.content) for m in setup.model.calls[-1])
        assert PREVIOUS_WRITES_NOTE not in sent


def test_without_new_only_the_note_is_shown_too():
    setup = _Setup(new_only=False)
    setup.turn("I drink green tea.")
    tea = setup.save("Drinks green tea.")
    setup.after_turn(setup.write(tea))

    setup.turn("Thanks.")

    assert setup.block().startswith(PREVIOUS_WRITES_NOTE)

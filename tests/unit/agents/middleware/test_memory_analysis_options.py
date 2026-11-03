"""How the post-run analysis reads a conversation (``PostRunAnalysisOptions``).

Each option is off by default, and the analysis is then what it always was:
the whole thread, the same prompt, up to four rounds. The tests run the
analysis on a real memory service over a real store, with a model that
answers with the tool calls a test gives it and records what it is sent.
"""

from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.errors import GraphInterrupt
from langgraph.store.memory import InMemoryStore

from inference_core.agents.middleware.memory import (
    ALREADY_CHANGED_NOTE,
    CONTRADICTION_RULE,
    LANGUAGE_PREFERENCE_RULE,
    MEMORY_ANALYSIS_THROUGH,
    MEMORY_TURN_WRITES,
    MEMORY_TURN_WRITES_EVENT,
    SINGLE_PASS_RULE,
    MemoryMiddleware,
    MemoryTurnWritesMiddleware,
    PostRunAnalysisOptions,
)
from inference_core.services.agent_memory_service import (
    HOLD,
    REFUSE,
    VIA_ANALYSIS,
    VIA_TOOL,
    AgentMemoryStoreService,
    MemorySpace,
    MemoryWriteDecision,
    MemoryWritePolicy,
    TurnWriteLog,
)

USER = "u1"
GENERAL = MemorySpace(key="general", label="General")


def _embed(texts: List[str]) -> List[List[float]]:
    return [[1.0, float(len(text) % 7) / 7.0] for text in texts]


def _service(policy=None, write_log=None, repeat_threshold=None):
    store = InMemoryStore(index={"embed": _embed, "dims": 2, "fields": ["content"]})
    return AgentMemoryStoreService(
        store=store,
        spaces=[GENERAL],
        write_policy=policy,
        write_log=write_log,
        repeat_threshold=repeat_threshold,
    )


class _Policy(MemoryWritePolicy):
    def __init__(self, action):
        self.action = action

    def decide(self, request):
        return MemoryWriteDecision(self.action, reason="test")


def _call(name: str, **args: Any) -> dict:
    return {"name": name, "args": args, "id": f"c-{name}-{len(args)}"}


def _response(*calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


class _Model:
    """Answers each call with the next of *responses* (then nothing) and
    keeps what it was sent."""

    def __init__(self, *responses: AIMessage) -> None:
        self.responses = list(responses)
        self.sent: List[List[Any]] = []
        self.bound = MagicMock()
        self.bound.ainvoke = AsyncMock(side_effect=self._answer)

    async def _answer(self, messages):
        self.sent.append(list(messages))
        if self.responses:
            return self.responses.pop(0)
        return AIMessage(content="")

    def bind_tools(self, tools):
        return self.bound

    @property
    def prompts(self) -> List[str]:
        return [batch[1]["content"] for batch in self.sent]


def _middleware(service, model, **options) -> MemoryMiddleware:
    return MemoryMiddleware(
        memory_service=service,
        user_id=USER,
        analysis_model=model,
        analysis_options=PostRunAnalysisOptions(**options) if options else None,
    )


def _thread(*texts: str) -> List[Any]:
    """A thread of alternating messages of the person and the assistant,
    with ids m0, m1, …"""
    messages: List[Any] = []
    for index, text in enumerate(texts):
        cls = HumanMessage if index % 2 == 0 else AIMessage
        messages.append(cls(content=text, id=f"m{index}"))
    return messages


async def _after(middleware, state):
    return await middleware.aafter_agent(state, MagicMock())


# --------------------------------------------------------------------------- #
# Off by default: the analysis as it always was
# --------------------------------------------------------------------------- #


async def test_without_options_the_prompt_is_the_template_as_it_was():
    service = _service()
    model = _Model()
    middleware = _middleware(service, model)
    state = {"messages": _thread("I like tea", "Noted", "And I live in Gdańsk", "OK")}

    await middleware._analyse_via_tool_call(state, model, None, USER, False)

    expected = MemoryMiddleware._TOOL_ANALYSIS_PROMPT.format(
        spaces_section="",
        existing_memories_section="",
        already_saved_note="",
        transcript=(
            "User: I like tea\n\nAssistant: Noted\n\n"
            "User: And I live in Gdańsk\n\nAssistant: OK"
        ),
    )
    assert model.prompts == [expected]


async def test_without_options_the_analysis_goes_up_to_four_rounds_and_keeps_no_keys():
    service = _service()
    model = _Model(
        *[_response(_call("save_memory_store", content=f"fact {i}")) for i in range(6)]
    )
    middleware = _middleware(service, model)
    state = {"messages": _thread("I like tea", "Noted")}

    update = await _after(middleware, state)

    assert len(model.sent) == 4
    assert update["session_analysis_saved"] is True
    assert MEMORY_TURN_WRITES not in update
    assert MEMORY_ANALYSIS_THROUGH not in update
    assert service.write_log is None


def test_the_rules_are_not_in_the_prompt_without_their_options():
    middleware = _middleware(_service(), _Model())
    prompt = middleware._analysis_prompt(
        spaces_section="",
        existing_memories_section="",
        already_saved_note="",
        transcript="User: ahoj",
    )
    assert "SAME language the user spoke" in prompt
    assert "User preferences (language," in prompt
    for rule in (LANGUAGE_PREFERENCE_RULE, CONTRADICTION_RULE, SINGLE_PASS_RULE):
        assert rule not in prompt


# --------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------- #


def _prompt(**options) -> str:
    middleware = _middleware(_service(), _Model(), **options)
    return middleware._analysis_prompt(
        spaces_section="",
        existing_memories_section="",
        already_saved_note="",
        transcript="User: ahoj!",
    )


def test_the_memories_are_written_in_the_language_given():
    prompt = _prompt(language="Polish")
    assert "SAME language" not in prompt
    assert "in Polish, whatever language the conversation is in" in prompt


def test_a_language_with_braces_does_not_break_the_template():
    assert "in Pol{ish}, whatever" in _prompt(language="Pol{ish}")


def test_a_preference_for_a_language_needs_an_explicit_request():
    prompt = _prompt(explicit_language_preference=True)
    assert LANGUAGE_PREFERENCE_RULE in prompt
    assert "User preferences (tone," in prompt
    assert "User preferences (language," not in prompt


def test_a_contradiction_changes_the_memory_it_contradicts():
    prompt = _prompt(contradictions_replace=True)
    assert CONTRADICTION_RULE in prompt
    assert prompt.index("4. Only call") < prompt.index(CONTRADICTION_RULE)


def test_one_pass_tells_the_model_one_topic_one_memory():
    prompt = _prompt(single_pass=True)
    assert SINGLE_PASS_RULE in prompt
    assert "You may call the tools multiple times" not in prompt


# --------------------------------------------------------------------------- #
# Only the new part of the conversation
# --------------------------------------------------------------------------- #


async def test_the_first_analysis_reads_the_whole_thread_and_marks_where_it_stopped():
    service = _service()
    model = _Model()
    middleware = _middleware(service, model, new_messages_only=True)
    state = {"messages": _thread("ahoj!", "Ahoj!")}

    update = await _after(middleware, state)

    assert "User: ahoj!" in model.prompts[0]
    assert "EARLIER IN THE CONVERSATION" not in model.prompts[0]
    assert update == {MEMORY_ANALYSIS_THROUGH: "m1"}


async def test_a_later_analysis_reads_the_new_part_with_the_earlier_as_context():
    service = _service()
    model = _Model()
    middleware = _middleware(
        service, model, new_messages_only=True, background_chars=40
    )
    messages = _thread(
        "ahoj! " + "x" * 200, "Ahoj!", "jam żem polak, nie czech :)", "Rozumiem"
    )
    state = {"messages": messages, MEMORY_ANALYSIS_THROUGH: "m1"}

    update = await _after(middleware, state)

    prompt = model.prompts[0]
    new_part = prompt.split("New part of the conversation (analyse this part only):")[1]
    assert "User: jam żem polak, nie czech :)" in new_part
    assert "ahoj" not in new_part
    background = prompt.split("EARLIER IN THE CONVERSATION")[1].split("New part")[0]
    assert "Assistant: Ahoj!" in background
    # Cut to the last 40 characters.
    assert "ahoj!" not in background
    assert update == {MEMORY_ANALYSIS_THROUGH: "m3"}


async def test_a_mark_a_summary_took_away_leaves_the_turn():
    model = _Model()
    middleware = _middleware(_service(), model, new_messages_only=True)
    messages = _thread("summary of before", "ok", "mam na imię Tomasz", "Miło mi")
    state = {"messages": messages, MEMORY_ANALYSIS_THROUGH: "gone"}

    await _after(middleware, state)

    new_part = model.prompts[0].split("New part of the conversation")[1]
    assert "mam na imię Tomasz" in new_part
    assert "summary of before" not in new_part


async def test_text_in_content_blocks_is_read_with_the_new_part():
    model = _Model()
    middleware = _middleware(_service(), model, new_messages_only=True)
    messages = [
        HumanMessage(content="I am a pilot", id="m0"),
        AIMessage(
            content=[
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": "Nice to meet a pilot"},
            ],
            id="m1",
        ),
    ]

    await _after(middleware, {"messages": messages})

    assert "Assistant: Nice to meet a pilot" in model.prompts[0]
    assert "hidden" not in model.prompts[0]


async def test_a_save_made_earlier_in_the_thread_is_not_a_save_of_this_run():
    middleware = _middleware(_service(), _Model(), new_messages_only=True)
    earlier = AIMessage(
        content="", id="m1", tool_calls=[_call("save_memory_store", content="x")]
    )
    messages = [HumanMessage(content="a", id="m0"), earlier, *_thread("b", "c")[0:0]]
    messages += [HumanMessage(content="b", id="m2"), AIMessage(content="c", id="m3")]
    state = {"messages": messages, MEMORY_ANALYSIS_THROUGH: "m1"}

    assert middleware._has_memory_save_in_run(state) is False
    assert middleware._has_memory_save_in_run({"messages": messages}) is True


async def test_the_search_for_similar_memories_reads_the_new_part():
    service = _service()
    service.recall_memories = AsyncMock(return_value=[])
    middleware = _middleware(service, _Model(), new_messages_only=True)
    messages = _thread("old fact about cats", "ok", "new fact about dogs", "ok")

    await _after(middleware, {"messages": messages, MEMORY_ANALYSIS_THROUGH: "m1"})

    query = service.recall_memories.await_args.kwargs["query"]
    assert "dogs" in query and "cats" not in query


# --------------------------------------------------------------------------- #
# One pass
# --------------------------------------------------------------------------- #


async def test_one_pass_ends_after_an_answer_that_only_writes():
    service = _service()
    model = _Model(
        _response(_call("save_memory_store", content="User's name is Tomasz")),
        _response(_call("save_memory_store", content="User is called Tomasz")),
    )
    middleware = _middleware(service, model, single_pass=True)

    await _after(middleware, {"messages": _thread("I'm Tomasz", "Hi Tomasz")})

    assert len(model.sent) == 1


async def test_one_pass_goes_on_after_a_recall():
    model = _Model(
        _response(_call("recall_memories_store", query="name")),
        _response(_call("save_memory_store", content="User's name is Tomasz")),
        _response(_call("save_memory_store", content="again")),
    )
    middleware = _middleware(_service(), model, single_pass=True)

    await _after(middleware, {"messages": _thread("I'm Tomasz", "Hi")})

    assert len(model.sent) == 2


async def test_one_pass_has_three_rounds_at_most():
    model = _Model(*[_response(_call("recall_memories_store", query="q")) for _ in range(5)])
    middleware = _middleware(_service(), model, single_pass=True)

    await _after(middleware, {"messages": _thread("a", "b")})

    assert len(model.sent) == 3


async def test_one_pass_makes_one_change_of_a_memory():
    service = _service()
    saved = await service.write_memory(USER, "User prefers Czech", space="general")
    model = _Model(
        _response(
            _call("update_memory_store", memory_id=saved.memory_id, content="Polish"),
            _call("update_memory_store", memory_id=saved.memory_id, content="Both"),
            _call("recall_memories_store", query="language"),
        ),
    )
    middleware = _middleware(service, model, single_pass=True)

    await _after(middleware, {"messages": _thread("I am Polish", "OK")})

    found = await service.find_memory(USER, saved.memory_id)
    assert found.item.value["content"] == "Polish"
    second = model.sent[1][-2]
    assert second.content == ALREADY_CHANGED_NOTE.format(memory_id=saved.memory_id)


# --------------------------------------------------------------------------- #
# The turn's writes
# --------------------------------------------------------------------------- #


async def test_the_writes_are_kept_under_the_persons_message_without_text():
    service = _service()
    existing = await service.write_memory(USER, "User prefers Czech", space="general")
    model = _Model(
        _response(
            _call("save_memory_store", content="User's name is Tomasz"),
            _call(
                "update_memory_store",
                memory_id=existing.memory_id,
                content="User is Polish",
            ),
        )
    )
    middleware = _middleware(
        service, model, record_writes=True, single_pass=True
    )
    assert isinstance(service.write_log, TurnWriteLog)

    update = await _after(middleware, {"messages": _thread("I'm Tomasz, a Pole", "Hi")})

    writes = update[MEMORY_TURN_WRITES]["m0"]
    assert [w["action"] for w in writes] == ["saved", "updated"]
    saved, updated = writes
    assert updated["id"] == existing.memory_id
    for write in writes:
        assert write["namespace"] == [USER, "semantic"]
        assert write["category"] == "semantic"
        assert write["space"] == "general"
        assert write["via"] == VIA_ANALYSIS
        assert write["replaces"] is None
        assert write["version"]
        assert "Tomasz" not in repr(write) and "Polish" not in repr(write)
    item = service.store.get(tuple(saved["namespace"]), saved["id"])
    assert item.value["content"] == "User's name is Tomasz"
    assert item.updated_at.isoformat() == saved["version"]


async def test_a_held_memory_and_a_held_change_are_listed_by_their_held_keys():
    service = _service()
    existing = await service.write_memory(USER, "Prefers tea", space="general")
    service.write_policy = _Policy(HOLD)
    model = _Model(
        _response(
            _call("save_memory_store", content="Likes coffee"),
            _call(
                "update_memory_store",
                memory_id=existing.memory_id,
                content="Prefers coffee",
            ),
        )
    )
    middleware = _middleware(service, model, record_writes=True, single_pass=True)

    update = await _after(middleware, {"messages": _thread("coffee now", "OK")})

    new, change = update[MEMORY_TURN_WRITES]["m0"]
    assert new["action"] == change["action"] == "held"
    assert new["namespace"] == change["namespace"] == [USER, "held", "semantic"]
    assert new["replaces"] is None
    assert change["replaces"] == existing.memory_id
    assert change["id"] != existing.memory_id
    assert service.store.get(tuple(change["namespace"]), change["id"]) is not None
    assert new["space"] == "general"


async def test_writes_that_wrote_nothing_are_not_listed():
    service = _service(repeat_threshold=0.9)
    await service.write_memory(USER, "User's name is Tomasz", space="general")
    model = _Model(
        _response(
            _call("save_memory_store", content="User's name is Tomasz"),
            _call("update_memory_store", memory_id="nope", content="x"),
        )
    )
    middleware = _middleware(service, model, record_writes=True, single_pass=True)

    update = await _after(middleware, {"messages": _thread("I'm Tomasz", "Hi")})

    assert update is None or MEMORY_TURN_WRITES not in update


async def test_a_refused_write_is_not_listed():
    service = _service(policy=_Policy(REFUSE))
    model = _Model(_response(_call("save_memory_store", content="x")))
    middleware = _middleware(service, model, record_writes=True, single_pass=True)

    update = await _after(middleware, {"messages": _thread("a", "b")})

    assert MEMORY_TURN_WRITES not in (update or {})


async def test_the_tools_writes_in_the_run_join_the_analysis_writes():
    service = _service()
    model = _Model(_response(_call("save_memory_store", content="Lives in Gdańsk")))
    middleware = _middleware(service, model, record_writes=True, single_pass=True)
    keeper = middleware.turn_writes_keeper()
    assert isinstance(keeper, MemoryTurnWritesMiddleware)
    messages = _thread("remember: I like tea; I live in Gdańsk", "Saved")

    # The assistant's own tool saves during the run, then a model call.
    await service.write_memory(USER, "Likes tea", space="general", via=VIA_TOOL)
    kept = await keeper.abefore_model({"messages": messages}, MagicMock())
    assert [w["via"] for w in kept[MEMORY_TURN_WRITES]["m0"]] == [VIA_TOOL]

    update = await _after(middleware, {"messages": messages, **kept})

    assert [w["via"] for w in update[MEMORY_TURN_WRITES]["m0"]] == [
        VIA_TOOL,
        VIA_ANALYSIS,
    ]


async def test_a_memory_saved_and_changed_in_one_turn_stays_saved_once():
    service = _service()
    middleware = _middleware(service, _Model(), record_writes=True)
    messages = _thread("a", "b")
    saved = await service.write_memory(USER, "one", space="general")
    await service.update_memory(USER, saved.memory_id, "two")

    update = middleware._turn_writes_update({"messages": messages})

    (write,) = update[MEMORY_TURN_WRITES]["m0"]
    assert write["action"] == "saved"


async def test_a_turn_without_writes_shows_none_and_gone_turns_are_dropped():
    service = _service()
    middleware = _middleware(service, _Model(), record_writes=True)
    earlier = {"gone": [{"id": "x"}], "m0": [{"id": "y", "namespace": []}]}
    messages = _thread("a", "b", "c", "d")

    update = middleware._turn_writes_update(
        {"messages": messages, MEMORY_TURN_WRITES: earlier}
    )

    assert update == {MEMORY_TURN_WRITES: {"m0": [{"id": "y", "namespace": []}]}}
    assert "m2" not in update[MEMORY_TURN_WRITES]


async def test_a_trashed_memory_is_listed_with_where_it_lay():
    log = TurnWriteLog()
    service = _service(write_log=log)
    service.store.put((USER, "trash", "semantic"), "k1", {"content": "x"})

    service.note_trashed(
        USER,
        "k1",
        (USER, "trash", "semantic"),
        "semantic",
        "general",
        (USER, "semantic"),
    )

    (entry,) = log.drain()
    assert entry["action"] == "trashed"
    assert entry["trashed_from"] == {"namespace": [USER, "semantic"], "key": "k1"}
    assert entry["version"]


async def test_the_turns_writes_go_out_as_a_stream_event(monkeypatch):
    sent = []
    monkeypatch.setattr("langgraph.config.get_stream_writer", lambda: sent.append)
    service = _service()
    model = _Model(_response(_call("save_memory_store", content="x")))
    middleware = _middleware(service, model, record_writes=True, single_pass=True)

    await _after(middleware, {"messages": _thread("a", "b")})

    events = [event for event in sent if MEMORY_TURN_WRITES_EVENT in event]
    assert len(events) == 1
    assert events[0][MEMORY_TURN_WRITES_EVENT]["turn"] == "m0"
    assert events[0][MEMORY_TURN_WRITES_EVENT]["writes"][0]["action"] == "saved"


def test_a_subagents_copy_keeps_nothing_in_its_thread():
    service = _service()
    middleware = _middleware(service, _Model(), record_writes=True)
    copy = middleware.copy_for_subagent()
    assert copy.analysis_options.record_writes is False
    assert copy.turn_writes_keeper() is None
    assert copy.memory_service.write_log is service.write_log


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


async def test_a_gate_that_says_no_leaves_the_analysis_out_and_moves_the_mark():
    asked = []

    async def gate(text):
        asked.append(text)
        return False

    model = _Model()
    middleware = _middleware(_service(), model, gate=gate, new_messages_only=True)
    messages = _thread("old", "ok", "cześć!", "Cześć!")

    update = await _after(middleware, {"messages": messages, MEMORY_ANALYSIS_THROUGH: "m1"})

    assert model.sent == []
    assert asked == ["cześć!"]
    assert update == {MEMORY_ANALYSIS_THROUGH: "m3"}


@pytest.mark.parametrize("verdict", [True, None])
async def test_a_gate_that_says_yes_or_nothing_runs_the_analysis(verdict):
    model = _Model()
    middleware = _middleware(_service(), model, gate=AsyncMock(return_value=verdict))

    await _after(middleware, {"messages": _thread("I'm Tomasz", "Hi")})

    assert len(model.sent) == 1


async def test_a_gate_that_fails_runs_the_analysis():
    model = _Model()
    gate = AsyncMock(side_effect=RuntimeError("down"))
    middleware = _middleware(_service(), model, gate=gate)

    await _after(middleware, {"messages": _thread("I'm Tomasz", "Hi")})

    assert len(model.sent) == 1


async def test_a_run_stopped_in_the_gate_runs_no_analysis_and_keeps_the_mark():
    model = _Model()
    gate = AsyncMock(side_effect=GraphInterrupt(()))
    middleware = _middleware(_service(), model, gate=gate, new_messages_only=True)

    update = await _after(middleware, {"messages": _thread("a", "b")})

    assert model.sent == []
    assert update is None


async def test_without_new_messages_only_the_gate_reads_the_last_message():
    gate = AsyncMock(return_value=False)
    middleware = _middleware(_service(), _Model(), gate=gate)

    await _after(middleware, {"messages": _thread("first", "x", "second", "y")})

    gate.assert_awaited_once_with("second")


def test_the_sync_hook_asks_the_gate_too():
    model = _Model()
    gate = AsyncMock(return_value=False)
    middleware = _middleware(_service(), model, gate=gate, new_messages_only=True)

    update = middleware.after_agent({"messages": _thread("hi", "hello")}, MagicMock())

    assert model.sent == []
    assert update == {MEMORY_ANALYSIS_THROUGH: "m1"}

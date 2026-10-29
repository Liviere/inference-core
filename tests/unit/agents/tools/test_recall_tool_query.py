"""The recall tool's query and the write policy.

The ``recall_memories_store`` tool searches memories by meaning: the query a
model wrote becomes an embedding at the embedding provider. The service's
write policy decides what the query may be (``recall_query``, purpose
``tool``); with ``None`` the tool recalls the newest memories, those that hold
the query's words first, and sends nothing to the provider. The analysis after
a run uses the same tool.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

from inference_core.agents.middleware.memory import MemoryMiddleware, MemoryState
from inference_core.agents.tools.memory_tools import (
    NOT_BY_MEANING_NOTE,
    RecallMemoryStoreTool,
)
from inference_core.services.agent_memory_service import (
    AgentMemoryStoreService,
    MemoryWritePolicy,
)

SECRET = "Zefiryna paid invoice FV/0917 from account 61 1090"


class _Embedder:
    """Records every text it is asked to embed."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[float(len(t) % 7), 1.0, 0.5, 0.25] for t in texts]


class _NoSearchByMeaning(MemoryWritePolicy):
    def __init__(self) -> None:
        self.asked: list[tuple[str, str, int]] = []

    def recall_query(
        self, text: Any, purpose: str, messages: Sequence[Any] = ()
    ) -> Optional[Any]:
        self.asked.append((text, purpose, len(messages)))
        return None


class _Failing(MemoryWritePolicy):
    def recall_query(self, text, purpose, messages=()):
        raise RuntimeError("down")


def _service(policy: Optional[MemoryWritePolicy]) -> tuple[Any, _Embedder]:
    embedder = _Embedder()
    store = InMemoryStore(index={"dims": 4, "embed": embedder})
    service = AgentMemoryStoreService(store, write_policy=policy)

    async def seed() -> None:
        for content, memory_type in (
            ("Prefers tea in the morning", "preferences"),
            ("Pays invoices on Fridays", "facts"),
            ("Works from Gdańsk", "facts"),
            ("Faktury wysyła księgowej", "facts"),
        ):
            await service.write_memory("u1", content, memory_type)
            await asyncio.sleep(0.001)

    asyncio.run(seed())
    embedder.texts.clear()
    return service, embedder


def _recall(service: Any, query: str, **kwargs: Any) -> str:
    tool = RecallMemoryStoreTool(memory_service=service, user_id="u1")
    return asyncio.run(tool._arun(query=query, **kwargs))


def test_without_a_policy_the_tool_searches_by_meaning():
    service, embedder = _service(None)

    result = _recall(service, SECRET)

    assert SECRET in embedder.texts
    assert NOT_BY_MEANING_NOTE not in result


def test_the_default_policy_searches_by_meaning_as_before():
    service, embedder = _service(MemoryWritePolicy())

    result = _recall(service, SECRET)

    assert SECRET in embedder.texts
    assert "relevant memories" in result


def test_a_policy_that_keeps_the_query_sends_the_embedder_nothing():
    policy = _NoSearchByMeaning()
    service, embedder = _service(policy)

    result = _recall(service, SECRET)

    assert embedder.texts == []
    assert policy.asked == [(SECRET, "tool", 0)]
    assert result.startswith(NOT_BY_MEANING_NOTE)


def test_the_newest_with_the_querys_words_come_first():
    service, _ = _service(_NoSearchByMeaning())

    result = _recall(service, "which invoices", max_results=3)

    lines = result.splitlines()
    assert "Pays invoices on Fridays" in lines[2]
    # The newest of the rest fill the list.
    assert "Faktury wysyła księgowej" in lines[3]
    assert "Works from Gdańsk" in lines[4]
    assert len(lines) == 5


def test_a_word_in_another_form_is_found():
    service, _ = _service(_NoSearchByMeaning())

    result = _recall(service, "faktura", max_results=1)

    assert "Faktury wysyła księgowej" in result


def test_the_type_filters_without_a_search():
    service, embedder = _service(_NoSearchByMeaning())

    result = _recall(service, "tea", memory_type="preferences")

    assert "Prefers tea in the morning" in result
    assert "Works from Gdańsk" not in result
    assert embedder.texts == []


def test_a_policy_that_fails_sends_the_embedder_nothing():
    service, embedder = _service(_Failing())

    result = _recall(service, SECRET)

    assert embedder.texts == []
    assert result.startswith(NOT_BY_MEANING_NOTE)


def test_the_sync_path_asks_the_policy_too():
    service, embedder = _service(_NoSearchByMeaning())
    tool = RecallMemoryStoreTool(memory_service=service, user_id="u1")

    result = tool._run(query=SECRET)

    assert embedder.texts == []
    assert result.startswith(NOT_BY_MEANING_NOTE)


def test_the_log_holds_no_query(caplog):
    service, _ = _service(MemoryWritePolicy())

    with caplog.at_level("DEBUG"):
        _recall(service, SECRET)
    with caplog.at_level("DEBUG"):
        service2, _ = _service(_NoSearchByMeaning())
        _recall(service2, SECRET)

    assert "Zefiryna" not in caplog.text


def test_the_analysis_after_a_run_asks_the_policy_for_its_recall(monkeypatch):
    """The analysis's own recall_memories_store call goes the same way."""
    service, embedder = _service(_NoSearchByMeaning())
    middleware = MemoryMiddleware(
        memory_service=service, user_id="u1", postrun_analysis=True
    )
    state = MemoryState(
        messages=[HumanMessage("What did the mail say?"), AIMessage(SECRET)]
    )
    recall = MagicMock()
    recall.tool_calls = [
        {"name": "recall_memories_store", "args": {"query": SECRET}, "id": "c1"}
    ]
    done = MagicMock()
    done.tool_calls = []
    model = MagicMock()

    async def answers(messages):
        return recall if len(messages) == 2 else done

    model.bind_tools.return_value.ainvoke = answers

    saved = asyncio.run(
        middleware._analyse_via_tool_call(state, model, None, "u1", False)
    )

    assert saved == 0
    assert (SECRET, "tool", 0) in service.write_policy.asked
    assert not [t for t in embedder.texts if "Zefiryna" in t]


@pytest.mark.parametrize(
    "words, expected",
    [("", ()), ("a an", ()), ("Invoices", ("invoic",)), ("tea", ("tea",))],
)
def test_word_stems(words, expected):
    from inference_core.services.agent_memory_service import _word_stems

    assert _word_stems(words) == expected

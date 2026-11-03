"""A field left without text loses its vector, as an option.

``store.put`` writes the vectors of the indexed fields that give text and
leaves the vector of one that no longer does: a memory whose topic was
cleared was still found by that topic. ``drop_vectors_without_text`` deletes
such vectors; ``edit_user_memory_entry(drop_cleared_vectors=True)`` and
``AgentMemoryStoreService(drop_cleared_field_vectors=True)`` call it after
their put. Without the option nothing changes.
"""

from typing import List

import pytest
from langgraph.store.memory import InMemoryStore

from inference_core.services.agent_memory_service import (
    INDEXED_FIELDS,
    AgentMemoryStoreService,
    MemoryOptions,
    MemorySpace,
    drop_vectors_without_text,
    edit_user_memory_entry,
)

USER = "u1"
NAMESPACE = (USER, "semantic")


def _embed(texts: List[str]) -> List[List[float]]:
    return [[1.0, 0.0] if text == "travel" else [0.0, 1.0] for text in texts]


@pytest.fixture(autouse=True)
def _plain_text(monkeypatch):
    monkeypatch.setenv("ENCRYPT_MEMORY_WRITES", "false")


@pytest.fixture
def store():
    return InMemoryStore(index={"embed": _embed, "dims": 2, "fields": INDEXED_FIELDS})


def _vectors(store, namespace=NAMESPACE, key="m1"):
    return set(store._vectors.get(namespace, {}).get(key, {}))


def _put(store, value, key="m1"):
    store.put(NAMESPACE, key, value, index=INDEXED_FIELDS)


class TestDropVectorsWithoutText:
    def test_a_cleared_topic(self, store):
        _put(store, {"content": "by train", "topic": "travel"})
        value = {"content": "by bus", "topic": None}
        _put(store, value)
        assert _vectors(store) == {"content", "topic"}

        assert drop_vectors_without_text(store, NAMESPACE, "m1", value) == 1

        assert _vectors(store) == {"content"}

    def test_a_field_gone_from_the_value(self, store):
        _put(store, {"content": "by train", "topic": "travel"})
        value = {"content": "by bus"}
        _put(store, value)

        assert drop_vectors_without_text(store, NAMESPACE, "m1", value) == 1
        assert _vectors(store) == {"content"}

    def test_a_field_of_several_texts(self, store):
        fields = ["content", "tags[*]"]
        store.put(
            NAMESPACE,
            "m1",
            {"content": "by train", "tags": ["travel", "rail"]},
            index=fields,
        )
        assert _vectors(store) == {"content", "tags[*].0", "tags[*].1"}
        value = {"content": "by bus", "tags": []}
        store.put(NAMESPACE, "m1", value, index=fields)

        assert drop_vectors_without_text(store, NAMESPACE, "m1", value, fields) == 2
        assert _vectors(store) == {"content"}

    def test_fields_with_text_keep_theirs(self, store):
        value = {"content": "by train", "topic": "travel"}
        _put(store, value)

        assert drop_vectors_without_text(store, NAMESPACE, "m1", value) == 0
        assert _vectors(store) == {"content", "topic"}

    def test_another_memory_keeps_its_vectors(self, store):
        _put(store, {"content": "by train", "topic": "travel"})
        _put(store, {"content": "by car", "topic": "travel"}, key="m2")
        value = {"content": "by bus", "topic": None}
        _put(store, value)

        drop_vectors_without_text(store, NAMESPACE, "m1", value)

        assert _vectors(store, key="m2") == {"content", "topic"}

    def test_another_store_is_left_as_it_is(self):
        class Other:
            pass

        assert drop_vectors_without_text(Other(), NAMESPACE, "m1", {"topic": None}) == 0

    def test_a_failure_leaves_the_vectors(self, store, caplog):
        _put(store, {"content": "by train", "topic": "travel"})
        store._vectors = None  # deleting from it fails

        assert drop_vectors_without_text(store, NAMESPACE, "m1", {"content": "x"}) == 0
        assert "Could not delete the vectors" in caplog.text
        assert "by train" not in caplog.text


class TestEditUserMemoryEntry:
    def test_a_cleared_topic_loses_its_vector_with_the_option(self, store):
        _put(store, {"content": "by train", "topic": "travel"})

        assert edit_user_memory_entry(
            store, USER, NAMESPACE, "m1", "by bus", topic="", drop_cleared_vectors=True
        )

        assert _vectors(store) == {"content"}
        [found] = store.search(NAMESPACE, query="travel", limit=1)
        assert found.score < 0.5

    def test_without_it_the_vector_stays(self, store):
        _put(store, {"content": "by train", "topic": "travel"})

        assert edit_user_memory_entry(store, USER, NAMESPACE, "m1", "by bus", topic="")

        assert _vectors(store) == {"content", "topic"}
        [found] = store.search(NAMESPACE, query="travel", limit=1)
        assert found.score == pytest.approx(1.0)

    def test_a_topic_left_as_it_is_keeps_its_vector(self, store):
        _put(store, {"content": "by train", "topic": "travel"})

        assert edit_user_memory_entry(
            store, USER, NAMESPACE, "m1", "by bus", drop_cleared_vectors=True
        )

        assert _vectors(store) == {"content", "topic"}


GENERAL = MemorySpace(key="general", label="General", description="About the user.")


class TestUpdateMemory:
    @staticmethod
    async def _memory(service):
        return await service.save_memory(USER, "by train", "context", topic="travel")

    @staticmethod
    def _where(store, mid):
        [namespace] = [ns for ns, items in store._vectors.items() if mid in items]
        return namespace

    @pytest.mark.asyncio
    async def test_a_change_without_a_topic_drops_its_vector_with_the_option(
        self, store
    ):
        service = MemoryOptions(
            spaces=(GENERAL,), drop_cleared_field_vectors=True
        ).service(store)
        mid = await self._memory(service)
        namespace = self._where(store, mid)
        assert _vectors(store, namespace, mid) == {"content", "topic"}

        await service.update_memory(USER, mid, "by bus")

        assert _vectors(store, namespace, mid) == {"content"}

    @pytest.mark.asyncio
    async def test_without_it_the_vector_stays(self, store):
        service = AgentMemoryStoreService(store=store, spaces=[GENERAL])
        assert service.drop_cleared_field_vectors is False
        mid = await self._memory(service)
        namespace = self._where(store, mid)

        await service.update_memory(USER, mid, "by bus")

        assert _vectors(store, namespace, mid) == {"content", "topic"}

    @pytest.mark.asyncio
    async def test_a_new_topic_replaces_the_old_one(self, store):
        service = MemoryOptions(
            spaces=(GENERAL,), drop_cleared_field_vectors=True
        ).service(store)
        mid = await self._memory(service)
        namespace = self._where(store, mid)

        await service.update_memory(USER, mid, "by bus", topic="commute")

        assert _vectors(store, namespace, mid) == {"content", "topic"}

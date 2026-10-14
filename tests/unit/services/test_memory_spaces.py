"""An agent's memory is a list of spaces: it reads all of them, writes some.

The store searches by namespace prefix, so a search of ``(user, "semantic")``
also finds what lies under ``(user, "semantic", "agent")``. These tests run on
a real store, since what a search returns is the store's to decide.
"""

from datetime import datetime, timedelta, timezone
from typing import List

import pytest
from langgraph.store.memory import InMemoryStore

from inference_core.services.agent_memory_service import (
    AGENT_SPACE_KEY,
    DELETED,
    NOT_FOUND,
    READ_ONLY,
    SHARED_SPACE_KEY,
    UPDATED,
    AgentMemoryStoreService,
    MemoryCategory,
    MemoryNamespaceBuilder,
    MemorySpace,
    MemorySpaceError,
    spaces_for_scope,
)


def _embed(texts: List[str]) -> List[List[float]]:
    return [[1.0 if "python" in text.lower() else 0.0, 0.5] for text in texts]


class _CountingStore(InMemoryStore):
    batches = 0

    def batch(self, ops):
        type(self).batches += 1
        return super().batch(ops)


@pytest.fixture
def store():
    _CountingStore.batches = 0
    return _CountingStore(index={"embed": _embed, "dims": 2, "fields": ["content"]})


GENERAL = MemorySpace(key="general", label="General", description="About the user.")
PRIVATE = MemorySpace(
    key="private", label="Own", description="Own work.", path=("agent:a1",)
)
TEAM = MemorySpace(key="team", label="Team", description="The team.", path=("space:t1",))


def _service(store, *spaces, default_write=None):
    return AgentMemoryStoreService(
        store=store, spaces=list(spaces), default_write=default_write
    )


# --------------------------------------------------------------------------- #
# Legacy scopes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scope", ["shared", "dedicated", "hybrid", "unknown"])
@pytest.mark.parametrize("agent_name", [None, "writer"])
@pytest.mark.parametrize("category", list(MemoryCategory))
def test_a_legacy_scope_writes_where_it_always_did(
    store, scope, agent_name, category
):
    builder = MemoryNamespaceBuilder(agent_name=agent_name, scope=scope)
    service = AgentMemoryStoreService(store=store, agent_name=agent_name, scope=scope)

    target = service.space_for_write(category)

    assert target.namespace("u1", category) == builder.namespace_for("u1", category)


def test_the_spaces_of_the_hybrid_scope():
    shared, own = spaces_for_scope("hybrid", "writer")

    assert shared.key == SHARED_SPACE_KEY
    assert shared.categories == frozenset({"semantic"})
    assert own.key == AGENT_SPACE_KEY
    assert own.namespace("u1", "episodic") == ("u1", "episodic", "writer")


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_space_does_not_read_what_lies_under_it(store):
    """The shared scope used to read every dedicated agent's memories."""
    shared = AgentMemoryStoreService(store=store, scope="shared")
    dedicated = AgentMemoryStoreService(
        store=store, agent_name="sales", scope="dedicated"
    )
    await dedicated.save_memory("u1", "from the sales mailbox", "facts")
    await shared.save_memory("u1", "the user's name", "facts")

    seen = await shared.recall_memories("u1", "anything", k=10)

    assert [d.content for d in seen] == ["the user's name"]


@pytest.mark.asyncio
async def test_every_space_is_read_in_one_batch_and_merged_by_score(store):
    service = _service(store, GENERAL, PRIVATE, TEAM)
    await service.save_memory("u1", "likes tea", "preferences", space="general")
    await service.save_memory("u1", "writes Python", "facts", space="team")
    await service.save_memory("u1", "Python tests first", "workflow", space="private")
    _CountingStore.batches = 0

    docs = await service.recall_memories("u1", "python", k=2)

    assert _CountingStore.batches == 1
    assert {d.content for d in docs} == {"writes Python", "Python tests first"}
    assert {d.space.key for d in docs} == {"team", "private"}


@pytest.mark.asyncio
async def test_spaces_an_agent_is_not_given_are_not_read(store):
    writer = _service(store, GENERAL, PRIVATE, TEAM)
    outsider = _service(store, GENERAL)
    await writer.save_memory("u1", "team fact", "facts", space="team")

    assert await outsider.recall_memories("u1", "fact") == []


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_memory_is_written_to_the_space_named(store):
    service = _service(store, GENERAL, PRIVATE, TEAM)

    mid = await service.save_memory("u1", "team goal", "goals", space="team")

    assert store.get(("u1", "space:t1", "semantic"), mid) is not None


@pytest.mark.asyncio
async def test_without_a_name_a_memory_goes_to_the_default_space(store):
    service = _service(store, GENERAL, PRIVATE, TEAM, default_write="private")

    mid = await service.save_memory("u1", "a lesson", "skill")

    assert store.get(("u1", "agent:a1", "procedural"), mid) is not None


@pytest.mark.asyncio
async def test_without_a_default_the_first_writable_space_is_used(store):
    service = _service(store, MemorySpace(key="general", writable=False), TEAM)

    mid = await service.save_memory("u1", "a fact", "facts")

    assert store.get(("u1", "space:t1", "semantic"), mid) is not None


@pytest.mark.asyncio
async def test_a_space_the_agent_only_reads_is_not_written(store):
    service = _service(store, MemorySpace(key="general", writable=False), PRIVATE)

    with pytest.raises(MemorySpaceError, match="Spaces it can be written to: private"):
        await service.save_memory("u1", "a fact", "facts", space="general")


@pytest.mark.asyncio
async def test_with_no_writable_space_nothing_is_written(store):
    service = _service(store, MemorySpace(key="general", writable=False))

    with pytest.raises(MemorySpaceError):
        await service.save_memory("u1", "a fact", "facts")


# --------------------------------------------------------------------------- #
# Updating and deleting
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_memory_is_updated_where_it_lies(store):
    service = _service(store, GENERAL, PRIVATE, TEAM, default_write="private")
    mid = await service.save_memory("u1", "the team meets Monday", "context", space="team")

    outcome = await service.update_memory("u1", mid, "the team meets Tuesday")

    assert outcome.status == UPDATED
    assert outcome.space.key == "team"
    [doc] = await service.recall_memories("u1", "meets")
    assert doc.content == "the team meets Tuesday"
    assert doc.namespace == ("u1", "space:t1", "episodic")


@pytest.mark.asyncio
async def test_a_memory_in_a_space_read_only_is_neither_changed_nor_copied(store):
    """It used to be written again, under the same id, in the writer's space."""
    owner = _service(store, GENERAL, TEAM)
    reader = _service(store, GENERAL, MemorySpace(key="team", path=("space:t1",), writable=False))
    mid = await owner.save_memory("u1", "team fact", "facts", space="team")

    outcome = await reader.update_memory("u1", mid, "changed")
    deleted = await reader.remove_memory("u1", mid)

    assert outcome.status == READ_ONLY
    assert deleted.status == READ_ONLY
    everything = store.search(("u1",), limit=50)
    assert [(tuple(i.namespace), i.value["content"]) for i in everything] == [
        (("u1", "space:t1", "semantic"), "team fact")
    ]


@pytest.mark.asyncio
async def test_an_unknown_id_is_not_found(store):
    service = _service(store, GENERAL)

    assert (await service.update_memory("u1", "nope", "x")).status == NOT_FOUND
    assert (await service.remove_memory("u1", "nope")).status == NOT_FOUND


@pytest.mark.asyncio
async def test_a_new_category_moves_the_memory_within_its_space(store):
    service = _service(store, GENERAL, TEAM)
    mid = await service.save_memory("u1", "Deploy on Fridays", "general", space="team")

    await service.update_memory("u1", mid, "Never deploy on Fridays", "instructions")

    assert store.get(("u1", "space:t1", "semantic"), mid) is None
    moved = store.get(("u1", "space:t1", "procedural"), mid)
    assert moved.value["memory_type"] == "instructions"


@pytest.mark.asyncio
async def test_an_update_keeps_when_the_memory_was_first_saved(store):
    service = _service(store, GENERAL)
    mid = await service.save_memory("u1", "a fact", "facts")
    first = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    item = store.get(("u1", "semantic"), mid)
    store.put(("u1", "semantic"), mid, {**item.value, "created_at": first})

    await service.update_memory("u1", mid, "a better fact")

    [doc] = await service.recall_memories("u1", "fact")
    assert doc.value["created_at"] == first
    assert doc.saved_at == datetime.fromisoformat(first)


@pytest.mark.asyncio
async def test_a_memory_is_deleted_where_it_lies(store):
    service = _service(store, GENERAL, PRIVATE)
    mid = await service.save_memory("u1", "a lesson", "skill", space="private")

    outcome = await service.remove_memory("u1", mid)

    assert outcome.status == DELETED
    assert store.get(("u1", "agent:a1", "procedural"), mid) is None


def test_space_keys_are_unique(store):
    with pytest.raises(ValueError, match="unique"):
        _service(store, GENERAL, MemorySpace(key="general", path=("x",)))

"""A memory's previous version, kept by a change so that it can be put back.

``keep_previous_version`` is off by default: a change then leaves nothing of
the version it replaced, as before. These tests run on a real store.
"""

from datetime import datetime, timedelta, timezone
from typing import List

import pytest
from langgraph.store.memory import InMemoryStore

from inference_core.services.agent_memory_service import (
    CHANGED_SINCE,
    NO_PREVIOUS,
    NOT_FOUND,
    PREVIOUS_VERSION_KEY,
    RESTORED,
    UPDATED,
    AgentMemoryStoreService,
    MemoryOptions,
    MemorySpace,
    drop_expired_previous_versions,
    restore_previous_version,
)


def _embed(texts: List[str]) -> List[List[float]]:
    return [[1.0 if "krak" in text.lower() else 0.0, 0.5] for text in texts]


@pytest.fixture(autouse=True)
def _plain_text(monkeypatch):
    monkeypatch.setenv("ENCRYPT_MEMORY_WRITES", "false")


@pytest.fixture
def store():
    return InMemoryStore(index={"embed": _embed, "dims": 2, "fields": ["content"]})


GENERAL = MemorySpace(key="general", label="General", description="About the user.")


def _service(store, *, keep: bool):
    return MemoryOptions(spaces=(GENERAL,), keep_previous_version=keep).service(store)


def _version(store, namespace, key):
    return store.get(namespace, key).updated_at.isoformat()


@pytest.mark.asyncio
async def test_without_the_option_a_change_keeps_nothing_of_what_it_replaced(store):
    service = _service(store, keep=False)
    mid = await service.save_memory("u1", "Works in Krakow", "facts")
    before = dict(store.get(("u1", "semantic"), mid).value)

    await service.update_memory("u1", mid, "Works in Warsaw")

    value = store.get(("u1", "semantic"), mid).value
    assert PREVIOUS_VERSION_KEY not in value
    assert set(value) == set(before)


@pytest.mark.asyncio
async def test_a_change_keeps_the_version_it_replaced_as_it_was_stored(store):
    service = _service(store, keep=True)
    mid = await service.save_memory("u1", "Works in Krakow", "facts", topic="work")
    ns = ("u1", "semantic")
    # What the store holds is kept as it is: an encrypted text stays encrypted.
    item = store.get(ns, mid)
    store.put(ns, mid, {**item.value, "content": "enc.v1.opaque"}, index=False)
    written = _version(store, ns, mid)

    outcome = await service.update_memory("u1", mid, "Works in Warsaw")

    assert outcome.status == UPDATED
    previous = store.get(ns, mid).value[PREVIOUS_VERSION_KEY]
    assert previous["content"] == "enc.v1.opaque"
    assert previous["topic"] == "work"
    assert previous["memory_category"] == "semantic"
    assert previous["updated_at"] == written
    assert previous["kept_at"]


@pytest.mark.asyncio
async def test_one_version_at_most_is_kept(store):
    service = _service(store, keep=True)
    mid = await service.save_memory("u1", "Works in Krakow", "facts")

    await service.update_memory("u1", mid, "Works in Warsaw")
    await service.update_memory("u1", mid, "Works in Gdansk")

    value = store.get(("u1", "semantic"), mid).value
    assert value[PREVIOUS_VERSION_KEY]["content"] == "Works in Warsaw"
    assert PREVIOUS_VERSION_KEY not in value[PREVIOUS_VERSION_KEY]


@pytest.mark.asyncio
async def test_the_kept_version_is_not_found_by_meaning(store):
    service = _service(store, keep=True)
    mid = await service.save_memory("u1", "Works in Krakow", "facts")
    await service.update_memory("u1", mid, "Works in Warsaw")

    [hit] = store.search(("u1",), query="krakow", limit=5)

    # Only the text the memory says now is embedded: "Krakow" would match whole.
    assert hit.score < 0.9


@pytest.mark.asyncio
async def test_the_previous_version_comes_back(store):
    service = _service(store, keep=True)
    mid = await service.save_memory("u1", "Works in Krakow", "facts")
    ns = ("u1", "semantic")
    first_saved = store.get(ns, mid).value["created_at"]
    await service.update_memory("u1", mid, "Works in Warsaw")

    outcome = restore_previous_version(
        store, "u1", ns, mid, expected_version=_version(store, ns, mid)
    )

    assert outcome.status == RESTORED
    assert outcome.namespace == ns
    value = store.get(ns, mid).value
    assert value["content"] == "Works in Krakow"
    assert value["created_at"] == first_saved
    assert PREVIOUS_VERSION_KEY not in value
    assert restore_previous_version(store, "u1", ns, mid).status == NO_PREVIOUS


@pytest.mark.asyncio
async def test_a_memory_changed_since_is_left_as_it_is(store):
    service = _service(store, keep=True)
    mid = await service.save_memory("u1", "Works in Krakow", "facts")
    ns = ("u1", "semantic")
    await service.update_memory("u1", mid, "Works in Warsaw")
    known = _version(store, ns, mid)
    await service.update_memory("u1", mid, "Works in Gdansk")

    outcome = restore_previous_version(store, "u1", ns, mid, expected_version=known)

    assert outcome.status == CHANGED_SINCE
    assert store.get(ns, mid).value["content"] == "Works in Gdansk"


@pytest.mark.asyncio
async def test_a_memory_the_change_moved_goes_back_to_its_category(store):
    service = _service(store, keep=True)
    mid = await service.save_memory("u1", "Deploy on Fridays", "general")
    await service.update_memory("u1", mid, "Never deploy on Fridays", "instructions")
    moved = ("u1", "procedural")

    outcome = restore_previous_version(store, "u1", moved, mid)

    assert outcome.status == RESTORED
    assert outcome.namespace == ("u1", "semantic")
    assert store.get(moved, mid) is None
    back = store.get(("u1", "semantic"), mid).value
    assert back["content"] == "Deploy on Fridays"
    assert back["memory_type"] == "general"


def test_another_users_memory_or_none_is_not_found(store):
    store.put(("u2", "semantic"), "m1", {"content": "x", PREVIOUS_VERSION_KEY: {"content": "y"}})

    assert restore_previous_version(store, "u1", ("u2", "semantic"), "m1").status == NOT_FOUND
    assert restore_previous_version(store, "u1", ("u1", "semantic"), "m9").status == NOT_FOUND
    assert store.get(("u2", "semantic"), "m1").value["content"] == "x"


@pytest.mark.asyncio
async def test_versions_kept_long_enough_are_dropped(store):
    service = _service(store, keep=True)
    old = await service.save_memory("u1", "Works in Krakow", "facts")
    new = await service.save_memory("u2", "Likes tea", "facts")
    await service.update_memory("u1", old, "Works in Warsaw")
    await service.update_memory("u2", new, "Likes coffee")
    ns = ("u1", "semantic")
    value = store.get(ns, old).value
    long_ago = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    store.put(
        ns,
        old,
        {**value, PREVIOUS_VERSION_KEY: {**value[PREVIOUS_VERSION_KEY], "kept_at": long_ago}},
        index=False,
    )

    dropped = drop_expired_previous_versions(
        store, kept_before=datetime.now(timezone.utc) - timedelta(days=7)
    )

    assert dropped == 1
    kept = store.get(ns, old).value
    assert PREVIOUS_VERSION_KEY not in kept
    assert kept["content"] == "Works in Warsaw"
    assert PREVIOUS_VERSION_KEY in store.get(("u2", "semantic"), new).value

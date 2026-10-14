"""Tests for inference_core.services.agent_memory_service.

Covers enums, dataclasses, validators, MemoryNamespaceBuilder,
and AgentMemoryStoreService (save, recall, delete, bucketing, formatting).
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from langgraph.store.memory import InMemoryStore

from inference_core.services.agent_memory_service import (
    MEMORY_TYPE_TO_CATEGORY,
    AgentMemoryStoreService,
    MemoryCategory,
    MemoryData,
    MemoryMetadata,
    MemoryNamespaceBuilder,
    MemoryStoreDocument,
    MemoryType,
    MemoryTypeDescription,
    format_memory_types_for_description,
    get_category_for_type,
    get_memory_category_literal,
    get_memory_type_literal,
    get_types_for_category,
    validate_memory_category,
    validate_memory_type,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store_item(
    key: str = "mem-1",
    value: Optional[Dict[str, Any]] = None,
    score: float = 0.9,
    created_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None,
) -> MagicMock:
    """Build a mock store Item with needed attributes."""
    item = MagicMock()
    item.key = key
    item.value = value or {"content": "test memory", "memory_type": "general"}
    item.score = score
    item.created_at = created_at
    item.updated_at = updated_at
    return item


# ============================================================================
# Enums & constants
# ============================================================================


class TestMemoryCategory:
    """MemoryCategory enum members and string behaviour."""

    def test_has_three_members(self):
        assert set(MemoryCategory) == {
            MemoryCategory.SEMANTIC,
            MemoryCategory.EPISODIC,
            MemoryCategory.PROCEDURAL,
        }

    def test_values_are_lowercase_strings(self):
        for cat in MemoryCategory:
            assert cat.value == cat.value.lower()
            assert isinstance(cat.value, str)


class TestMemoryType:
    """MemoryType enum members and mapping consistency."""

    def test_all_types_mapped_to_category(self):
        """Every MemoryType value must appear in MEMORY_TYPE_TO_CATEGORY."""
        for mt in MemoryType:
            assert mt.value in MEMORY_TYPE_TO_CATEGORY

    def test_semantic_types(self):
        semantic = {
            mt
            for mt, cat in MEMORY_TYPE_TO_CATEGORY.items()
            if cat == MemoryCategory.SEMANTIC
        }
        assert "preferences" in semantic
        assert "facts" in semantic
        assert "goals" in semantic
        assert "general" in semantic

    def test_episodic_types(self):
        episodic = {
            mt
            for mt, cat in MEMORY_TYPE_TO_CATEGORY.items()
            if cat == MemoryCategory.EPISODIC
        }
        assert "context" in episodic
        assert "session_summary" in episodic
        assert "interaction" in episodic

    def test_procedural_types(self):
        procedural = {
            mt
            for mt, cat in MEMORY_TYPE_TO_CATEGORY.items()
            if cat == MemoryCategory.PROCEDURAL
        }
        assert "instructions" in procedural
        assert "workflow" in procedural
        assert "skill" in procedural


class TestMemoryTypeDescription:
    """Every MemoryType has a matching description enum member."""

    def test_descriptions_exist_for_all_types(self):
        for mt in MemoryType:
            assert hasattr(
                MemoryTypeDescription, mt.name
            ), f"Missing description for {mt.name}"


# ============================================================================
# Standalone functions
# ============================================================================


class TestGetCategoryForType:
    """get_category_for_type returns correct category or falls back to SEMANTIC."""

    @pytest.mark.parametrize(
        "memory_type, expected",
        [
            ("preferences", MemoryCategory.SEMANTIC),
            ("facts", MemoryCategory.SEMANTIC),
            ("context", MemoryCategory.EPISODIC),
            ("session_summary", MemoryCategory.EPISODIC),
            ("instructions", MemoryCategory.PROCEDURAL),
            ("workflow", MemoryCategory.PROCEDURAL),
        ],
    )
    def test_known_types(self, memory_type, expected):
        assert get_category_for_type(memory_type) == expected

    def test_unknown_type_falls_back_to_semantic(self):
        assert get_category_for_type("nonexistent") == MemoryCategory.SEMANTIC


class TestGetTypesForCategory:
    """get_types_for_category returns correct list of types."""

    def test_semantic_returns_four_types(self):
        result = get_types_for_category(MemoryCategory.SEMANTIC)
        assert set(result) == {"preferences", "facts", "goals", "general"}

    def test_episodic_returns_three_types(self):
        result = get_types_for_category(MemoryCategory.EPISODIC)
        assert set(result) == {"context", "session_summary", "interaction"}

    def test_procedural_returns_three_types(self):
        result = get_types_for_category(MemoryCategory.PROCEDURAL)
        assert set(result) == {"instructions", "workflow", "skill"}


class TestValidateMemoryType:
    """validate_memory_type accepts valid types and rejects invalid."""

    def test_valid_string(self):
        assert validate_memory_type("preferences") == "preferences"

    def test_valid_enum_value(self):
        """Accepts enum instance and unwraps its .value."""
        assert validate_memory_type(MemoryType.FACTS) == "facts"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid memory_type"):
            validate_memory_type("bogus")


class TestValidateMemoryCategory:
    """validate_memory_category accepts valid categories and rejects invalid."""

    def test_valid_string(self):
        assert validate_memory_category("semantic") == "semantic"

    def test_valid_enum_value(self):
        assert validate_memory_category(MemoryCategory.EPISODIC) == "episodic"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid memory_category"):
            validate_memory_category("unknown")


class TestFormatMemoryTypesForDescription:
    """format_memory_types_for_description returns structured text."""

    def test_contains_all_categories(self):
        result = format_memory_types_for_description()
        for cat in MemoryCategory:
            assert cat.value.upper() in result

    def test_contains_all_types(self):
        result = format_memory_types_for_description()
        for mt in MemoryType:
            assert mt.value in result


class TestLiteralHelpers:
    """get_memory_type_literal and get_memory_category_literal generate Literal types."""

    def test_memory_type_literal(self):
        lit = get_memory_type_literal()
        # It should be a typing.Literal containing all MemoryType values
        from typing import get_args

        args = get_args(lit)
        for mt in MemoryType:
            assert mt.value in args

    def test_memory_category_literal(self):
        lit = get_memory_category_literal()
        from typing import get_args

        args = get_args(lit)
        for mc in MemoryCategory:
            assert mc.value in args


# ============================================================================
# Dataclasses
# ============================================================================


class TestMemoryMetadata:
    """MemoryMetadata serialisation and default fields."""

    def test_to_dict_required_fields(self):
        md = MemoryMetadata(user_id="u1")
        d = md.to_dict()
        assert d["user_id"] == "u1"
        assert "memory_type" in d
        assert "memory_category" in d
        assert "created_at" in d

    def test_to_dict_optional_fields_omitted(self):
        md = MemoryMetadata(user_id="u1")
        d = md.to_dict()
        assert "session_id" not in d
        assert "topic" not in d
        assert "agent_name" not in d

    def test_to_dict_optional_fields_included(self):
        md = MemoryMetadata(
            user_id="u1",
            session_id="s1",
            topic="testing",
            agent_name="bot",
            extra={"k": "v"},
        )
        d = md.to_dict()
        assert d["session_id"] == "s1"
        assert d["topic"] == "testing"
        assert d["agent_name"] == "bot"
        assert d["k"] == "v"


class TestMemoryStoreDocument:
    """MemoryStoreDocument properties and edge cases."""

    def test_content_property(self):
        doc = MemoryStoreDocument(
            id="1",
            value={"content": "hello"},
            metadata={},
        )
        assert doc.content == "hello"

    def test_content_missing_returns_empty(self):
        doc = MemoryStoreDocument(id="1", value={}, metadata={})
        assert doc.content == ""

    def test_topic_property(self):
        doc = MemoryStoreDocument(
            id="1",
            value={"topic": "AI"},
            metadata={},
        )
        assert doc.topic == "AI"

    def test_memory_type_and_category(self):
        doc = MemoryStoreDocument(
            id="1",
            value={"memory_type": "facts"},
            metadata={},
        )
        assert doc.memory_type == "facts"
        assert doc.memory_category == "semantic"

    def test_memory_category_none_when_no_type(self):
        doc = MemoryStoreDocument(id="1", value={}, metadata={})
        assert doc.memory_category is None

    def test_created_at_iso(self):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        doc = MemoryStoreDocument(id="1", value={}, metadata={}, created_at=dt)
        assert doc.created_at_iso == dt.isoformat()

    def test_created_at_iso_none(self):
        doc = MemoryStoreDocument(id="1", value={}, metadata={})
        assert doc.created_at_iso is None

    def test_updated_at_iso(self):
        dt = datetime(2025, 6, 1, tzinfo=timezone.utc)
        doc = MemoryStoreDocument(id="1", value={}, metadata={}, updated_at=dt)
        assert doc.updated_at_iso == dt.isoformat()


class TestMemoryData:
    """MemoryData auto-resolves category in __post_init__."""

    def test_auto_resolves_category(self):
        data = MemoryData(content="hello", memory_type="workflow")
        assert data.memory_category == "procedural"

    def test_explicit_category_not_overridden(self):
        data = MemoryData(
            content="hello",
            memory_type="workflow",
            memory_category="episodic",
        )
        assert data.memory_category == "episodic"


# ============================================================================
# MemoryNamespaceBuilder
# ============================================================================


class TestMemoryNamespaceBuilder:
    """Namespace routing rules per CoALA spec."""

    def test_semantic_always_shared(self):
        """Semantic namespace never contains agent_name."""
        builder = MemoryNamespaceBuilder(agent_name="bot1")
        ns = builder.namespace_for("user1", MemoryCategory.SEMANTIC)
        assert ns == ("user1", "semantic")

    def test_episodic_per_agent(self):
        builder = MemoryNamespaceBuilder(agent_name="bot1")
        ns = builder.namespace_for("user1", MemoryCategory.EPISODIC)
        assert ns == ("user1", "episodic", "bot1")

    def test_episodic_shared_when_no_agent(self):
        builder = MemoryNamespaceBuilder(agent_name=None)
        ns = builder.namespace_for("user1", MemoryCategory.EPISODIC)
        assert ns == ("user1", "episodic")

    def test_procedural_per_agent(self):
        builder = MemoryNamespaceBuilder(agent_name="bot2")
        ns = builder.namespace_for("user1", MemoryCategory.PROCEDURAL)
        assert ns == ("user1", "procedural", "bot2")

    def test_procedural_shared_when_no_agent(self):
        builder = MemoryNamespaceBuilder()
        ns = builder.namespace_for("user1", MemoryCategory.PROCEDURAL)
        assert ns == ("user1", "procedural")

    def test_namespace_for_type_convenience(self):
        builder = MemoryNamespaceBuilder(agent_name="bot")
        ns = builder.namespace_for_type("user1", "instructions")
        expected = builder.namespace_for("user1", MemoryCategory.PROCEDURAL)
        assert ns == expected

    def test_legacy_namespace(self):
        builder = MemoryNamespaceBuilder()
        ns = builder.legacy_namespace("user1")
        assert ns == ("user1", "memories")


# ============================================================================
# AgentMemoryStoreService
# ============================================================================


@pytest.fixture
def mock_store():
    """Minimal mock store with put/search/delete."""
    store = MagicMock()
    store.search.return_value = []
    return store


@pytest.fixture
def service(mock_store):
    return AgentMemoryStoreService(store=mock_store, agent_name="testbot")


class TestSaveMemory:
    """save_memory validates type, resolves namespace, delegates to store.put."""

    @pytest.mark.asyncio
    async def test_save_returns_generated_id(self, service):
        mid = await service.save_memory(
            user_id="u1",
            content="user likes tests",
            memory_type="preferences",
        )
        assert isinstance(mid, str)
        assert len(mid) > 0

    @pytest.mark.asyncio
    async def test_save_calls_store_put(self, service, mock_store):
        await service.save_memory(
            user_id="u1",
            content="hello",
            memory_type="general",
        )
        mock_store.put.assert_called_once()
        args, kwargs = mock_store.put.call_args
        # First arg is namespace tuple
        assert isinstance(args[0], tuple)
        # Index fields should be passed
        assert kwargs.get("index") or "index" in {k for k in (kwargs or {})}

    @pytest.mark.asyncio
    async def test_save_with_explicit_category(self, service, mock_store):
        await service.save_memory(
            user_id="u1",
            content="log entry",
            memory_type="context",
            category="episodic",
        )
        ns_arg = mock_store.put.call_args[0][0]
        assert "episodic" in ns_arg

    @pytest.mark.asyncio
    async def test_save_invalid_type_raises(self, service):
        with pytest.raises(AssertionError, match="Invalid memory type"):
            await service.save_memory(
                user_id="u1", content="x", memory_type="invalid_type"
            )

    @pytest.mark.asyncio
    async def test_save_with_custom_id(self, service, mock_store):
        mid = await service.save_memory(
            user_id="u1",
            content="x",
            memory_type="general",
            memory_id="custom-id-123",
        )
        assert mid == "custom-id-123"
        assert mock_store.put.call_args[0][1] == "custom-id-123"


def _embed(texts: List[str]) -> List[List[float]]:
    """Two dimensions: how much a text is about Python, and the rest."""
    return [[1.0 if "python" in text.lower() else 0.0, 0.5] for text in texts]


class _CountingStore(InMemoryStore):
    """An in-memory store that counts the batches it was sent."""

    batches: int = 0

    def batch(self, ops):
        type(self).batches += 1
        return super().batch(ops)


@pytest.fixture
def real_store():
    _CountingStore.batches = 0
    return _CountingStore(index={"embed": _embed, "dims": 2, "fields": ["content"]})


@pytest.fixture
def agent_service(real_store):
    return AgentMemoryStoreService(
        store=real_store, agent_name="testbot", scope="dedicated"
    )


class TestRecallMemories:
    """recall_memories searches the agent's own namespaces, in one batch."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_results(self, agent_service):
        result = await agent_service.recall_memories(user_id="u1", query="anything")
        assert result == []

    @pytest.mark.asyncio
    async def test_every_category_is_searched_in_one_batch(
        self, agent_service, real_store
    ):
        for memory_type in ("facts", "context", "workflow"):
            await agent_service.save_memory("u1", f"a {memory_type}", memory_type)
        _CountingStore.batches = 0

        docs = await agent_service.recall_memories(user_id="u1", query="test")

        assert _CountingStore.batches == 1
        assert sorted(d.memory_type for d in docs) == ["context", "facts", "workflow"]

    @pytest.mark.asyncio
    async def test_searches_single_category(self, agent_service):
        await agent_service.save_memory("u1", "a fact", "facts")
        await agent_service.save_memory("u1", "a step", "workflow")

        docs = await agent_service.recall_memories(
            user_id="u1", query="test", category="semantic"
        )

        assert [d.content for d in docs] == ["a fact"]

    @pytest.mark.asyncio
    async def test_auto_resolves_category_from_type(self, agent_service):
        await agent_service.save_memory("u1", "a fact", "facts")
        await agent_service.save_memory("u1", "a step", "workflow")

        docs = await agent_service.recall_memories(
            user_id="u1", query="test", memory_type="workflow"
        )

        assert [d.content for d in docs] == ["a step"]
        assert docs[0].namespace == ("u1", "procedural", "testbot")

    @pytest.mark.asyncio
    async def test_sort_by_score(self, agent_service):
        """Default sort: the closest first."""
        await agent_service.save_memory("u1", "likes tea", "preferences")
        await agent_service.save_memory("u1", "writes Python", "facts")

        docs = await agent_service.recall_memories(
            user_id="u1", query="python", category="semantic"
        )

        assert [d.content for d in docs] == ["writes Python", "likes tea"]

    @pytest.mark.asyncio
    async def test_sort_by_time(self, agent_service):
        """sort_by_time=True → the latest saved first."""
        now = datetime.now(timezone.utc)
        old = await agent_service.save_memory("u1", "old", "facts")
        await agent_service.save_memory("u1", "new", "facts")
        item = agent_service.store.get(("u1", "semantic", "testbot"), old)
        agent_service.store.put(
            ("u1", "semantic", "testbot"),
            old,
            {**item.value, "created_at": (now - timedelta(days=5)).isoformat()},
        )

        docs = await agent_service.recall_memories(
            user_id="u1", query="test", category="semantic", sort_by_time=True
        )

        assert [d.content for d in docs] == ["new", "old"]

    @pytest.mark.asyncio
    async def test_a_document_says_where_it_was_found(self, agent_service):
        await agent_service.save_memory("u1", "a fact", "facts")

        [doc] = await agent_service.recall_memories(user_id="u1", query="fact")

        assert doc.space.key == "agent"
        assert doc.space_key == "agent"
        assert doc.namespace == ("u1", "semantic", "testbot")


class TestDeleteMemory:
    """delete_memory deletes a memory where it lies, and says whether it did."""

    @pytest.mark.asyncio
    async def test_delete_with_category(self, agent_service):
        mid = await agent_service.save_memory("u1", "a fact", "facts")

        assert await agent_service.delete_memory("u1", mid, category="semantic")
        assert await agent_service.recall_memories("u1", "fact") == []

    @pytest.mark.asyncio
    async def test_an_episodic_memory_is_deleted_without_its_category(
        self, agent_service
    ):
        """It used to report success after deleting from the semantic
        namespace, where the memory was not, and leave it."""
        mid = await agent_service.save_memory("u1", "met on Monday", "context")

        assert await agent_service.delete_memory("u1", mid) is True
        assert await agent_service.recall_memories("u1", "met") == []

    @pytest.mark.asyncio
    async def test_a_missing_id_is_not_reported_deleted(self, agent_service):
        assert await agent_service.delete_memory("u1", "does-not-exist") is False

    @pytest.mark.asyncio
    async def test_delete_no_store_support(self, mock_store):
        """Store without .delete returns False."""
        del mock_store.delete  # remove the attribute
        svc = AgentMemoryStoreService(store=mock_store)
        result = await svc.delete_memory(user_id="u1", memory_id="m1")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_failure(self, agent_service, real_store, monkeypatch):
        """An exception during delete returns False."""
        mid = await agent_service.save_memory("u1", "a fact", "facts")

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(real_store, "delete", boom)
        assert await agent_service.delete_memory("u1", mid) is False


class TestDeleteUserMemories:
    """delete_user_memories bulk-deletes by type, category, or all."""

    @pytest.mark.asyncio
    async def test_delete_by_type(self, agent_service):
        await agent_service.save_memory("u1", "likes tea", "preferences")
        await agent_service.save_memory("u1", "lives in Krakow", "facts")

        count = await agent_service.delete_user_memories(
            user_id="u1", memory_type="preferences"
        )

        assert count == 1
        assert [d.content for d in await agent_service.recall_memories("u1", "x")] == [
            "lives in Krakow"
        ]

    @pytest.mark.asyncio
    async def test_delete_all(self, agent_service):
        for memory_type in ("facts", "context", "workflow"):
            await agent_service.save_memory("u1", f"a {memory_type}", memory_type)

        assert await agent_service.delete_user_memories(user_id="u1") == 3
        assert await agent_service.recall_memories("u1", "x") == []

    @pytest.mark.asyncio
    async def test_only_the_agents_own_memories_are_deleted(self, real_store):
        shared = AgentMemoryStoreService(store=real_store, scope="shared")
        own = AgentMemoryStoreService(
            store=real_store, agent_name="testbot", scope="dedicated"
        )
        await shared.save_memory("u1", "everyone's fact", "facts")
        await own.save_memory("u1", "own fact", "facts")

        assert await shared.delete_user_memories(user_id="u1") == 1
        assert [d.content for d in await own.recall_memories("u1", "x")] == [
            "own fact"
        ]


class TestBucketMemoriesByTime:
    """_bucket_memories_by_time groups into 5 temporal buckets and deduplicates."""

    def _make_doc(self, content: str, days_ago: int = 0) -> MemoryStoreDocument:
        dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
        return MemoryStoreDocument(
            id=str(uuid.uuid4()),
            value={"content": content},
            metadata={},
            created_at=dt,
        )

    def test_today_bucket(self, service):
        doc = self._make_doc("today entry", days_ago=0)
        buckets = service._bucket_memories_by_time([doc])
        assert len(buckets["today"]) == 1

    def test_yesterday_bucket(self, service):
        doc = self._make_doc("yesterday entry", days_ago=1)
        buckets = service._bucket_memories_by_time([doc])
        assert len(buckets["yesterday"]) == 1

    def test_prev5_bucket(self, service):
        doc = self._make_doc("3 days ago", days_ago=3)
        buckets = service._bucket_memories_by_time([doc])
        assert len(buckets["prev5_before_yesterday"]) == 1

    def test_prev90_bucket(self, service):
        doc = self._make_doc("30 days ago", days_ago=30)
        buckets = service._bucket_memories_by_time([doc])
        assert len(buckets["prev90_excluding_above"]) == 1

    def test_older_bucket(self, service):
        doc = self._make_doc("200 days ago", days_ago=200)
        buckets = service._bucket_memories_by_time([doc])
        assert len(buckets["older"]) == 1

    def test_deduplicates_by_content(self, service):
        docs = [
            self._make_doc("same content", days_ago=0),
            self._make_doc("same content", days_ago=1),
        ]
        buckets = service._bucket_memories_by_time(docs)
        total = sum(len(v) for v in buckets.values())
        assert total == 1

    def test_empty_content_skipped(self, service):
        doc = MemoryStoreDocument(
            id="1",
            value={"content": ""},
            metadata={},
            created_at=datetime.now(timezone.utc),
        )
        buckets = service._bucket_memories_by_time([doc])
        total = sum(len(v) for v in buckets.values())
        assert total == 0

    def test_no_created_at_goes_to_older(self, service):
        """Memories without timestamp fall into 'older' bucket."""
        doc = MemoryStoreDocument(
            id="1",
            value={"content": "no timestamp"},
            metadata={},
            created_at=None,
        )
        buckets = service._bucket_memories_by_time([doc])
        assert len(buckets["older"]) == 1


class TestFormatContextForPrompt:
    """format_context_for_prompt aggregates and renders CoALA XML."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_memories(self, service):
        result = await service.format_context_for_prompt(user_id="u1")
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_coala_xml_sections(self, agent_service):
        """Returned string contains category XML tags."""
        await agent_service.save_memory("u1", "user likes Python", "preferences")
        result = await agent_service.format_context_for_prompt(user_id="u1")
        assert "<semantic_memory>" in result
        assert "</semantic_memory>" in result
        assert "user likes Python" in result


class TestGetMemoryDate:
    """_get_memory_date extracts date from MemoryStoreDocument."""

    def test_with_timezone_aware(self, service):
        dt = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
        doc = MemoryStoreDocument(id="1", value={}, metadata={}, created_at=dt)
        assert service._get_memory_date(doc) == dt.date()

    def test_with_naive_datetime(self, service):
        """Naive datetime treated as UTC."""
        dt = datetime(2025, 6, 15, 12, 0)
        doc = MemoryStoreDocument(id="1", value={}, metadata={}, created_at=dt)
        assert service._get_memory_date(doc) == dt.date()

    def test_none_returns_none(self, service):
        doc = MemoryStoreDocument(id="1", value={}, metadata={})
        assert service._get_memory_date(doc) is None


# ============================================================================
# Standalone purge / count helpers (InMemoryStore-backed)
# ============================================================================


class TestPurgeAndCountUserMemories:
    """Cover purge_user_memories / count_user_memories on a real store.

    Uses a real InMemoryStore seeded across all CoALA namespaces, including
    per-agent episodic/procedural sub-namespaces, to lock in correct
    namespace-aware deletion and user isolation.
    """

    @pytest.fixture
    def store(self):
        from langgraph.store.memory import InMemoryStore

        s = InMemoryStore()
        # user u1 across all categories (episodic/procedural are per-agent)
        s.put(("u1", "semantic"), "s1", {"content": "fact", "memory_type": "facts"})
        s.put(
            ("u1", "semantic"), "s2", {"content": "pref", "memory_type": "preferences"}
        )
        s.put(
            ("u1", "episodic", "assistant"),
            "e1",
            {"content": "session", "memory_type": "interaction"},
        )
        s.put(
            ("u1", "procedural", "assistant"),
            "p1",
            {"content": "rule", "memory_type": "skill"},
        )
        # a different user must remain untouched
        s.put(("u2", "semantic"), "o1", {"content": "other", "memory_type": "facts"})
        return s

    def test_count_all_categories(self, store):
        from inference_core.services.agent_memory_service import count_user_memories

        counts = count_user_memories(store, "u1")
        assert counts == {
            "semantic": 2,
            "episodic": 1,
            "procedural": 1,
            "total": 4,
        }

    def test_purge_single_category_only(self, store):
        from inference_core.services.agent_memory_service import (
            count_user_memories,
            purge_user_memories,
        )

        deleted = purge_user_memories(store, "u1", "semantic")
        assert deleted == 2
        counts = count_user_memories(store, "u1")
        assert counts == {"semantic": 0, "episodic": 1, "procedural": 1, "total": 2}

    def test_purge_per_agent_namespace(self, store):
        """Episodic items live under (user, 'episodic', agent_name)."""
        from inference_core.services.agent_memory_service import (
            count_user_memories,
            purge_user_memories,
        )

        assert purge_user_memories(store, "u1", "episodic") == 1
        assert count_user_memories(store, "u1")["episodic"] == 0

    def test_purge_all(self, store):
        from inference_core.services.agent_memory_service import (
            count_user_memories,
            purge_user_memories,
        )

        deleted = purge_user_memories(store, "u1")
        assert deleted == 4
        assert count_user_memories(store, "u1")["total"] == 0

    def test_other_user_untouched(self, store):
        from inference_core.services.agent_memory_service import (
            count_user_memories,
            purge_user_memories,
        )

        purge_user_memories(store, "u1")
        assert count_user_memories(store, "u2")["total"] == 1

    def test_invalid_category_raises(self, store):
        from inference_core.services.agent_memory_service import purge_user_memories

        with pytest.raises(ValueError):
            purge_user_memories(store, "u1", "not-a-category")


class TestListAndDeleteMemoryEntries:
    """Cover list_user_memories + entry-level delete helpers on a real store."""

    @pytest.fixture
    def store(self):
        from langgraph.store.memory import InMemoryStore

        s = InMemoryStore()
        s.put(
            ("u1", "semantic"),
            "s1",
            {
                "content": "Lives in Warsaw",
                "memory_type": "facts",
                "topic": "location",
                "created_at": "2026-01-01T10:00:00+00:00",
            },
        )
        s.put(
            ("u1", "semantic"),
            "s2",
            {
                "content": "Prefers bullet points",
                "memory_type": "preferences",
                "created_at": "2026-02-01T10:00:00+00:00",
            },
        )
        s.put(
            ("u1", "episodic", "assistant"),
            "e1",
            {
                "content": "Discussed Q1 goals",
                "memory_type": "interaction",
                "created_at": "2026-03-01T10:00:00+00:00",
            },
        )
        s.put(("u2", "semantic"), "o1", {"content": "other", "memory_type": "facts"})
        return s

    def test_list_all_sorted_desc(self, store):
        from inference_core.services.agent_memory_service import list_user_memories

        items, total = list_user_memories(store, "u1")
        assert total == 3
        # newest first by created_at
        assert [i["key"] for i in items] == ["e1", "s2", "s1"]
        # category/agent derived from namespace
        assert items[0]["memory_category"] == "episodic"
        assert items[0]["agent_name"] == "assistant"

    def test_list_filter_by_category(self, store):
        from inference_core.services.agent_memory_service import list_user_memories

        items, total = list_user_memories(store, "u1", category="semantic")
        assert total == 2
        assert {i["key"] for i in items} == {"s1", "s2"}

    def test_list_filter_by_type(self, store):
        from inference_core.services.agent_memory_service import list_user_memories

        items, total = list_user_memories(store, "u1", memory_type="preferences")
        assert total == 1
        assert items[0]["key"] == "s2"

    def test_list_search_substring(self, store):
        from inference_core.services.agent_memory_service import list_user_memories

        items, total = list_user_memories(store, "u1", search="WARSAW")
        assert total == 1
        assert items[0]["key"] == "s1"

    def test_list_pagination(self, store):
        from inference_core.services.agent_memory_service import list_user_memories

        items, total = list_user_memories(store, "u1", limit=1, offset=1)
        assert total == 3
        assert [i["key"] for i in items] == ["s2"]

    def test_delete_entry_ownership_guard(self, store):
        from inference_core.services.agent_memory_service import (
            delete_user_memory_entry,
            list_user_memories,
        )

        # Refuse deleting another user's item even if namespace is supplied.
        assert delete_user_memory_entry(store, "u1", ["u2", "semantic"], "o1") is False
        assert list_user_memories(store, "u2")[1] == 1

    def test_delete_entry_owned(self, store):
        from inference_core.services.agent_memory_service import (
            delete_user_memory_entry,
            list_user_memories,
        )

        assert (
            delete_user_memory_entry(store, "u1", ["u1", "episodic", "assistant"], "e1")
            is True
        )
        assert list_user_memories(store, "u1")[1] == 2

    def test_bulk_delete_counts_only_deleted(self, store):
        from inference_core.services.agent_memory_service import (
            delete_user_memory_entries,
            list_user_memories,
        )

        deleted = delete_user_memory_entries(
            store,
            "u1",
            [
                (["u1", "semantic"], "s1"),
                (["u2", "semantic"], "o1"),  # not owned -> skipped
                (["u1", "semantic"], "s2"),
            ],
        )
        assert deleted == 2
        assert list_user_memories(store, "u1")[1] == 1
        assert list_user_memories(store, "u2")[1] == 1


# =============================================================================
# open_memory_store
# =============================================================================


class _FakeEmbeddings:
    """Stands in for the process's embedding service: two-dimensional vectors
    that put texts about food apart from everything else."""

    def __init__(self):
        self.embedded: List[str] = []

    def get_dimension(self) -> int:
        return 2

    def get_embed_fn(self):
        def embed(texts):
            self.embedded.extend(texts)
            return [
                [1.0, 0.0] if "pizza" in text or "eat" in text else [0.0, 1.0]
                for text in texts
            ]

        return embed


class TestOpenMemoryStore:
    @pytest.fixture
    def embeddings(self):
        fake = _FakeEmbeddings()
        with patch(
            "inference_core.services.embedding_service.get_embedding_service",
            return_value=fake,
        ):
            yield fake

    @pytest.fixture
    def unknown_dialect(self):
        """A database the store has no backend for: an in-memory store is used."""
        settings = MagicMock(database_url="oracle://example/db")
        with patch("inference_core.core.config.get_settings", return_value=settings):
            yield

    def test_without_an_index_nothing_is_embedded(self, embeddings, unknown_dialect):
        from inference_core.services.agent_memory_service import open_memory_store

        with open_memory_store() as store:
            store.put(("u1", "semantic"), "a", {"content": "likes pizza"})
            assert [item.key for item in store.search(("u1",))] == ["a"]

        assert embeddings.embedded == []

    def test_with_an_index_a_search_ranks_by_meaning(self, embeddings, unknown_dialect):
        from inference_core.services.agent_memory_service import open_memory_store

        with open_memory_store(with_index=True) as store:
            store.put(("u1", "semantic"), "car", {"content": "drives a blue car"})
            store.put(("u1", "semantic"), "food", {"content": "likes pizza"})
            found = store.search(("u1",), query="what does the user eat", limit=1)

        assert [item.key for item in found] == ["food"]
        assert embeddings.embedded

    @pytest.mark.parametrize(
        "url, store_path",
        [
            ("sqlite+aiosqlite:///./x.db", "langgraph.store.sqlite.SqliteStore"),
            ("postgresql+asyncpg://u@h/db", "langgraph.store.postgres.PostgresStore"),
        ],
    )
    @pytest.mark.parametrize("with_index", [False, True])
    def test_the_index_goes_to_the_database_store(
        self, embeddings, url, store_path, with_index
    ):
        from inference_core.services.agent_memory_service import open_memory_store

        opened = MagicMock()
        settings = MagicMock(database_url=url)
        with (
            patch("inference_core.core.config.get_settings", return_value=settings),
            patch(f"{store_path}.from_conn_string") as from_conn_string,
        ):
            from_conn_string.return_value.__enter__.return_value = opened
            with open_memory_store(with_index=with_index) as store:
                assert store is opened

        (sync_url,), kwargs = from_conn_string.call_args
        assert "+" not in sync_url
        if with_index:
            assert kwargs["index"]["dims"] == 2
            assert callable(kwargs["index"]["embed"])
        else:
            assert kwargs["index"] is None
        opened.setup.assert_called_once()

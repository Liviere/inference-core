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
            assert hasattr(MemoryTypeDescription, mt.name), (
                f"Missing description for {mt.name}"
            )


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
        doc = MemoryStoreDocument(
            id="1", value={}, metadata={}, updated_at=dt
        )
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


class TestRecallMemories:
    """recall_memories searches correct namespaces and sorts results."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_results(self, service):
        result = await service.recall_memories(user_id="u1", query="anything")
        assert result == []

    @pytest.mark.asyncio
    async def test_searches_all_categories_when_none_specified(
        self, service, mock_store
    ):
        await service.recall_memories(user_id="u1", query="test")
        # Should call store.search once per MemoryCategory
        assert mock_store.search.call_count == len(MemoryCategory)

    @pytest.mark.asyncio
    async def test_searches_single_category(self, service, mock_store):
        await service.recall_memories(
            user_id="u1", query="test", category="semantic"
        )
        assert mock_store.search.call_count == 1

    @pytest.mark.asyncio
    async def test_auto_resolves_category_from_type(self, service, mock_store):
        await service.recall_memories(
            user_id="u1", query="test", memory_type="workflow"
        )
        # Only procedural namespace searched
        assert mock_store.search.call_count == 1
        ns_arg = mock_store.search.call_args[0][0]
        assert "procedural" in ns_arg

    @pytest.mark.asyncio
    async def test_sort_by_score(self, service, mock_store):
        """Default sort: highest score first."""
        items = [
            _make_store_item(key="low", score=0.3),
            _make_store_item(key="high", score=0.9),
        ]
        mock_store.search.return_value = items

        docs = await service.recall_memories(
            user_id="u1", query="test", category="semantic"
        )
        assert docs[0].id == "high"
        assert docs[1].id == "low"

    @pytest.mark.asyncio
    async def test_sort_by_time(self, service, mock_store):
        """sort_by_time=True → newest created_at first."""
        now = datetime.now(timezone.utc)
        items = [
            _make_store_item(key="old", created_at=now - timedelta(days=5)),
            _make_store_item(key="new", created_at=now),
        ]
        mock_store.search.return_value = items

        docs = await service.recall_memories(
            user_id="u1",
            query="test",
            category="semantic",
            sort_by_time=True,
        )
        assert docs[0].id == "new"


class TestDeleteMemory:
    """delete_memory removes by id, with or without known category."""

    @pytest.mark.asyncio
    async def test_delete_with_category(self, service, mock_store):
        result = await service.delete_memory(
            user_id="u1", memory_id="m1", category="semantic"
        )
        assert result is True
        mock_store.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_without_category_tries_all(self, service, mock_store):
        """Iterates all namespaces until delete succeeds."""
        result = await service.delete_memory(user_id="u1", memory_id="m1")
        assert result is True
        # First namespace call should succeed
        assert mock_store.delete.call_count >= 1

    @pytest.mark.asyncio
    async def test_delete_no_store_support(self, mock_store):
        """Store without .delete returns False."""
        del mock_store.delete  # remove the attribute
        svc = AgentMemoryStoreService(store=mock_store)
        result = await svc.delete_memory(user_id="u1", memory_id="m1")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_with_category_failure(self, service, mock_store):
        """Exception during delete returns False."""
        mock_store.delete.side_effect = RuntimeError("boom")
        result = await service.delete_memory(
            user_id="u1", memory_id="m1", category="semantic"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_without_category_all_fail(self, service, mock_store):
        """All namespace attempts fail → returns False."""
        mock_store.delete.side_effect = RuntimeError("not found")
        result = await service.delete_memory(user_id="u1", memory_id="m1")
        assert result is False


class TestDeleteUserMemories:
    """delete_user_memories bulk-deletes by type, category, or all."""

    @pytest.mark.asyncio
    async def test_delete_by_type(self, service, mock_store):
        mock_store.search.return_value = [_make_store_item(key="m1")]
        count = await service.delete_user_memories(
            user_id="u1", memory_type="preferences"
        )
        assert count == 1
        mock_store.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_all(self, service, mock_store):
        mock_store.search.return_value = [_make_store_item(key="m1")]
        count = await service.delete_user_memories(user_id="u1")
        # Called once per category
        assert count == len(MemoryCategory)

    @pytest.mark.asyncio
    async def test_delete_type_error_fallback(self, service, mock_store):
        """Falls back to query=None when TypeError is raised."""
        call_count = 0

        def _search_side_effect(ns, query=None, filter=None):
            nonlocal call_count
            call_count += 1
            if query == "":
                raise TypeError("bad query")
            return [_make_store_item(key=f"m{call_count}")]

        mock_store.search.side_effect = _search_side_effect
        count = await service.delete_user_memories(user_id="u1")
        assert count >= 1


class TestBucketMemoriesByTime:
    """_bucket_memories_by_time groups into 5 temporal buckets and deduplicates."""

    def _make_doc(
        self, content: str, days_ago: int = 0
    ) -> MemoryStoreDocument:
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
    async def test_returns_coala_xml_sections(self, service, mock_store):
        """Returned string contains category XML tags."""
        now = datetime.now(timezone.utc)
        mock_store.search.return_value = [
            _make_store_item(
                key="m1",
                value={
                    "content": "user likes Python",
                    "memory_type": "preferences",
                },
                created_at=now,
            ),
        ]
        result = await service.format_context_for_prompt(user_id="u1")
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

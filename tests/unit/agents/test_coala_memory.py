"""
Tests for CoALA Memory Architecture.

Covers:
- MemoryCategory enum and mapping
- MemoryNamespaceBuilder namespace resolution
- AgentMemoryStoreService CoALA-aware CRUD
- Memory tools category parameter
- MemoryMiddleware multi-category recall
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from langgraph.store.memory import InMemoryStore

from inference_core.services.agent_memory_service import (
    MEMORY_TYPE_TO_CATEGORY,
    AgentMemoryStoreService,
    MemoryCategory,
    MemoryData,
    MemoryNamespaceBuilder,
    MemoryStoreDocument,
    MemoryType,
    format_memory_types_for_description,
    get_category_for_type,
    get_types_for_category,
    validate_memory_category,
    validate_memory_type,
)

# =============================================================================
# MemoryCategory enum tests
# =============================================================================


class TestMemoryCategory:
    def test_category_values(self):
        assert MemoryCategory.SEMANTIC.value == "semantic"
        assert MemoryCategory.EPISODIC.value == "episodic"
        assert MemoryCategory.PROCEDURAL.value == "procedural"

    def test_all_categories_present(self):
        assert len(MemoryCategory) == 3


# =============================================================================
# MemoryType → Category mapping tests
# =============================================================================


class TestMemoryTypeCategoryMapping:
    """Verify every MemoryType is mapped to exactly one category."""

    def test_all_types_mapped(self):
        for mt in MemoryType:
            assert (
                mt.value in MEMORY_TYPE_TO_CATEGORY
            ), f"MemoryType {mt.value} missing from MEMORY_TYPE_TO_CATEGORY"

    @pytest.mark.parametrize(
        "memory_type,expected_category",
        [
            ("preferences", MemoryCategory.SEMANTIC),
            ("facts", MemoryCategory.SEMANTIC),
            ("goals", MemoryCategory.SEMANTIC),
            ("general", MemoryCategory.SEMANTIC),
            ("context", MemoryCategory.EPISODIC),
            ("session_summary", MemoryCategory.EPISODIC),
            ("interaction", MemoryCategory.EPISODIC),
            ("instructions", MemoryCategory.PROCEDURAL),
            ("workflow", MemoryCategory.PROCEDURAL),
            ("skill", MemoryCategory.PROCEDURAL),
        ],
    )
    def test_type_to_category(self, memory_type, expected_category):
        assert get_category_for_type(memory_type) == expected_category

    def test_unknown_type_defaults_to_semantic(self):
        """Unknown types should fall back to SEMANTIC for backward compat."""
        assert get_category_for_type("nonexistent_type") == MemoryCategory.SEMANTIC

    def test_get_types_for_category_semantic(self):
        types = get_types_for_category(MemoryCategory.SEMANTIC)
        assert set(types) == {"preferences", "facts", "goals", "general"}

    def test_get_types_for_category_episodic(self):
        types = get_types_for_category(MemoryCategory.EPISODIC)
        assert set(types) == {"context", "session_summary", "interaction"}

    def test_get_types_for_category_procedural(self):
        types = get_types_for_category(MemoryCategory.PROCEDURAL)
        assert set(types) == {"instructions", "workflow", "skill"}


# =============================================================================
# Validator tests
# =============================================================================


class TestValidators:
    def test_validate_memory_type_valid(self):
        for mt in MemoryType:
            assert validate_memory_type(mt.value) == mt.value

    def test_validate_memory_type_enum_instance(self):
        assert validate_memory_type(MemoryType.FACTS) == "facts"

    def test_validate_memory_type_invalid(self):
        with pytest.raises(ValueError, match="Invalid memory_type"):
            validate_memory_type("nonexistent")

    def test_validate_memory_category_valid(self):
        for mc in MemoryCategory:
            assert validate_memory_category(mc.value) == mc.value

    def test_validate_memory_category_invalid(self):
        with pytest.raises(ValueError, match="Invalid memory_category"):
            validate_memory_category("nonexistent")

    def test_validate_memory_category_enum_instance(self):
        assert validate_memory_category(MemoryCategory.EPISODIC) == "episodic"


# =============================================================================
# MemoryNamespaceBuilder tests
# =============================================================================


class TestMemoryNamespaceBuilder:

    def test_semantic_always_shared(self):
        """Semantic namespace never includes agent_name."""
        builder = MemoryNamespaceBuilder(agent_name="my_agent")
        ns = builder.namespace_for("user-1", MemoryCategory.SEMANTIC)
        assert ns == ("user-1", "semantic")

    def test_episodic_with_agent(self):
        builder = MemoryNamespaceBuilder(agent_name="browser_agent")
        ns = builder.namespace_for("user-1", MemoryCategory.EPISODIC)
        assert ns == ("user-1", "episodic", "browser_agent")

    def test_episodic_without_agent(self):
        builder = MemoryNamespaceBuilder(agent_name=None)
        ns = builder.namespace_for("user-1", MemoryCategory.EPISODIC)
        assert ns == ("user-1", "episodic")

    def test_procedural_with_agent(self):
        builder = MemoryNamespaceBuilder(agent_name="deep_planner")
        ns = builder.namespace_for("user-1", MemoryCategory.PROCEDURAL)
        assert ns == ("user-1", "procedural", "deep_planner")

    def test_procedural_without_agent(self):
        builder = MemoryNamespaceBuilder(agent_name=None)
        ns = builder.namespace_for("user-1", MemoryCategory.PROCEDURAL)
        assert ns == ("user-1", "procedural")

    def test_namespace_for_type_routes_correctly(self):
        builder = MemoryNamespaceBuilder(agent_name="agent_x")

        # preferences → semantic (shared)
        ns = builder.namespace_for_type("user-1", "preferences")
        assert ns == ("user-1", "semantic")

        # instructions → procedural (per-agent)
        ns = builder.namespace_for_type("user-1", "instructions")
        assert ns == ("user-1", "procedural", "agent_x")

        # context → episodic (per-agent)
        ns = builder.namespace_for_type("user-1", "context")
        assert ns == ("user-1", "episodic", "agent_x")

    def test_legacy_namespace(self):
        builder = MemoryNamespaceBuilder(agent_name="test")
        ns = builder.legacy_namespace("user-99")
        assert ns == ("user-99", "memories")

    def test_user_id_converted_to_string(self):
        builder = MemoryNamespaceBuilder()
        ns = builder.namespace_for(123, MemoryCategory.SEMANTIC)
        assert ns == ("123", "semantic")


# =============================================================================
# MemoryData auto-category tests
# =============================================================================


class TestMemoryData:
    def test_auto_resolves_category(self):
        data = MemoryData(content="test", memory_type="preferences")
        assert data.memory_category == "semantic"

    def test_auto_resolves_episodic_category(self):
        data = MemoryData(content="test", memory_type="session_summary")
        assert data.memory_category == "episodic"

    def test_explicit_category_preserved(self):
        data = MemoryData(
            content="test",
            memory_type="facts",
            memory_category="episodic",
        )
        assert data.memory_category == "episodic"


# =============================================================================
# MemoryStoreDocument.memory_category tests
# =============================================================================


class TestMemoryStoreDocument:
    def test_memory_category_property(self):
        doc = MemoryStoreDocument(
            id="1",
            value={"content": "test", "memory_type": "instructions"},
            metadata={},
        )
        assert doc.memory_category == "procedural"

    def test_memory_category_none_when_no_type(self):
        doc = MemoryStoreDocument(id="1", value={"content": "test"}, metadata={})
        assert doc.memory_category is None


# =============================================================================
# AgentMemoryStoreService CoALA tests
# =============================================================================


def _make_mock_store():
    """Create a mock store with put/search/delete."""
    store = MagicMock()
    store.put = MagicMock()
    store.delete = MagicMock()
    store.search = MagicMock(return_value=[])
    return store


def _real_store() -> InMemoryStore:
    """An in-memory store whose search embeds: Python or not, and the rest."""

    def embed(texts):
        return [[1.0 if "python" in text.lower() else 0.0, 0.5] for text in texts]

    return InMemoryStore(index={"embed": embed, "dims": 2, "fields": ["content"]})


class TestAgentMemoryStoreServiceCoALA:

    @pytest.fixture
    def store(self):
        return _make_mock_store()

    @pytest.fixture
    def service(self, store):
        return AgentMemoryStoreService(
            store=store,
            base_namespace=("agent_memory",),
            agent_name="test_agent",
        )

    async def test_save_routes_to_semantic_namespace(self, service, store):
        """Saving preferences should route to semantic namespace."""
        await service.save_memory(
            user_id="u1",
            content="likes bullet points",
            memory_type="preferences",
        )
        call_args = store.put.call_args
        namespace = call_args[0][0]
        assert namespace == ("u1", "semantic")

    async def test_save_routes_to_episodic_namespace(self, service, store):
        """Saving session_summary should route to episodic + agent_name."""
        await service.save_memory(
            user_id="u1",
            content="Discussed Q1 plan",
            memory_type="session_summary",
        )
        call_args = store.put.call_args
        namespace = call_args[0][0]
        assert namespace == ("u1", "episodic", "test_agent")

    async def test_save_routes_to_procedural_namespace(self, service, store):
        """Saving instructions should route to procedural + agent_name."""
        await service.save_memory(
            user_id="u1",
            content="Always run tests",
            memory_type="instructions",
        )
        call_args = store.put.call_args
        namespace = call_args[0][0]
        assert namespace == ("u1", "procedural", "test_agent")

    async def test_save_explicit_category_override(self, service, store):
        """Explicit category should override auto-resolution."""
        await service.save_memory(
            user_id="u1",
            content="Manual override",
            memory_type="general",
            category="episodic",
        )
        call_args = store.put.call_args
        namespace = call_args[0][0]
        # general normally goes to semantic, but explicit override to episodic
        assert namespace == ("u1", "episodic", "test_agent")

    async def test_save_stores_category_in_data(self, service, store):
        """Saved data should include memory_category field."""
        await service.save_memory(
            user_id="u1",
            content="test",
            memory_type="facts",
        )
        call_args = store.put.call_args
        value = call_args[0][2]
        assert value["memory_category"] == "semantic"

    @pytest.fixture
    def real(self):
        """The same service on a real store, whose search embeds."""
        return AgentMemoryStoreService(
            store=_real_store(),
            base_namespace=("agent_memory",),
            agent_name="test_agent",
        )

    async def test_recall_all_categories(self, real):
        """Recall without category searches the namespace of every category."""
        for memory_type in ("facts", "session_summary", "instructions"):
            await real.save_memory("u1", f"a {memory_type}", memory_type)

        docs = await real.recall_memories(user_id="u1", query="test")

        assert sorted(d.namespace for d in docs) == [
            ("u1", "episodic", "test_agent"),
            ("u1", "procedural", "test_agent"),
            ("u1", "semantic"),
        ]

    async def test_recall_single_category(self, real):
        """Recall with category searches only that namespace."""
        await real.save_memory("u1", "a fact", "facts")
        await real.save_memory("u1", "a rule", "instructions")

        docs = await real.recall_memories(
            user_id="u1", query="test", category="semantic"
        )

        assert [d.namespace for d in docs] == [("u1", "semantic")]

    async def test_recall_auto_resolves_category_from_type(self, real):
        """Recall with memory_type should auto-resolve category."""
        await real.save_memory("u1", "a fact", "facts")
        await real.save_memory("u1", "a rule", "instructions")

        docs = await real.recall_memories(
            user_id="u1", query="test", memory_type="instructions"
        )

        assert [d.namespace for d in docs] == [("u1", "procedural", "test_agent")]

    async def test_recall_merges_results_by_score(self, real):
        """Multi-namespace recall merges and sorts by score."""
        await real.save_memory("u1", "a fact", "facts")
        await real.save_memory("u1", "python session", "context")
        await real.save_memory("u1", "Python rules", "instructions")

        docs = await real.recall_memories(user_id="u1", query="python", k=2)

        assert sorted(d.content for d in docs) == ["Python rules", "python session"]

    async def test_delete_finds_the_namespace(self, real):
        """Delete without category deletes where the memory lies."""
        mid = await real.save_memory("u1", "a rule", "instructions")

        assert await real.delete_memory(user_id="u1", memory_id=mid) is True
        assert await real.recall_memories("u1", "rule") == []

    async def test_delete_with_category(self, real):
        mid = await real.save_memory("u1", "a fact", "facts")

        assert await real.delete_memory("u1", mid, category="semantic") is True
        assert await real.recall_memories("u1", "fact") == []


# =============================================================================
# Service without agent_name (shared episodic/procedural)
# =============================================================================


class TestAgentMemoryStoreServiceShared:

    @pytest.fixture
    def store(self):
        return _make_mock_store()

    @pytest.fixture
    def service(self, store):
        return AgentMemoryStoreService(
            store=store,
            base_namespace=("agent_memory",),
            agent_name=None,
        )

    async def test_episodic_without_agent_shared(self, service, store):
        await service.save_memory(
            user_id="u1",
            content="session note",
            memory_type="session_summary",
        )
        namespace = store.put.call_args[0][0]
        assert namespace == ("u1", "episodic")

    async def test_procedural_without_agent_shared(self, service, store):
        await service.save_memory(
            user_id="u1",
            content="always lint",
            memory_type="workflow",
        )
        namespace = store.put.call_args[0][0]
        assert namespace == ("u1", "procedural")


# =============================================================================
# format_memory_types_for_description tests
# =============================================================================


class TestFormatDescription:
    def test_includes_all_types(self):
        desc = format_memory_types_for_description()
        for mt in MemoryType:
            assert mt.value in desc

    def test_includes_category_headers(self):
        desc = format_memory_types_for_description()
        assert "[SEMANTIC]" in desc
        assert "[EPISODIC]" in desc
        assert "[PROCEDURAL]" in desc


# =============================================================================
# format_context_for_prompt CoALA structure tests
# =============================================================================


class TestFormatContextCoALA:

    @pytest.fixture
    def store(self):
        return _make_mock_store()

    @pytest.fixture
    def service(self, store):
        return AgentMemoryStoreService(
            store=store,
            base_namespace=("agent_memory",),
            agent_name="test_agent",
        )

    async def test_empty_results_returns_empty_string(self, service, store):
        store.search.return_value = []
        result = await service.format_context_for_prompt(user_id="u1")
        assert result == ""

    async def test_coala_xml_tags_present(self):
        """Output should contain CoALA XML section tags."""
        service = AgentMemoryStoreService(store=_real_store(), agent_name="test_agent")
        await service.save_memory("u1", "likes Python", "preferences")

        result = await service.format_context_for_prompt(
            user_id="u1",
            query="preferences",
        )
        assert "<semantic_memory>" in result
        assert "</semantic_memory>" in result
        assert "likes Python" in result

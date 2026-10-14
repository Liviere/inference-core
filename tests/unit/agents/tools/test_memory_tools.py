"""Tests for CoALA memory tools: Save, Recall, Update, Delete, factory.

Covers: _run happy path, _arun async path, error handling (ValueError),
_format_results with various payloads, get_memory_tools factory.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_core.agents.tools.memory_tools import (
    DeleteMemoryStoreTool,
    RecallMemoryStoreTool,
    SaveMemoryStoreTool,
    UpdateMemoryStoreTool,
    generate_memory_tools_system_instructions,
    get_memory_tools,
)
from inference_core.services.agent_memory_service import (
    DELETED,
    NOT_FOUND,
    READ_ONLY,
    UPDATED,
    AgentMemoryStoreService,
    MemorySpace,
    MemoryType,
    MemoryWriteOutcome,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memory_service():
    """Provide an AsyncMock AgentMemoryStoreService."""
    service = AsyncMock()
    service.spaces = ()
    service.space_for_write = MagicMock(return_value=MemorySpace(key="shared"))
    service.save_memory.return_value = "mem-12345678-abcd-1234"
    service.recall_memories.return_value = []
    service.delete_memory.return_value = True
    service.update_memory.return_value = MemoryWriteOutcome(UPDATED, "mem-abc")
    service.remove_memory.return_value = MemoryWriteOutcome(DELETED, "mem-del")
    return service


# ===========================================================================
# SaveMemoryStoreTool
# ===========================================================================


class TestSaveMemoryStoreTool:
    """Test SaveMemoryStoreTool._run and _arun."""

    def test_run_happy_path(self, mock_memory_service):
        """_run saves memory and returns success message."""
        tool = SaveMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
            session_id="sess-1",
        )
        result = tool._run(
            content="User likes Python", memory_type=MemoryType.PREFERENCES
        )
        assert "✓ Memory saved" in result
        assert "mem-1234" in result
        mock_memory_service.save_memory.assert_awaited_once()

    def test_run_invalid_memory_type(self, mock_memory_service):
        """_run returns error for invalid memory type."""
        tool = SaveMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._run(content="test", memory_type="invalid_type_xyz")
        assert "✗" in result

    async def test_arun_happy_path(self, mock_memory_service):
        """_arun awaits save_memory directly."""
        tool = SaveMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
            session_id="sess-1",
        )
        result = await tool._arun(content="Remember this")
        assert "✓ Memory saved" in result
        mock_memory_service.save_memory.assert_awaited_once()

    async def test_arun_invalid_type(self, mock_memory_service):
        """_arun returns error for invalid memory type."""
        tool = SaveMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = await tool._arun(content="test", memory_type="bogus")
        assert "✗" in result

    def test_run_with_category_override(self, mock_memory_service):
        """_run respects explicit category override."""
        tool = SaveMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._run(content="test", category="procedural")
        assert "procedural" in result


class TestSavingToASpace:
    """The save tool on a real service with several spaces."""

    @pytest.fixture
    def service(self):
        from langgraph.store.memory import InMemoryStore

        return AgentMemoryStoreService(
            store=InMemoryStore(),
            spaces=[
                MemorySpace(key="general", description="About the user."),
                MemorySpace(key="private", path=("agent:a1",), description="Own work."),
                MemorySpace(key="team", path=("space:t1",), writable=False),
            ],
            default_write="private",
        )

    async def test_a_memory_goes_to_the_space_named(self, service):
        tool = SaveMemoryStoreTool(memory_service=service, user_id="u1")

        result = await tool._arun(content="Likes tea", space="general")

        assert "space: general" in result
        assert service.store.search(("u1", "semantic"))

    async def test_without_a_name_it_goes_to_the_default_space(self, service):
        tool = SaveMemoryStoreTool(memory_service=service, user_id="u1")

        result = await tool._arun(content="Check the totals twice", memory_type="skill")

        assert "space: private" in result
        assert service.store.search(("u1", "agent:a1", "procedural"))

    async def test_a_space_read_only_is_refused_in_words(self, service):
        tool = SaveMemoryStoreTool(memory_service=service, user_id="u1")

        result = await tool._arun(content="Team fact", space="team")

        assert result.startswith("✗")
        assert "general" in result and "private" in result
        assert not service.store.search(("u1", "space:t1", "semantic"))

    def test_the_tool_lists_the_spaces_it_can_save_to(self, service):
        [save] = get_memory_tools(service, user_id="u1", include_tools=["save_memory_store"])

        assert "- general: About the user." in save.description
        assert "- private: Own work." in save.description
        assert "- team" not in save.description


# ===========================================================================
# RecallMemoryStoreTool
# ===========================================================================


class TestRecallMemoryStoreTool:
    """Test RecallMemoryStoreTool._run, _arun, and _format_results."""

    @patch("inference_core.agents.tools.memory_tools.run_async_safely")
    def test_run_no_results(self, mock_run_async, mock_memory_service):
        """_run returns 'no relevant memories' when empty."""
        mock_run_async.return_value = []
        tool = RecallMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._run(query="test query")
        assert "No relevant memories" in result

    @patch("inference_core.agents.tools.memory_tools.run_async_safely")
    def test_run_with_results(self, mock_run_async, mock_memory_service):
        """_run formats recalled memories."""
        mem = MagicMock()
        mem.content = "User likes cats"
        mem.memory_type = "preferences"
        mem.score = 0.95
        mem.topic = "pets"
        mem.id = "mem-abc"
        mem.created_at = None
        mem.memory_category = "semantic"
        mock_run_async.return_value = [mem]

        tool = RecallMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._run(query="what does user like")
        assert "1 relevant memories" in result
        assert "User likes cats" in result
        assert "0.95" in result

    def test_format_results_empty(self, mock_memory_service):
        """_format_results returns message for empty list."""
        tool = RecallMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        assert tool._format_results([]) == "No relevant memories found."

    def test_format_results_with_timestamp(self, mock_memory_service):
        """_format_results includes created_at when available."""
        from datetime import datetime

        mem = MagicMock()
        mem.content = "Test content"
        mem.memory_type = "facts"
        mem.score = None
        mem.topic = None
        mem.id = "mem-xyz"
        mem.created_at = datetime(2026, 1, 15, 10, 30)
        mem.memory_category = None

        tool = RecallMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._format_results([mem])
        assert "2026-01-15 10:30" in result

    def test_format_results_with_string_timestamp(self, mock_memory_service):
        """_format_results handles string-type created_at."""
        mem = MagicMock()
        mem.content = "Content"
        mem.memory_type = "general"
        mem.score = None
        mem.topic = None
        mem.id = "mem-1"
        mem.created_at = "2026-01-15T10:30:00Z"
        mem.memory_category = "semantic"

        tool = RecallMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._format_results([mem])
        assert "2026-01-15T10:30" in result

    async def test_arun_happy_path(self, mock_memory_service):
        """_arun awaits recall_memories directly."""
        mock_memory_service.recall_memories.return_value = []
        tool = RecallMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = await tool._arun(query="test")
        assert "No relevant memories" in result

    @patch("inference_core.agents.tools.memory_tools.run_async_safely")
    def test_run_invalid_memory_type(self, mock_run_async, mock_memory_service):
        """_run returns error for invalid memory_type filter."""
        tool = RecallMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._run(query="test", memory_type="invalid")
        assert "✗" in result


# ===========================================================================
# UpdateMemoryStoreTool
# ===========================================================================


class TestUpdateMemoryStoreTool:
    """Test UpdateMemoryStoreTool._run and _arun."""

    def test_run_happy_path(self, mock_memory_service):
        """_run updates memory and returns success."""
        tool = UpdateMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._run(memory_id="mem-abc", content="Updated content")
        assert "✓ Memory updated" in result

    def test_run_invalid_type(self, mock_memory_service):
        """_run returns error for invalid memory type."""
        tool = UpdateMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._run(memory_id="mem-1", content="test", memory_type="bad")
        assert "✗" in result

    async def test_arun_happy_path(self, mock_memory_service):
        """_arun changes the memory where it lies, not by writing it again."""
        tool = UpdateMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = await tool._arun(memory_id="mem-abc", content="New content")
        assert "✓ Memory updated" in result
        mock_memory_service.update_memory.assert_awaited_once()
        mock_memory_service.save_memory.assert_not_called()

    async def test_a_memory_the_agent_only_reads_is_not_changed(
        self, mock_memory_service
    ):
        mock_memory_service.update_memory.return_value = MemoryWriteOutcome(
            READ_ONLY, "mem-abc", MemorySpace(key="team", writable=False)
        )
        tool = UpdateMemoryStoreTool(
            memory_service=mock_memory_service, user_id="user-1"
        )

        result = await tool._arun(memory_id="mem-abc", content="New content")

        assert result.startswith("✗")
        assert "'team', which you can only read" in result

    async def test_an_unknown_id_is_not_found(self, mock_memory_service):
        mock_memory_service.update_memory.return_value = MemoryWriteOutcome(
            NOT_FOUND, "nope"
        )
        tool = UpdateMemoryStoreTool(
            memory_service=mock_memory_service, user_id="user-1"
        )

        result = await tool._arun(memory_id="nope", content="x")

        assert result == "✗ Memory nope was not found."


# ===========================================================================
# DeleteMemoryStoreTool
# ===========================================================================


class TestDeleteMemoryStoreTool:
    """Test DeleteMemoryStoreTool._run and _arun."""

    def test_run_success(self, mock_memory_service):
        """_run returns success message when deletion succeeds."""
        tool = DeleteMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._run(memory_id="mem-to-delete")
        assert "✓ Memory deleted" in result

    def test_run_not_found(self, mock_memory_service):
        """_run returns failure when the memory is nowhere."""
        mock_memory_service.remove_memory.return_value = MemoryWriteOutcome(
            NOT_FOUND, "nonexistent"
        )
        tool = DeleteMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._run(memory_id="nonexistent")
        assert "✗" in result
        assert "not found" in result

    async def test_arun_success(self, mock_memory_service):
        """_arun deletes the memory where it lies."""
        tool = DeleteMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = await tool._arun(memory_id="mem-del")
        assert "✓ Memory deleted" in result
        mock_memory_service.remove_memory.assert_awaited_once()

    async def test_arun_not_found(self, mock_memory_service):
        """_arun returns failure when not found."""
        mock_memory_service.remove_memory.return_value = MemoryWriteOutcome(
            NOT_FOUND, "nope"
        )
        tool = DeleteMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = await tool._arun(memory_id="nope")
        assert "✗" in result


# ===========================================================================
# get_memory_tools factory
# ===========================================================================


class TestGetMemoryTools:
    """Test get_memory_tools factory function."""

    def test_returns_four_tools(self, mock_memory_service):
        """Factory returns list of 4 tools (save, recall, update, delete)."""
        tools = get_memory_tools(
            memory_service=mock_memory_service,
            user_id="user-1",
            session_id="sess-1",
        )
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert "save_memory_store" in names
        assert "recall_memories_store" in names
        assert "update_memory_store" in names
        assert "delete_memory_store" in names

    def test_custom_max_recall(self, mock_memory_service):
        """Factory respects max_recall_results parameter."""
        tools = get_memory_tools(
            memory_service=mock_memory_service,
            user_id="user-1",
            max_recall_results=10,
        )
        recall_tool = [t for t in tools if t.name == "recall_memories_store"][0]
        assert recall_tool.default_max_results == 10

    def test_factory_accepts_none_user_id(self, mock_memory_service):
        """Factory creates tools with user_id=None for Agent Server."""
        tools = get_memory_tools(
            memory_service=mock_memory_service,
            user_id=None,
        )
        assert len(tools) == 4
        for t in tools:
            assert t.user_id is None


# ===========================================================================
# Deferred user_id resolution (Agent Server)
# ===========================================================================


class TestDeferredUserId:
    """Verify tools resolve user_id from context var when user_id=None."""

    def test_save_tool_resolves_from_ctx(self, mock_memory_service):
        """SaveMemoryStoreTool._run resolves user_id from context var."""
        import uuid

        uid = uuid.uuid4()
        tool = SaveMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id=None,
        )
        with patch(
            "inference_core.agents.tools.memory_tools._resolve_tool_user_id",
            return_value=str(uid),
        ) as mock_resolve:
            result = tool._run(content="test data")
        mock_resolve.assert_called_once_with(None)
        assert "✓ Memory saved" in result

    async def test_save_tool_arun_resolves_from_ctx(self, mock_memory_service):
        """SaveMemoryStoreTool._arun resolves user_id from context var."""
        tool = SaveMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id=None,
        )
        with patch(
            "inference_core.agents.tools.memory_tools._resolve_tool_user_id",
            return_value="ctx-user",
        ):
            result = await tool._arun(content="async test")
        assert "✓ Memory saved" in result
        call_kwargs = mock_memory_service.save_memory.call_args[1]
        assert call_kwargs["user_id"] == "ctx-user"

    @patch("inference_core.agents.tools.memory_tools.run_async_safely")
    def test_recall_tool_resolves_from_ctx(self, mock_run_async, mock_memory_service):
        """RecallMemoryStoreTool._run resolves user_id from context var."""
        mock_run_async.return_value = []
        tool = RecallMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id=None,
        )
        with patch(
            "inference_core.agents.tools.memory_tools._resolve_tool_user_id",
            return_value="ctx-user",
        ):
            result = tool._run(query="what do I like?")
        assert "No relevant memories" in result

    def test_delete_tool_resolves_from_ctx(self, mock_memory_service):
        """DeleteMemoryStoreTool._run resolves user_id from context var."""
        tool = DeleteMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id=None,
        )
        with patch(
            "inference_core.agents.tools.memory_tools._resolve_tool_user_id",
            return_value="ctx-user",
        ):
            result = tool._run(memory_id="mem-123")
        assert "✓ Memory deleted" in result

    def test_resolve_raises_when_no_user_id(self, mock_memory_service):
        """_resolve_tool_user_id raises RuntimeError when both sources empty."""
        from inference_core.agents.tools.memory_tools import _resolve_tool_user_id

        with patch(
            "inference_core.agents.tools.memory_tools._resolve_tool_user_id",
            wraps=_resolve_tool_user_id,
        ):
            with patch(
                "inference_core.agents.middleware._runtime_context.get_user_id",
                return_value=None,
            ):
                with pytest.raises(RuntimeError, match="requires user_id"):
                    _resolve_tool_user_id(None)


# ===========================================================================
# get_memory_tools — include_tools filtering
# ===========================================================================


class TestGetMemoryToolsFiltering:
    """Verify include_tools parameter filters returned tools."""

    def test_include_tools_none_returns_all(self, mock_memory_service):
        """include_tools=None returns all 4 tools (backward compat)."""
        tools = get_memory_tools(
            memory_service=mock_memory_service,
            user_id="u1",
            include_tools=None,
        )
        assert len(tools) == 4

    def test_include_tools_empty_list_returns_empty(self, mock_memory_service):
        """include_tools=[] disables all model-facing tools."""
        tools = get_memory_tools(
            memory_service=mock_memory_service,
            user_id="u1",
            include_tools=[],
        )
        assert tools == []

    def test_include_tools_subset(self, mock_memory_service):
        """include_tools with subset returns only matching tools."""
        tools = get_memory_tools(
            memory_service=mock_memory_service,
            user_id="u1",
            include_tools=["save_memory_store", "recall_memories_store"],
        )
        names = {t.name for t in tools}
        assert names == {"save_memory_store", "recall_memories_store"}
        assert len(tools) == 2

    def test_include_tools_single(self, mock_memory_service):
        """include_tools with single tool returns only that tool."""
        tools = get_memory_tools(
            memory_service=mock_memory_service,
            user_id="u1",
            include_tools=["delete_memory_store"],
        )
        assert len(tools) == 1
        assert tools[0].name == "delete_memory_store"

    def test_include_tools_unknown_name_ignored(self, mock_memory_service):
        """Unknown tool names in include_tools are silently ignored."""
        tools = get_memory_tools(
            memory_service=mock_memory_service,
            user_id="u1",
            include_tools=["save_memory_store", "nonexistent_tool"],
        )
        assert len(tools) == 1
        assert tools[0].name == "save_memory_store"


# ===========================================================================
# generate_memory_tools_system_instructions — active_tool_names filtering
# ===========================================================================


class TestGenerateMemoryToolsInstructions:
    """Verify per-tool instruction generation."""

    def test_none_generates_all_sections(self):
        """active_tool_names=None generates instructions for all 4 tools."""
        result = generate_memory_tools_system_instructions(active_tool_names=None)
        assert "save_memory_store" in result
        assert "recall_memories_store" in result
        assert "update_memory_store" in result
        assert "delete_memory_store" in result
        assert "Mandatory save triggers" in result

    def test_empty_list_returns_empty_string(self):
        """active_tool_names=[] returns empty string."""
        result = generate_memory_tools_system_instructions(active_tool_names=[])
        assert result == ""

    def test_subset_only_includes_active_tools(self):
        """Only sections for active tools are generated."""
        result = generate_memory_tools_system_instructions(
            active_tool_names=["recall_memories_store", "delete_memory_store"]
        )
        assert "#### recall_memories_store" in result
        assert "#### delete_memory_store" in result
        # save section header must NOT appear
        assert "#### save_memory_store" not in result
        assert "Mandatory save triggers" not in result

    def test_save_only_includes_triggers(self):
        """When save_memory_store is active, save triggers are included."""
        result = generate_memory_tools_system_instructions(
            active_tool_names=["save_memory_store"]
        )
        assert "save_memory_store" in result
        assert "Mandatory save triggers" in result

    def test_recall_only_no_triggers(self):
        """When only recall is active, save triggers are NOT included."""
        result = generate_memory_tools_system_instructions(
            active_tool_names=["recall_memories_store"]
        )
        assert "recall_memories_store" in result
        assert "Mandatory save triggers" not in result

    def test_unknown_names_filtered_out(self):
        """Unknown tool names are ignored; no crash."""
        result = generate_memory_tools_system_instructions(
            active_tool_names=["nonexistent_tool"]
        )
        assert result == ""

    def test_nothing_is_said_about_who_shares_a_category(self):
        """It said semantic memory was shared across all agents and the rest
        scoped per agent, which held for one layout only."""
        result = generate_memory_tools_system_instructions(active_tool_names=None)
        assert "Shared across all agents" not in result
        assert "Scoped per agent" not in result
        assert "### Memory spaces" not in result

    def test_the_spaces_and_where_to_save_are_described(self):
        result = generate_memory_tools_system_instructions(
            active_tool_names=None,
            spaces=[
                MemorySpace(key="general", label="General", description="About the user."),
                MemorySpace(key="private", label="Own", description="Own work."),
                MemorySpace(key="team", label="Team", writable=False),
            ],
            default_write="private",
        )
        assert "### Memory spaces" in result
        assert "- `general` (General, writable). About the user." in result
        assert "- `team` (Team, read only)." in result
        assert "Without `space`, a memory goes to `private`." in result

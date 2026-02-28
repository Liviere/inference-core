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
    get_memory_tools,
)
from inference_core.services.agent_memory_service import MemoryType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memory_service():
    """Provide an AsyncMock AgentMemoryStoreService."""
    service = AsyncMock()
    service.save_memory.return_value = "mem-12345678-abcd-1234"
    service.recall_memories.return_value = []
    service.delete_memory.return_value = True
    return service


# ===========================================================================
# SaveMemoryStoreTool
# ===========================================================================


class TestSaveMemoryStoreTool:
    """Test SaveMemoryStoreTool._run and _arun."""

    @patch("inference_core.agents.tools.memory_tools.run_async_safely")
    def test_run_happy_path(self, mock_run_async, mock_memory_service):
        """_run saves memory and returns success message."""
        mock_run_async.return_value = "mem-12345678-abcd-1234"
        tool = SaveMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
            session_id="sess-1",
        )
        result = tool._run(content="User likes Python", memory_type=MemoryType.PREFERENCES)
        assert "✓ Memory saved" in result
        assert "mem-1234" in result
        mock_run_async.assert_called_once()

    @patch("inference_core.agents.tools.memory_tools.run_async_safely")
    def test_run_invalid_memory_type(self, mock_run_async, mock_memory_service):
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

    @patch("inference_core.agents.tools.memory_tools.run_async_safely")
    def test_run_with_category_override(self, mock_run_async, mock_memory_service):
        """_run respects explicit category override."""
        mock_run_async.return_value = "mem-12345678-1234-5678"
        tool = SaveMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._run(content="test", category="procedural")
        assert "procedural" in result


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

    @patch("inference_core.agents.tools.memory_tools.run_async_safely")
    def test_run_happy_path(self, mock_run_async, mock_memory_service):
        """_run updates memory and returns success."""
        mock_run_async.return_value = "mem-updated-id"
        tool = UpdateMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._run(memory_id="mem-abc", content="Updated content")
        assert "✓ Memory updated" in result

    @patch("inference_core.agents.tools.memory_tools.run_async_safely")
    def test_run_invalid_type(self, mock_run_async, mock_memory_service):
        """_run returns error for invalid memory type."""
        tool = UpdateMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._run(memory_id="mem-1", content="test", memory_type="bad")
        assert "✗" in result

    async def test_arun_happy_path(self, mock_memory_service):
        """_arun awaits save_memory with memory_id."""
        tool = UpdateMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = await tool._arun(memory_id="mem-abc", content="New content")
        assert "✓ Memory updated" in result
        mock_memory_service.save_memory.assert_awaited_once()


# ===========================================================================
# DeleteMemoryStoreTool
# ===========================================================================


class TestDeleteMemoryStoreTool:
    """Test DeleteMemoryStoreTool._run and _arun."""

    @patch("inference_core.agents.tools.memory_tools.run_async_safely")
    def test_run_success(self, mock_run_async, mock_memory_service):
        """_run returns success message when deletion succeeds."""
        mock_run_async.return_value = True
        tool = DeleteMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._run(memory_id="mem-to-delete")
        assert "✓ Memory deleted" in result

    @patch("inference_core.agents.tools.memory_tools.run_async_safely")
    def test_run_not_found(self, mock_run_async, mock_memory_service):
        """_run returns failure when delete returns False."""
        mock_run_async.return_value = False
        tool = DeleteMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = tool._run(memory_id="nonexistent")
        assert "✗" in result
        assert "not found" in result

    async def test_arun_success(self, mock_memory_service):
        """_arun awaits delete_memory directly."""
        tool = DeleteMemoryStoreTool(
            memory_service=mock_memory_service,
            user_id="user-1",
        )
        result = await tool._arun(memory_id="mem-del")
        assert "✓ Memory deleted" in result

    async def test_arun_not_found(self, mock_memory_service):
        """_arun returns failure when not found."""
        mock_memory_service.delete_memory.return_value = False
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

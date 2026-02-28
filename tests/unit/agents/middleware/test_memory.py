"""Tests for MemoryMiddleware (CoALA Architecture).

Covers lifecycle hooks (before_agent, wrap_model_call) and helper methods
(_extract_user_input, _recall_and_format).
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from inference_core.agents.middleware.memory import (
    MemoryMiddleware,
    MemoryState,
    _MemoryMiddlewareContext,
    create_memory_middleware,
)
from inference_core.services.agent_memory_service import MemoryCategory, MemoryType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memory_service():
    """Mock AgentMemoryStoreService with async methods."""
    service = MagicMock()
    return service


@pytest.fixture
def middleware(mock_memory_service):
    """MemoryMiddleware wired to mocked memory service."""
    return MemoryMiddleware(
        memory_service=mock_memory_service,
        user_id="user-123",
        auto_recall=True,
        max_recall_results=5,
    )


# ---------------------------------------------------------------------------
# _extract_user_input
# ---------------------------------------------------------------------------


class TestExtractUserInput:
    """Verify extraction of last user message from agent state."""

    def test_human_message_with_type_attr(self, middleware):
        """Finds message with type='human'."""
        msg = MagicMock()
        msg.type = "human"
        msg.content = "Hello agent"

        state = MemoryState(messages=[msg])
        assert middleware._extract_user_input(state) == "Hello agent"

    def test_user_message_with_role_attr(self, middleware):
        """Finds message with role='user'."""
        msg = MagicMock(spec=[])
        msg.role = "user"
        msg.content = "What is 2+2?"

        state = MemoryState(messages=[msg])
        assert middleware._extract_user_input(state) == "What is 2+2?"

    def test_dict_message_with_type(self, middleware):
        """Finds dict message with type='human'."""
        state = MemoryState(
            messages=[{"type": "human", "content": "Dict message"}]
        )
        assert middleware._extract_user_input(state) == "Dict message"

    def test_dict_message_with_role(self, middleware):
        """Finds dict message with role='user'."""
        state = MemoryState(
            messages=[{"role": "user", "content": "User role dict"}]
        )
        assert middleware._extract_user_input(state) == "User role dict"

    def test_picks_last_human_message(self, middleware):
        """When multiple messages exist, picks the last human one."""
        sys_msg = MagicMock()
        sys_msg.type = "system"

        human1 = MagicMock()
        human1.type = "human"
        human1.content = "first"

        ai_msg = MagicMock()
        ai_msg.type = "ai"

        human2 = MagicMock()
        human2.type = "human"
        human2.content = "second"

        state = MemoryState(messages=[sys_msg, human1, ai_msg, human2])
        assert middleware._extract_user_input(state) == "second"

    def test_empty_messages_returns_none(self, middleware):
        """Empty message list yields None."""
        state = MemoryState(messages=[])
        assert middleware._extract_user_input(state) is None

    def test_fallback_to_last_message_content(self, middleware):
        """When no human/user message found, falls back to last message."""
        msg = MagicMock(spec=[])
        msg.content = "fallback content"

        state = MemoryState(messages=[msg])
        assert middleware._extract_user_input(state) == "fallback content"


# ---------------------------------------------------------------------------
# before_agent
# ---------------------------------------------------------------------------


class TestBeforeAgent:
    """Verify before_agent recall, caching, and state updates."""

    def test_returns_none_when_auto_recall_disabled(self, mock_memory_service):
        """Skips recall if auto_recall=False."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            auto_recall=False,
        )
        state = MemoryState(messages=[MagicMock(type="human", content="hi")])
        result = mw.before_agent(state, MagicMock())

        assert result is None

    def test_returns_none_when_no_user_input(self, middleware):
        """Skips recall when no user input can be extracted."""
        state = MemoryState(messages=[])
        result = middleware.before_agent(state, MagicMock())

        assert result is None

    def test_returns_zero_recall_when_no_memories(self, middleware):
        """Returns count=0 when _recall_and_format returns empty context."""
        with patch.object(
            middleware, "_recall_and_format", return_value=("", {"latency_ms": 5.0})
        ):
            msg = MagicMock()
            msg.type = "human"
            msg.content = "test"
            state = MemoryState(messages=[msg])

            result = middleware.before_agent(state, MagicMock())

        assert result["memories_recalled"] == 0

    def test_returns_full_state_when_memories_found(self, middleware):
        """Returns complete state update with memory context and metrics."""
        metrics = {
            "count": 3,
            "latency_ms": 12.5,
            "types": ["preferences", "facts"],
            "categories": ["semantic"],
        }
        with patch.object(
            middleware,
            "_recall_and_format",
            return_value=("<memory>some context</memory>", metrics),
        ):
            msg = MagicMock()
            msg.type = "human"
            msg.content = "Tell me about X"
            state = MemoryState(messages=[msg])

            result = middleware.before_agent(state, MagicMock())

        assert result["memory_context"] == "<memory>some context</memory>"
        assert result["memories_recalled"] == 3
        assert result["memory_recall_latency_ms"] == 12.5
        assert result["memory_types_recalled"] == ["preferences", "facts"]
        assert result["memory_categories_recalled"] == ["semantic"]
        assert middleware._cached_memory_context is not None

    def test_handles_recall_error_gracefully(self, middleware):
        """Returns empty state on _recall_and_format exception."""
        with patch.object(
            middleware,
            "_recall_and_format",
            side_effect=Exception("DB down"),
        ):
            # This will be caught by the try/except in before_agent
            msg = MagicMock()
            msg.type = "human"
            msg.content = "test"
            state = MemoryState(messages=[msg])

            result = middleware.before_agent(state, MagicMock())

        assert result["memories_recalled"] == 0


# ---------------------------------------------------------------------------
# wrap_model_call
# ---------------------------------------------------------------------------


class TestWrapModelCall:
    """Verify system prompt injection of memory context."""

    def test_passthrough_when_no_cached_context(self, middleware):
        """Handler called unchanged when no memory context was cached."""
        middleware._cached_memory_context = None
        request = MagicMock()
        response = MagicMock()
        handler = MagicMock(return_value=response)

        result = middleware.wrap_model_call(request, handler)

        handler.assert_called_once_with(request)
        assert result is response

    def test_injects_memory_into_system_message(self, middleware):
        """Memory context is appended as content block to system message."""
        middleware._cached_memory_context = "User likes Python"

        # Prepare request with system message
        original_block = {"type": "text", "text": "You are a helpful assistant."}
        request = MagicMock()
        request.system_message.content_blocks = [original_block]

        captured_request = None

        def capture_handler(r):
            nonlocal captured_request
            captured_request = r
            return MagicMock()

        handler = capture_handler

        with patch(
            "inference_core.agents.middleware.memory.SystemMessage"
        ) as MockSysMsg:
            mock_sys_msg = MagicMock()
            MockSysMsg.return_value = mock_sys_msg
            request.override.return_value = MagicMock()

            middleware.wrap_model_call(request, handler)

            # Verify SystemMessage was created with memory block appended
            call_args = MockSysMsg.call_args
            new_content = call_args[1]["content"] if call_args[1] else call_args[0][0]

            # Should have original block + memory block
            assert len(new_content) == 2
            memory_block = new_content[1]
            assert "<user_memory_context>" in memory_block["text"]
            assert "User likes Python" in memory_block["text"]

    def test_falls_back_on_injection_error(self, middleware):
        """On error during injection, falls back to original request."""
        middleware._cached_memory_context = "some context"

        request = MagicMock()
        # Make content_blocks access raise to trigger the except branch
        request.system_message.content_blocks = None  # Will fail on list()

        response = MagicMock()
        handler = MagicMock(return_value=response)

        result = middleware.wrap_model_call(request, handler)

        # Handler should be called with original request
        handler.assert_called_once_with(request)
        assert result is response


# ---------------------------------------------------------------------------
# _recall_and_format
# ---------------------------------------------------------------------------


class TestRecallAndFormat:
    """Verify async memory recall via run_async_safely."""

    def test_returns_context_and_metrics(self, middleware, mock_memory_service):
        """Successful recall returns formatted context and metrics dict."""

        async def mock_format_context(**kwargs):
            return "<semantic>facts here</semantic>"

        async def mock_get_user_context(**kwargs):
            return {
                "preferences": [MagicMock(), MagicMock()],
                "facts": [MagicMock()],
            }

        mock_memory_service.format_context_for_prompt = mock_format_context
        mock_memory_service.get_user_context = mock_get_user_context

        with patch(
            "inference_core.agents.middleware.memory.run_async_safely"
        ) as mock_run:
            mock_run.return_value = (
                "<semantic>facts here</semantic>",
                {"preferences": [1, 2], "facts": [3]},
            )

            with patch(
                "inference_core.services.agent_memory_service.get_category_for_type",
                return_value=MemoryCategory.SEMANTIC,
            ):
                context, metrics = middleware._recall_and_format("Tell me about X")

        assert context == "<semantic>facts here</semantic>"
        assert metrics["count"] == 3  # 2 + 1
        assert "latency_ms" in metrics

    def test_returns_empty_on_timeout(self, middleware):
        """TimeoutError returns empty context and zero metrics."""
        with patch(
            "inference_core.agents.middleware.memory.run_async_safely",
            side_effect=TimeoutError("Memory recall timed out"),
        ):
            context, metrics = middleware._recall_and_format("query")

        assert context == ""
        assert metrics["count"] == 0

    def test_returns_empty_on_generic_error(self, middleware):
        """Generic exception returns empty context and zero metrics."""
        with patch(
            "inference_core.agents.middleware.memory.run_async_safely",
            side_effect=RuntimeError("connection refused"),
        ):
            context, metrics = middleware._recall_and_format("query")

        assert context == ""
        assert metrics["count"] == 0


# ---------------------------------------------------------------------------
# create_memory_middleware factory
# ---------------------------------------------------------------------------


class TestCreateMemoryMiddleware:
    """Verify factory function produces correctly configured middleware."""

    def test_creates_with_defaults(self, mock_memory_service):
        """Factory creates middleware with default CoALA categories."""
        mw = create_memory_middleware(
            memory_service=mock_memory_service,
            user_id="user-42",
        )

        assert mw.user_id == "user-42"
        assert mw.auto_recall is True
        assert mw.max_recall_results == 5
        assert MemoryCategory.SEMANTIC.value in mw.include_categories
        assert MemoryCategory.EPISODIC.value in mw.include_categories
        assert MemoryCategory.PROCEDURAL.value in mw.include_categories

    def test_creates_with_custom_categories(self, mock_memory_service):
        """Factory passes custom categories and types through."""
        mw = create_memory_middleware(
            memory_service=mock_memory_service,
            user_id="user-42",
            include_categories=["semantic"],
            include_memory_types=["facts"],
            max_recall_results=10,
        )

        assert mw.include_categories == ["semantic"]
        assert mw.include_memory_types == ["facts"]
        assert mw.max_recall_results == 10

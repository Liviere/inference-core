"""Tests for MemoryMiddleware (CoALA Architecture).

Covers lifecycle hooks (before_agent, wrap_model_call) and helper methods
(_extract_user_input, _recall_and_format).
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

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
        state = MemoryState(messages=[{"type": "human", "content": "Dict message"}])
        assert middleware._extract_user_input(state) == "Dict message"

    def test_dict_message_with_role(self, middleware):
        """Finds dict message with role='user'."""
        state = MemoryState(messages=[{"role": "user", "content": "User role dict"}])
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

    def test_creates_with_none_user_id(self, mock_memory_service):
        """Factory accepts user_id=None for Agent Server deferred resolution."""
        mw = create_memory_middleware(
            memory_service=mock_memory_service,
            user_id=None,
        )
        assert mw.user_id is None
        assert mw.auto_recall is True


# ---------------------------------------------------------------------------
# Deferred user_id resolution (Agent Server)
# ---------------------------------------------------------------------------


class TestDeferredUserId:
    """Verify _resolve_user_id falls back to runtime context var."""

    def test_returns_instance_user_id_when_set(self, mock_memory_service):
        """Uses self.user_id when it is not None."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="explicit-uid",
        )
        assert mw._resolve_user_id() == "explicit-uid"

    def test_falls_back_to_context_var(self, mock_memory_service):
        """Resolves from _runtime_context when self.user_id is None."""
        import uuid

        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id=None,
        )
        uid = uuid.uuid4()
        with patch(
            "inference_core.agents.middleware.memory._ctx_get_user_id",
            return_value=uid,
        ):
            assert mw._resolve_user_id() == str(uid)

    def test_returns_none_when_both_empty(self, mock_memory_service):
        """Returns None when neither instance attr nor context var is set."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id=None,
        )
        with patch(
            "inference_core.agents.middleware.memory._ctx_get_user_id",
            return_value=None,
        ):
            assert mw._resolve_user_id() is None

    def test_before_agent_skips_when_no_user_id(self, mock_memory_service):
        """before_agent returns None when user_id cannot be resolved."""
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id=None,
        )
        msg = MagicMock()
        msg.type = "human"
        msg.content = "hi"
        state = MemoryState(messages=[msg])

        with patch(
            "inference_core.agents.middleware.memory._ctx_get_user_id",
            return_value=None,
        ):
            result = mw.before_agent(state, MagicMock())

        assert result is None

    def test_before_agent_resolves_from_ctx(self, mock_memory_service):
        """before_agent uses context var user_id and passes it to _recall_and_format."""
        import uuid

        uid = uuid.uuid4()
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id=None,
        )
        msg = MagicMock()
        msg.type = "human"
        msg.content = "test query"
        state = MemoryState(messages=[msg])

        with (
            patch(
                "inference_core.agents.middleware.memory._ctx_get_user_id",
                return_value=uid,
            ),
            patch.object(
                mw,
                "_recall_and_format",
                return_value=(
                    "ctx",
                    {"count": 1, "latency_ms": 1, "types": [], "categories": []},
                ),
            ) as mock_recall,
        ):
            result = mw.before_agent(state, MagicMock())

        mock_recall.assert_called_once_with("test query", str(uid))
        assert result["memories_recalled"] == 1


# ---------------------------------------------------------------------------
# _arecall_and_format (async recall path)
# ---------------------------------------------------------------------------


class TestAsyncRecallAndFormat:
    """Verify async recall path for Agent Server context."""

    async def test_returns_context_and_metrics(self, mock_memory_service):
        """_arecall_and_format directly awaits memory service."""
        mock_memory_service.format_context_for_prompt = AsyncMock(
            return_value="<semantic>data</semantic>"
        )
        mock_memory_service.get_user_context = AsyncMock(
            return_value={"facts": [MagicMock(), MagicMock()]}
        )

        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u-async",
        )

        with patch(
            "inference_core.services.agent_memory_service.get_category_for_type",
            return_value=MemoryCategory.SEMANTIC,
        ):
            context, metrics = await mw._arecall_and_format("some query")

        assert context == "<semantic>data</semantic>"
        assert metrics["count"] == 2
        assert "latency_ms" in metrics

    async def test_returns_empty_on_error(self, mock_memory_service):
        """_arecall_and_format returns empty on exception."""
        mock_memory_service.format_context_for_prompt = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u-err",
        )

        context, metrics = await mw._arecall_and_format("query")
        assert context == ""
        assert metrics["count"] == 0

    async def test_uses_provided_user_id_over_self(self, mock_memory_service):
        """Explicit user_id param takes precedence over self.user_id."""
        mock_memory_service.format_context_for_prompt = AsyncMock(return_value="")
        mock_memory_service.get_user_context = AsyncMock(return_value={})

        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="self-uid",
        )
        await mw._arecall_and_format("q", user_id="override-uid")

        call_kwargs = mock_memory_service.format_context_for_prompt.call_args[1]
        assert call_kwargs["user_id"] == "override-uid"


# ---------------------------------------------------------------------------
# _has_memory_save_in_run
# ---------------------------------------------------------------------------


class TestHasMemorySaveInRun:
    """Verify detection of save_memory_store tool calls in run history."""

    def test_returns_false_when_no_messages(self, middleware):
        state = MemoryState(messages=[])
        assert middleware._has_memory_save_in_run(state) is False

    def test_detects_tool_result_message_by_name(self, middleware):
        """ToolMessage with name='save_memory_store' is detected."""
        msg = MagicMock()
        msg.name = "save_memory_store"
        msg.tool_calls = None
        state = MemoryState(messages=[msg])
        assert middleware._has_memory_save_in_run(state) is True

    def test_detects_ai_message_with_tool_call_dict(self, middleware):
        """AI message with tool_calls list of dicts is detected."""
        msg = MagicMock()
        msg.name = None
        msg.tool_calls = [{"name": "save_memory_store", "args": {}}]
        state = MemoryState(messages=[msg])
        assert middleware._has_memory_save_in_run(state) is True

    def test_detects_ai_message_with_tool_call_object(self, middleware):
        """AI message with tool_calls list of objects is detected."""
        tc = MagicMock()
        tc.name = "save_memory_store"
        msg = MagicMock()
        msg.name = None
        msg.tool_calls = [tc]
        state = MemoryState(messages=[msg])
        assert middleware._has_memory_save_in_run(state) is True

    def test_returns_false_for_other_tool_names(self, middleware):
        """Other tool names are not confused with save_memory_store."""
        msg = MagicMock()
        msg.name = "recall_memories_store"
        msg.tool_calls = [{"name": "recall_memories_store", "args": {}}]
        state = MemoryState(messages=[msg])
        assert middleware._has_memory_save_in_run(state) is False

    def test_returns_false_when_tool_calls_is_none(self, middleware):
        msg = MagicMock()
        msg.name = None
        msg.tool_calls = None
        state = MemoryState(messages=[msg])
        assert middleware._has_memory_save_in_run(state) is False


# ---------------------------------------------------------------------------
# Helpers for message construction
# ---------------------------------------------------------------------------


def _make_human(content: str):
    m = MagicMock()
    m.type = "human"
    m.role = None
    m.content = content
    return m


def _make_ai(content: str):
    m = MagicMock()
    m.type = "ai"
    m.role = None
    m.content = content
    return m


# ---------------------------------------------------------------------------
# _get_analysis_model
# ---------------------------------------------------------------------------


class TestGetAnalysisModel:
    """Verify model selection priority for post-run extraction."""

    def test_returns_none_when_both_absent(self, middleware):
        """When neither _captured_model nor postrun_analysis_model is set → None."""
        assert middleware._get_analysis_model() is None

    def test_returns_captured_model(self, middleware):
        mock_model = MagicMock()
        middleware._captured_model = mock_model
        assert middleware._get_analysis_model() is mock_model

    def test_postrun_analysis_model_name_stored(self, mock_memory_service):
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            postrun_analysis_model="gpt-4.1-mini",
        )
        assert mw.postrun_analysis_model == "gpt-4.1-mini"

    def test_falls_back_to_captured_when_no_override(self, mock_memory_service):
        """Without postrun_analysis_model set, returns _captured_model."""
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        fallback = MagicMock()
        mw._captured_model = fallback
        assert mw._get_analysis_model() is fallback


# ---------------------------------------------------------------------------
# _extract_via_model
# ---------------------------------------------------------------------------


class TestExtractViaModel:
    """Verify model-based session extraction."""

    async def test_returns_none_for_empty_messages(self, middleware):
        state = MemoryState(messages=[])
        model = MagicMock()
        content, mtype = await middleware._extract_via_model(state, model)
        assert content is None
        assert mtype == ""

    async def test_returns_content_on_worth_saving(self, middleware):
        from inference_core.agents.middleware.memory import _MemoryExtractionResult

        state = MemoryState(
            messages=[_make_human("I prefer TypeScript"), _make_ai("Got it.")]
        )
        mock_result = _MemoryExtractionResult(
            worth_saving=True,
            content="User prefers TypeScript",
            memory_type="session_summary",
        )
        model = MagicMock()
        model.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=mock_result
        )
        content, mtype = await middleware._extract_via_model(state, model)
        assert content == "User prefers TypeScript"
        assert mtype == "session_summary"

    async def test_returns_none_when_worth_saving_false(self, middleware):
        from inference_core.agents.middleware.memory import _MemoryExtractionResult

        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        mock_result = _MemoryExtractionResult(
            worth_saving=False, content="", memory_type=""
        )
        model = MagicMock()
        model.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=mock_result
        )
        content, _ = await middleware._extract_via_model(state, model)
        assert content is None

    async def test_returns_none_on_model_exception(self, middleware):
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        model = MagicMock()
        model.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=RuntimeError("model error")
        )
        content, _ = await middleware._extract_via_model(state, model)
        assert content is None

    async def test_defaults_memory_type_to_session_summary(self, middleware):
        """Empty memory_type from model defaults to session_summary."""
        from inference_core.agents.middleware.memory import _MemoryExtractionResult

        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        mock_result = _MemoryExtractionResult(
            worth_saving=True, content="some content", memory_type=""
        )
        model = MagicMock()
        model.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=mock_result
        )
        _, mtype = await middleware._extract_via_model(state, model)
        assert mtype == "session_summary"

    async def test_skips_non_text_content(self, middleware):
        """Messages with non-string content are skipped gracefully."""
        msg = MagicMock()
        msg.type = "human"
        msg.role = None
        msg.content = None  # non-string
        state = MemoryState(messages=[msg])
        model = MagicMock()
        content, _ = await middleware._extract_via_model(state, model)
        assert content is None


# ---------------------------------------------------------------------------
# after_agent (sync)
# ---------------------------------------------------------------------------


class TestAfterAgent:
    """Verify post-run persistence hook — sync path."""

    def test_skips_when_postrun_analysis_disabled(self, mock_memory_service):
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            postrun_analysis=False,
        )
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        assert mw.after_agent(state, MagicMock()) is None

    def test_skips_when_no_user_id(self, mock_memory_service):
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id=None,
        )
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        with patch(
            "inference_core.agents.middleware.memory._ctx_get_user_id",
            return_value=None,
        ):
            assert mw.after_agent(state, MagicMock()) is None

    def test_skips_when_fewer_than_two_messages(self, middleware):
        state = MemoryState(messages=[_make_human("hi")])
        assert middleware.after_agent(state, MagicMock()) is None

    def test_skips_when_no_model_available(self, middleware):
        """When _get_analysis_model returns None, analysis is skipped."""
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        with patch.object(middleware, "_get_analysis_model", return_value=None):
            assert middleware.after_agent(state, MagicMock()) is None

    def test_returns_none_when_analyse_returns_none(self, middleware):
        """When run_async_safely returns None (nothing saved), after_agent returns None."""
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        middleware._captured_model = MagicMock()
        with patch(
            "inference_core.agents.middleware.memory.run_async_safely",
            return_value=None,
        ):
            assert middleware.after_agent(state, MagicMock()) is None

    def test_saves_and_returns_state_on_success(self, middleware):
        """When run_async_safely returns memory_type string, state dict is returned."""
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        middleware._captured_model = MagicMock()
        with patch(
            "inference_core.agents.middleware.memory.run_async_safely",
            return_value="session_summary",
        ):
            result = middleware.after_agent(state, MagicMock())
        assert result is not None
        assert result["session_analysis_saved"] is True
        assert "session_analysis_latency_ms" in result

    def test_returns_none_on_exception(self, middleware):
        """run_async_safely raising is swallowed (fail-open)."""
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        middleware._captured_model = MagicMock()
        with patch(
            "inference_core.agents.middleware.memory.run_async_safely",
            side_effect=RuntimeError("DB down"),
        ):
            assert middleware.after_agent(state, MagicMock()) is None


# ---------------------------------------------------------------------------
# aafter_agent (async)
# ---------------------------------------------------------------------------


class TestAAfterAgent:
    """Verify post-run persistence hook — async path."""

    async def test_skips_when_postrun_analysis_disabled(self, mock_memory_service):
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            postrun_analysis=False,
        )
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        assert await mw.aafter_agent(state, MagicMock()) is None

    async def test_skips_when_no_user_id(self, mock_memory_service):
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id=None,
        )
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        with patch(
            "inference_core.agents.middleware.memory._ctx_get_user_id",
            return_value=None,
        ):
            assert await mw.aafter_agent(state, MagicMock()) is None

    async def test_skips_when_no_model_available(self, mock_memory_service):
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        with patch.object(mw, "_get_analysis_model", return_value=None):
            assert await mw.aafter_agent(state, MagicMock()) is None

    async def test_skips_when_extract_returns_none(self, mock_memory_service):
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        mw._captured_model = MagicMock()
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        with patch.object(
            mw, "_extract_via_model", new=AsyncMock(return_value=(None, ""))
        ):
            assert await mw.aafter_agent(state, MagicMock()) is None

    async def test_saves_and_returns_state_on_success(self, mock_memory_service):
        mock_memory_service.save_memory = AsyncMock(return_value="mem-id")
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        mw._captured_model = MagicMock()
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        with patch.object(
            mw,
            "_extract_via_model",
            new=AsyncMock(return_value=("some preference", "session_summary")),
        ):
            result = await mw.aafter_agent(state, MagicMock())
        assert result is not None
        assert result["session_analysis_saved"] is True
        mock_memory_service.save_memory.assert_awaited_once()

    async def test_skips_non_summary_when_already_saved(self, mock_memory_service):
        """When agent already called save and extraction type=interaction, skip."""
        mock_memory_service.save_memory = AsyncMock()
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        mw._captured_model = MagicMock()
        tc = MagicMock()
        tc.name = "save_memory_store"
        ai_msg = MagicMock()
        ai_msg.name = None
        ai_msg.tool_calls = [tc]
        state = MemoryState(messages=[_make_human("msg"), ai_msg])
        with patch.object(
            mw,
            "_extract_via_model",
            new=AsyncMock(return_value=("correction", "interaction")),
        ):
            result = await mw.aafter_agent(state, MagicMock())
        assert result is None
        mock_memory_service.save_memory.assert_not_awaited()

    async def test_returns_none_on_service_exception(self, mock_memory_service):
        mock_memory_service.save_memory = AsyncMock(
            side_effect=RuntimeError("DB down")
        )
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        mw._captured_model = MagicMock()
        state = MemoryState(messages=[_make_human("hi"), _make_ai("hello")])
        with patch.object(
            mw,
            "_extract_via_model",
            new=AsyncMock(return_value=("content", "session_summary")),
        ):
            assert await mw.aafter_agent(state, MagicMock()) is None


# ---------------------------------------------------------------------------
# postrun_analysis init param + factory
# ---------------------------------------------------------------------------


class TestPostrunAnalysisParam:
    """Verify postrun_analysis and postrun_analysis_model are wired correctly."""

    def test_default_postrun_analysis_is_true(self, mock_memory_service):
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        assert mw.postrun_analysis is True

    def test_postrun_analysis_can_be_disabled(self, mock_memory_service):
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            postrun_analysis=False,
        )
        assert mw.postrun_analysis is False

    def test_postrun_analysis_model_default_none(self, mock_memory_service):
        mw = MemoryMiddleware(memory_service=mock_memory_service, user_id="u1")
        assert mw.postrun_analysis_model is None

    def test_postrun_analysis_model_stored(self, mock_memory_service):
        mw = MemoryMiddleware(
            memory_service=mock_memory_service,
            user_id="u1",
            postrun_analysis_model="gpt-4.1-mini",
        )
        assert mw.postrun_analysis_model == "gpt-4.1-mini"

    def test_factory_passes_postrun_analysis(self, mock_memory_service):
        mw = create_memory_middleware(
            memory_service=mock_memory_service,
            user_id="u42",
            postrun_analysis=False,
        )
        assert mw.postrun_analysis is False

    def test_factory_passes_postrun_analysis_model(self, mock_memory_service):
        mw = create_memory_middleware(
            memory_service=mock_memory_service,
            user_id="u42",
            postrun_analysis_model="gpt-4.1-mini",
        )
        assert mw.postrun_analysis_model == "gpt-4.1-mini"

    def test_factory_default_postrun_analysis_is_true(self, mock_memory_service):
        mw = create_memory_middleware(
            memory_service=mock_memory_service,
            user_id="u42",
        )
        assert mw.postrun_analysis is True

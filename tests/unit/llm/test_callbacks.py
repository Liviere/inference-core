"""Tests for LLMUsageCallbackHandler and ToolUsageCallbackHandler.

Covers: on_llm_end (delta calculation, fallback to llm_output, detail extraction),
on_llm_error (partial flag), ToolUsageCallbackHandler events lifecycle.
"""

from unittest.mock import MagicMock

from inference_core.llm.callbacks import (
    LLMUsageCallbackHandler,
    ToolUsageCallbackHandler,
)
from inference_core.llm.usage_logging import UsageLoggingConfig, UsageSession

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(model: str = "test-model") -> UsageSession:
    return UsageSession(
        task_type="unit-test",
        request_mode="sync",
        model_name=model,
        provider="test",
        pricing_config=None,
        logging_config=UsageLoggingConfig(enabled=False),
    )


class _FakeLLMResult:
    """Minimal stand-in for LangChain LLMResult."""

    def __init__(self, llm_output=None, generations=None):
        self.llm_output = llm_output
        self.generations = generations or [[]]


class _FakeAIMessage:
    """Stand-in for AIMessage with usage_metadata."""

    def __init__(self, usage_metadata, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}


class _FakeGeneration:
    def __init__(self, message):
        self.message = message


# ===========================================================================
# LLMUsageCallbackHandler
# ===========================================================================


class TestLLMUsageCallbackOnLlmEnd:
    """Test on_llm_end usage accumulation via usage_metadata and llm_output."""

    def test_accumulates_via_usage_metadata(self):
        """Delta calculation from usage_metadata accumulates correctly."""
        session = _make_session()
        handler = LLMUsageCallbackHandler(usage_session=session)

        # Simulate base class populating usage_metadata
        handler.usage_metadata = {
            "test-model": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        }

        result = _FakeLLMResult()
        handler.on_llm_end(result)

        assert session.accumulated_usage.get("input_tokens") == 100
        assert session.accumulated_usage.get("output_tokens") == 50

    def test_delta_prevents_double_count(self):
        """Second call only accumulates the delta, not re-counted totals."""
        session = _make_session()
        handler = LLMUsageCallbackHandler(usage_session=session)

        # First call
        handler.usage_metadata = {
            "test-model": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        }
        handler.on_llm_end(_FakeLLMResult())

        # Second call – cumulative totals increased
        handler.usage_metadata = {
            "test-model": {
                "input_tokens": 150,
                "output_tokens": 80,
                "total_tokens": 230,
            }
        }
        handler.on_llm_end(_FakeLLMResult())

        # Deltas: input 50, output 30
        assert session.accumulated_usage.get("input_tokens") == 150
        assert session.accumulated_usage.get("output_tokens") == 80

    def test_fallback_to_llm_output_token_usage(self):
        """When usage_metadata is empty, falls back to llm_output.token_usage."""
        session = _make_session()
        handler = LLMUsageCallbackHandler(usage_session=session)
        handler.usage_metadata = {}

        result = _FakeLLMResult(
            llm_output={"token_usage": {"prompt_tokens": 20, "completion_tokens": 15}}
        )
        handler.on_llm_end(result)

        assert session.accumulated_usage.get("prompt_tokens") == 20
        assert session.accumulated_usage.get("completion_tokens") == 15

    def test_handles_none_llm_output_gracefully(self):
        """No crash when llm_output is None."""
        session = _make_session()
        handler = LLMUsageCallbackHandler(usage_session=session)
        handler.usage_metadata = {}
        handler.on_llm_end(_FakeLLMResult(llm_output=None))
        assert session.accumulated_usage == {}

    def test_input_token_details_extracted(self):
        """Detail tokens (cache_read, reasoning) are extracted from metadata."""
        session = _make_session()
        handler = LLMUsageCallbackHandler(usage_session=session)
        handler.usage_metadata = {
            "test-model": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "input_token_details": {"cache_read": 30},
                "output_token_details": {"reasoning": 10},
            }
        }
        handler.on_llm_end(_FakeLLMResult())
        assert session.accumulated_usage.get("cache_read_tokens") == 30
        assert session.accumulated_usage.get("reasoning_tokens") == 10


class TestLLMUsageCallbackOnLlmError:
    """Test on_llm_error sets partial flag."""

    def test_sets_partial_flag(self):
        """On error the session is marked as partial."""
        session = _make_session()
        handler = LLMUsageCallbackHandler(usage_session=session)
        handler.on_llm_error(RuntimeError("boom"))
        assert session.partial is True


class TestLLMUsageCallbackGetAccumulated:
    """Test get_accumulated returns session data."""

    def test_returns_copy_of_accumulated(self):
        """get_accumulated returns a dict copy of accumulated_usage."""
        session = _make_session()
        handler = LLMUsageCallbackHandler(usage_session=session)
        session.accumulated_usage["input_tokens"] = 42
        result = handler.get_accumulated()
        assert result["input_tokens"] == 42
        assert isinstance(result, dict)


# ===========================================================================
# ToolUsageCallbackHandler
# ===========================================================================


class TestToolUsageCallbackHandler:
    """Test ToolUsageCallbackHandler event lifecycle."""

    def test_on_tool_start_captures_event(self):
        """on_tool_start records tool name and input."""
        handler = ToolUsageCallbackHandler()
        handler.on_tool_start({"name": "browser_navigate"}, input_str="https://example.com")

        assert len(handler.events) == 1
        assert handler.events[0]["tool"] == "browser_navigate"
        assert handler.events[0]["input"] == "https://example.com"
        assert handler.events[0]["event"] == "start"

    def test_on_tool_end_annotates_last_start(self):
        """on_tool_end annotates the last 'start' event with output."""
        handler = ToolUsageCallbackHandler()
        handler.on_tool_start({"name": "search"}, input_str="query")
        handler.on_tool_end(output="3 results found")

        assert handler.events[0]["event"] == "finish"
        assert handler.events[0]["output"] == "3 results found"

    def test_on_tool_end_truncates_long_output(self):
        """Very long output is truncated to 2000 chars."""
        handler = ToolUsageCallbackHandler()
        handler.on_tool_start({"name": "fetch"}, input_str="url")
        long_output = "x" * 3000
        handler.on_tool_end(output=long_output)

        assert len(handler.events[0]["output"]) < 3000
        assert handler.events[0]["output"].endswith("…")

    def test_on_tool_error_captures_error(self):
        """on_tool_error records the error."""
        handler = ToolUsageCallbackHandler()
        handler.on_tool_error(RuntimeError("connection failed"))

        assert len(handler.events) == 1
        assert handler.events[0]["event"] == "error"
        assert "connection failed" in handler.events[0]["error"]

    def test_get_events_returns_copy(self):
        """get_events returns a list copy."""
        handler = ToolUsageCallbackHandler()
        handler.on_tool_start({"name": "tool1"})
        events = handler.get_events()
        assert len(events) == 1
        events.clear()
        assert len(handler.events) == 1  # original unchanged

    def test_on_tool_start_unknown_name(self):
        """Handles serialized dict without a 'name' key."""
        handler = ToolUsageCallbackHandler()
        handler.on_tool_start({}, input_str="test")
        assert handler.events[0]["tool"] == "<unknown>"

    def test_on_tool_start_uses_tool_key(self):
        """Falls back to 'tool' key in serialized dict."""
        handler = ToolUsageCallbackHandler()
        handler.on_tool_start({"tool": "my_tool"}, input_str="arg")
        assert handler.events[0]["tool"] == "my_tool"

    def test_on_tool_end_without_prior_start(self):
        """on_tool_end without prior start appends standalone finish event."""
        handler = ToolUsageCallbackHandler()
        handler.on_tool_end(output="orphan result")
        assert handler.events[0]["event"] == "finish"
        assert handler.events[0]["output"] == "orphan result"

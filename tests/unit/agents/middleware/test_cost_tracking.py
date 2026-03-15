"""Tests for CostTrackingMiddleware.

Covers the static helper methods (_extract_usage_fragment, _merge_extra_tokens,
_detect_provider) and the lifecycle hooks (before_agent, after_model,
wrap_model_call, wrap_tool_call).
"""

import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
from langchain.messages import ToolMessage

from inference_core.agents.middleware.cost_tracking import (
    CostTrackingMiddleware,
    CostTrackingState,
    _MiddlewareContext,
    create_cost_tracking_middleware,
)
from inference_core.services._cancel import AgentCancelled

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def middleware():
    """Basic CostTrackingMiddleware with all external deps mocked away."""
    return CostTrackingMiddleware(
        pricing_config=None,
        user_id=uuid.uuid4(),
        session_id="sess-1",
        request_id="req-1",
        task_type="agent",
        request_mode="sync",
        provider="openai",
        model_name="gpt-4o",
    )


# ---------------------------------------------------------------------------
# _extract_usage_fragment  (pure static, no mocking needed)
# ---------------------------------------------------------------------------


class TestExtractUsageFragment:
    """Verify token extraction from LangChain usage_metadata dicts."""

    def test_basic_tokens(self):
        """Standard input/output/total extraction."""
        meta = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        fragment, extras = CostTrackingMiddleware._extract_usage_fragment(meta)

        assert fragment["input_tokens"] == 100
        assert fragment["output_tokens"] == 50
        assert fragment["total_tokens"] == 150
        assert extras == {}

    def test_total_defaults_to_sum(self):
        """total_tokens defaults to input + output when absent."""
        meta = {"input_tokens": 30, "output_tokens": 20}
        fragment, _ = CostTrackingMiddleware._extract_usage_fragment(meta)
        assert fragment["total_tokens"] == 50

    def test_none_values_treated_as_zero(self):
        """None token counts are coerced to 0 (total falls back to sum)."""
        meta = {"input_tokens": None, "output_tokens": None}
        fragment, _ = CostTrackingMiddleware._extract_usage_fragment(meta)
        assert fragment["input_tokens"] == 0
        assert fragment["output_tokens"] == 0
        # total_tokens defaults to input + output when key is absent
        assert fragment["total_tokens"] == 0

    def test_input_detail_tokens(self):
        """Tokens from input_token_details are extracted to extras."""
        meta = {
            "input_tokens": 200,
            "output_tokens": 50,
            "total_tokens": 250,
            "input_token_details": {"cached": 80, "audio": 0},
        }
        fragment, extras = CostTrackingMiddleware._extract_usage_fragment(meta)

        assert fragment["cached_tokens"] == 80
        assert extras["cached_tokens"] == 80
        # audio=0 should be excluded (not > 0)
        assert "audio_tokens" not in extras

    def test_output_detail_tokens(self):
        """Tokens from output_token_details are extracted to extras."""
        meta = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "output_token_details": {"reasoning": 30},
        }
        fragment, extras = CostTrackingMiddleware._extract_usage_fragment(meta)

        assert fragment["reasoning_tokens"] == 30
        assert extras["reasoning_tokens"] == 30

    def test_top_level_extra_tokens(self):
        """Arbitrary *_tokens keys at top level are captured in extras."""
        meta = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cache_read_tokens": 25,
        }
        fragment, extras = CostTrackingMiddleware._extract_usage_fragment(meta)

        assert extras["cache_read_tokens"] == 25
        assert fragment["cache_read_tokens"] == 25

    def test_empty_metadata(self):
        """Empty dict produces all-zero fragment."""
        fragment, extras = CostTrackingMiddleware._extract_usage_fragment({})
        assert fragment["input_tokens"] == 0
        assert fragment["output_tokens"] == 0
        assert fragment["total_tokens"] == 0
        assert extras == {}


# ---------------------------------------------------------------------------
# _merge_extra_tokens  (pure static)
# ---------------------------------------------------------------------------


class TestMergeExtraTokens:
    """Verify additive merging of extra-token counters."""

    def test_disjoint_keys(self):
        """Non-overlapping keys are simply combined."""
        merged = CostTrackingMiddleware._merge_extra_tokens({"a": 10}, {"b": 20})
        assert merged == {"a": 10, "b": 20}

    def test_overlapping_keys_are_summed(self):
        """Overlapping keys have their values added."""
        merged = CostTrackingMiddleware._merge_extra_tokens(
            {"reasoning_tokens": 5}, {"reasoning_tokens": 10}
        )
        assert merged == {"reasoning_tokens": 15}

    def test_empty_inputs(self):
        """Two empty dicts produce empty result."""
        assert CostTrackingMiddleware._merge_extra_tokens({}, {}) == {}

    def test_original_not_mutated(self):
        """Accumulated dict must not be mutated in place."""
        accumulated = {"a": 1}
        CostTrackingMiddleware._merge_extra_tokens(accumulated, {"a": 2})
        assert accumulated == {"a": 1}


# ---------------------------------------------------------------------------
# _detect_provider  (static, parametrized)
# ---------------------------------------------------------------------------


class TestDetectProvider:
    """Verify model-name → provider mapping heuristic."""

    @pytest.mark.parametrize(
        "model_name, expected_provider",
        [
            ("gpt-4o", "openai"),
            ("gpt-3.5-turbo", "openai"),
            ("o1-preview", "openai"),
            ("o3-mini", "openai"),
            ("davinci-002", "openai"),
            ("claude-3-opus", "anthropic"),
            ("anthropic-claude", "anthropic"),
            ("gemini-1.5-pro", "google"),
            ("palm-2", "google"),
            ("llama-3-70b", "meta"),
            ("mistral-large", "meta"),
            ("mixtral-8x7b", "meta"),
            ("deepseek-coder", "deepseek"),
            ("my-custom-model", "unknown"),
        ],
    )
    def test_detect_provider(self, model_name, expected_provider):
        """Provider is detected from substrings in model name."""
        assert CostTrackingMiddleware._detect_provider(model_name) == expected_provider


# ---------------------------------------------------------------------------
# before_agent
# ---------------------------------------------------------------------------


class TestBeforeAgent:
    """Verify that before_agent initialises context and state counters."""

    def test_initialises_counters(self, middleware):
        """Returns state dict with all tracking fields zeroed."""
        state = CostTrackingState(messages=[])
        runtime = MagicMock()

        updates = middleware.before_agent(state, runtime)

        assert updates is not None
        assert updates["accumulated_input_tokens"] == 0
        assert updates["accumulated_output_tokens"] == 0
        assert updates["accumulated_total_tokens"] == 0
        assert updates["accumulated_extra_tokens"] == {}
        assert updates["tool_call_count"] == 0
        assert updates["model_call_count"] == 0
        assert updates["model_call_latencies"] == []
        assert updates["tool_call_latencies"] == []
        assert "usage_session_id" in updates

    def test_creates_fresh_context(self, middleware):
        """Internal _ctx is freshly created on each before_agent call."""
        state = CostTrackingState(messages=[])
        runtime = MagicMock()

        middleware.before_agent(state, runtime)
        assert middleware._ctx is not None
        assert middleware._ctx.model_call_start_time is None


# ---------------------------------------------------------------------------
# wrap_model_call
# ---------------------------------------------------------------------------


class TestWrapModelCall:
    """Verify that wrap_model_call records start time and delegates."""

    def test_records_start_time(self, middleware):
        """model_call_start_time is set before calling handler."""
        middleware._ctx = _MiddlewareContext()
        request = MagicMock()
        response = MagicMock()
        handler = MagicMock(return_value=response)

        result = middleware.wrap_model_call(request, handler)

        assert middleware._ctx.model_call_start_time is not None
        handler.assert_called_once_with(request)
        assert result is response

    def test_no_ctx_still_works(self, middleware):
        """If _ctx is None, handler is still called (no crash)."""
        middleware._ctx = None
        handler = MagicMock(return_value="ok")

        result = middleware.wrap_model_call(MagicMock(), handler)

        assert result == "ok"
        handler.assert_called_once()


# ---------------------------------------------------------------------------
# wrap_tool_call
# ---------------------------------------------------------------------------


class TestWrapToolCall:
    """Verify tool call wrapping and error handling."""

    def test_delegates_to_handler(self, middleware):
        """Handler is called with request and its result is returned."""
        request = MagicMock()
        request.tool_call = {"name": "calculator"}
        response = MagicMock()
        handler = MagicMock(return_value=response)

        result = middleware.wrap_tool_call(request, handler)

        handler.assert_called_once_with(request)
        assert result is response

    def test_returns_error_tool_message_on_handler_exception(self, middleware):
        """When handler raises, ToolMessage preserves the error and tool call ID."""
        request = MagicMock()
        request.tool_call = {"name": "broken_tool", "id": "tool-call-1"}
        handler = MagicMock(side_effect=RuntimeError("boom"))

        result = middleware.wrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.content == "Error: boom"
        assert result.tool_call_id == "tool-call-1"


# ---------------------------------------------------------------------------
# after_model  (requires mocked state + UsageSession)
# ---------------------------------------------------------------------------


class TestAfterModel:
    """Verify that after_model extracts usage and persists via UsageSession."""

    def test_initializes_ctx_when_none(self, middleware):
        """If _ctx is None, after_model initializes it and returns updates."""
        middleware._ctx = None
        state = CostTrackingState(messages=[])
        result = middleware.after_model(state, MagicMock())
        assert result == {"model_call_count": 1}
        assert middleware._ctx is not None

    def test_increments_model_call_count(self, middleware):
        """model_call_count is bumped by 1 on each after_model call."""
        middleware._ctx = _MiddlewareContext()

        ai_msg = MagicMock()
        ai_msg.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        ai_msg.response_metadata = {"model_name": "gpt-4o"}

        state = CostTrackingState(
            messages=[ai_msg],
            model_call_count=2,
            accumulated_input_tokens=0,
            accumulated_output_tokens=0,
            accumulated_total_tokens=0,
            accumulated_extra_tokens={},
        )

        with patch(
            "inference_core.agents.middleware.cost_tracking.UsageSession"
        ) as MockSession:
            mock_session_instance = MagicMock()
            MockSession.return_value = mock_session_instance

            updates = middleware.after_model(state, MagicMock())

        assert updates["model_call_count"] == 3

    def test_accumulates_tokens(self, middleware):
        """Token counts are added to running accumulators."""
        middleware._ctx = _MiddlewareContext()

        ai_msg = MagicMock()
        ai_msg.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        ai_msg.response_metadata = {"model_name": "gpt-4o"}

        state = CostTrackingState(
            messages=[ai_msg],
            model_call_count=0,
            accumulated_input_tokens=200,
            accumulated_output_tokens=100,
            accumulated_total_tokens=300,
            accumulated_extra_tokens={},
        )

        with patch(
            "inference_core.agents.middleware.cost_tracking.UsageSession"
        ) as MockSession:
            mock_session_instance = MagicMock()
            MockSession.return_value = mock_session_instance

            updates = middleware.after_model(state, MagicMock())

        assert updates["accumulated_input_tokens"] == 300
        assert updates["accumulated_output_tokens"] == 150
        assert updates["accumulated_total_tokens"] == 450

    def test_persists_usage_session(self, middleware):
        """UsageSession.finalize_sync is called for each model step."""
        middleware._ctx = _MiddlewareContext()
        middleware._ctx.model_call_start_time = time.monotonic()

        ai_msg = MagicMock()
        ai_msg.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        ai_msg.response_metadata = {"model_name": "gpt-4o"}

        state = CostTrackingState(
            messages=[ai_msg],
            model_call_count=0,
            accumulated_input_tokens=0,
            accumulated_output_tokens=0,
            accumulated_total_tokens=0,
            accumulated_extra_tokens={},
        )

        with patch(
            "inference_core.agents.middleware.cost_tracking.UsageSession"
        ) as MockSession:
            mock_session_instance = MagicMock()
            MockSession.return_value = mock_session_instance

            middleware.after_model(state, MagicMock())

            MockSession.assert_called_once()
            mock_session_instance.finalize_sync.assert_called_once_with(
                success=True,
                error=None,
                final_usage={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
                streamed=False,
                partial=False,
                details=None,
            )

    def test_no_messages_returns_count_only(self, middleware):
        """Empty messages list still increments model_call_count."""
        middleware._ctx = _MiddlewareContext()
        state = CostTrackingState(messages=[], model_call_count=0)

        updates = middleware.after_model(state, MagicMock())

        assert updates["model_call_count"] == 1
        # No token keys since no message to extract from
        assert "accumulated_input_tokens" not in updates

    def test_clears_model_call_start_time(self, middleware):
        """After extraction, model_call_start_time is reset to None."""
        middleware._ctx = _MiddlewareContext()
        middleware._ctx.model_call_start_time = 12345.0

        ai_msg = MagicMock()
        ai_msg.usage_metadata = {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        }
        ai_msg.response_metadata = {}

        state = CostTrackingState(
            messages=[ai_msg],
            model_call_count=0,
            accumulated_input_tokens=0,
            accumulated_output_tokens=0,
            accumulated_total_tokens=0,
            accumulated_extra_tokens={},
        )

        with patch("inference_core.agents.middleware.cost_tracking.UsageSession"):
            middleware.after_model(state, MagicMock())

        assert middleware._ctx.model_call_start_time is None


# ---------------------------------------------------------------------------
# after_agent  (no-op, just verify contract)
# ---------------------------------------------------------------------------


class TestAfterAgent:
    """Verify after_agent is a no-op (per-step logging in after_model)."""

    def test_returns_none(self, middleware):
        """after_agent returns None (nothing additional to persist)."""
        state = CostTrackingState(messages=[])
        result = middleware.after_agent(state, MagicMock())
        assert result is None


# ---------------------------------------------------------------------------
# _get_pricing_config
# ---------------------------------------------------------------------------


class TestGetPricingConfig:
    """Verify pricing config resolution (llm_config → middleware default)."""

    def test_returns_model_pricing_from_llm_config(self, middleware):
        """If llm_config has per-model pricing, return it."""
        mock_pricing = MagicMock()
        mock_model_cfg = MagicMock()
        mock_model_cfg.pricing = mock_pricing

        mock_llm_config = MagicMock()
        mock_llm_config.models = {"gpt-4o": mock_model_cfg}

        with patch(
            "inference_core.agents.middleware.cost_tracking.get_llm_config",
            return_value=mock_llm_config,
        ):
            result = middleware._get_pricing_config("gpt-4o")

        assert result is mock_pricing

    def test_falls_back_to_middleware_pricing(self, middleware):
        """If llm_config has no model, fall back to middleware's pricing_config."""
        fallback_pricing = MagicMock()
        middleware.pricing_config = fallback_pricing

        mock_llm_config = MagicMock()
        mock_llm_config.models = {}

        with patch(
            "inference_core.agents.middleware.cost_tracking.get_llm_config",
            return_value=mock_llm_config,
        ):
            result = middleware._get_pricing_config("unknown-model")

        assert result is fallback_pricing

    def test_returns_none_when_nothing_available(self, middleware):
        """If neither llm_config nor middleware has pricing, return None."""
        middleware.pricing_config = None

        mock_llm_config = MagicMock()
        mock_llm_config.models = {}

        with patch(
            "inference_core.agents.middleware.cost_tracking.get_llm_config",
            return_value=mock_llm_config,
        ):
            result = middleware._get_pricing_config("unknown-model")

        assert result is None


# ---------------------------------------------------------------------------
# create_cost_tracking_middleware factory
# ---------------------------------------------------------------------------


class TestCreateCostTrackingMiddleware:
    """Verify factory function creates middleware with sensible defaults."""

    def test_creates_with_logging_config(self):
        """Factory loads logging config from llm_config when available."""
        mock_logging_cfg = MagicMock()
        mock_llm_config = MagicMock()
        mock_llm_config.usage_logging = mock_logging_cfg

        with patch(
            "inference_core.agents.middleware.cost_tracking.get_llm_config",
            return_value=mock_llm_config,
        ):
            mw = create_cost_tracking_middleware(
                user_id=uuid.uuid4(),
                session_id="s1",
            )

        assert mw.logging_config is mock_logging_cfg
        assert mw.task_type == "agent"

    def test_creates_with_defaults_on_config_error(self):
        """Factory still works when get_llm_config throws."""
        with patch(
            "inference_core.agents.middleware.cost_tracking.get_llm_config",
            side_effect=RuntimeError("no config"),
        ):
            mw = create_cost_tracking_middleware()

        assert mw.logging_config is not None  # UsageLoggingConfig() default
        assert mw.user_id is None


# ---------------------------------------------------------------------------
# _persist_from_response  (cancel-safe DB persistence in wrap_model_call)
# ---------------------------------------------------------------------------


def _make_ai_message(
    input_tokens=10, output_tokens=5, total_tokens=15, model_name="gpt-4o"
):
    """Helper: create a MagicMock that looks like an AIMessage with usage."""
    msg = MagicMock()
    msg.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    msg.response_metadata = {"model_name": model_name}
    msg.content = "Hello"
    return msg


def _make_model_response(ai_message):
    """Helper: create a MagicMock ModelResponse wrapping an AIMessage."""
    resp = MagicMock()
    resp.result = [ai_message]
    return resp


def _make_model_request():
    """Helper: create a MagicMock ModelRequest."""
    req = MagicMock()
    req.messages = []
    req.system_message = None
    return req


class TestPersistFromResponse:
    """Verify _persist_from_response extracts usage and writes LLMRequestLog."""

    def test_persists_log_and_populates_ctx(self, middleware):
        """Successful persistence stores log_id, cost, model_name in _ctx."""
        middleware._ctx = _MiddlewareContext()
        middleware._ctx.model_call_start_time = time.monotonic()
        ai_msg = _make_ai_message()
        response = _make_model_response(ai_msg)
        request = _make_model_request()

        mock_pricing = MagicMock()
        mock_cost = {"cost_total_usd": 0.0042}
        fake_log_id = uuid.uuid4()

        with (
            patch(
                "inference_core.agents.middleware.cost_tracking.UsageSession"
            ) as MockSession,
            patch(
                "inference_core.agents.middleware.cost_tracking.PricingCalculator"
            ) as MockCalc,
        ):
            MockSession.return_value.finalize_sync.return_value = fake_log_id
            MockCalc.compute_cost.return_value = mock_cost
            middleware._get_pricing_config = MagicMock(return_value=mock_pricing)

            middleware._persist_from_response(response, request)

        assert middleware._ctx.persisted_log_id == str(fake_log_id)
        assert middleware._ctx.persisted_cost_usd == 0.0042
        assert middleware._ctx.persisted_model_name == "gpt-4o"
        assert middleware._ctx.persisted_fragment["input_tokens"] == 10
        assert middleware._ctx.persisted_fragment["output_tokens"] == 5

    def test_no_ctx_returns_silently(self, middleware):
        """If _ctx is None, method returns without error."""
        middleware._ctx = None
        middleware._persist_from_response(MagicMock(), MagicMock())
        # No exception = pass

    def test_empty_result_returns_silently(self, middleware):
        """If response.result is empty, method returns without persisting."""
        middleware._ctx = _MiddlewareContext()
        response = MagicMock()
        response.result = []

        middleware._persist_from_response(response, MagicMock())

        assert middleware._ctx.persisted_log_id is None

    def test_finalize_failure_is_non_fatal(self, middleware):
        """If finalize_sync raises, the exception is swallowed."""
        middleware._ctx = _MiddlewareContext()
        ai_msg = _make_ai_message()
        response = _make_model_response(ai_msg)
        request = _make_model_request()

        with patch(
            "inference_core.agents.middleware.cost_tracking.UsageSession"
        ) as MockSession:
            MockSession.return_value.finalize_sync.side_effect = RuntimeError("DB down")

            middleware._persist_from_response(response, request)

        # No exception, persisted_log_id stays None
        assert middleware._ctx.persisted_log_id is None

    def test_no_usage_metadata_falls_back_to_estimation(self, middleware):
        """When usage_metadata is None, estimation is used."""
        middleware._ctx = _MiddlewareContext()
        ai_msg = MagicMock()
        ai_msg.usage_metadata = None
        ai_msg.response_metadata = {"model_name": "custom-model"}
        ai_msg.content = "short"

        response = MagicMock()
        response.result = [ai_msg]
        request = _make_model_request()

        fake_log_id = uuid.uuid4()

        with patch(
            "inference_core.agents.middleware.cost_tracking.UsageSession"
        ) as MockSession:
            MockSession.return_value.finalize_sync.return_value = fake_log_id
            middleware._get_pricing_config = MagicMock(return_value=None)

            middleware._persist_from_response(response, request)

        assert middleware._ctx.persisted_log_id == str(fake_log_id)
        assert middleware._ctx.persisted_is_estimated is True


# ---------------------------------------------------------------------------
# wrap_model_call + _persist_from_response integration
# ---------------------------------------------------------------------------


class TestWrapModelCallPersistence:
    """Verify wrap_model_call calls _persist_from_response after handler."""

    def test_persist_called_after_handler(self, middleware):
        """_persist_from_response is invoked after handler returns."""
        middleware._ctx = _MiddlewareContext()
        request = _make_model_request()
        ai_msg = _make_ai_message()
        response = _make_model_response(ai_msg)
        handler = MagicMock(return_value=response)

        with patch.object(middleware, "_persist_from_response") as mock_persist:
            result = middleware.wrap_model_call(request, handler)

        handler.assert_called_once_with(request)
        mock_persist.assert_called_once_with(response, request)
        assert result is response

    def test_persist_not_called_when_cancelled_before_model(self, middleware):
        """If cancel_check fires before handler, _persist is never called."""
        middleware._ctx = _MiddlewareContext()
        middleware.cancel_check = MagicMock(return_value=True)
        handler = MagicMock()

        with patch.object(middleware, "_persist_from_response") as mock_persist:
            with pytest.raises(AgentCancelled):
                middleware.wrap_model_call(MagicMock(), handler)

        handler.assert_not_called()
        mock_persist.assert_not_called()


# ---------------------------------------------------------------------------
# after_model fast-path (pre-persisted data from _ctx)
# ---------------------------------------------------------------------------


class TestAfterModelFastPath:
    """Verify after_model reads pre-persisted data and skips DB write."""

    def test_uses_ctx_persisted_data(self, middleware):
        """When _ctx has persisted_log_id, after_model uses it directly."""
        middleware._ctx = _MiddlewareContext()
        middleware._ctx.persisted_log_id = "log-abc-123"
        middleware._ctx.persisted_cost_usd = 0.005
        middleware._ctx.persisted_model_name = "gpt-4o"
        middleware._ctx.persisted_fragment = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        middleware._ctx.persisted_extra_tokens = {}

        state = CostTrackingState(
            messages=[MagicMock()],
            model_call_count=1,
            accumulated_input_tokens=200,
            accumulated_output_tokens=100,
            accumulated_total_tokens=300,
            accumulated_extra_tokens={},
        )

        # UsageSession should NOT be called (no DB write)
        with patch(
            "inference_core.agents.middleware.cost_tracking.UsageSession"
        ) as MockSession:
            updates = middleware.after_model(state, MagicMock())

        MockSession.assert_not_called()
        assert updates["last_request_log_id"] == "log-abc-123"
        assert updates["last_request_cost_usd"] == 0.005
        assert updates["last_request_model_name"] == "gpt-4o"
        assert updates["accumulated_input_tokens"] == 300
        assert updates["accumulated_output_tokens"] == 150
        assert updates["accumulated_total_tokens"] == 450
        assert updates["model_call_count"] == 2

    def test_resets_persisted_data_after_use(self, middleware):
        """After fast-path, persisted_* fields are reset for next model call."""
        middleware._ctx = _MiddlewareContext()
        middleware._ctx.persisted_log_id = "log-xyz"
        middleware._ctx.persisted_cost_usd = 0.001
        middleware._ctx.persisted_model_name = "gpt-4o"
        middleware._ctx.persisted_fragment = {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        }
        middleware._ctx.persisted_extra_tokens = {}

        state = CostTrackingState(
            messages=[MagicMock()],
            model_call_count=0,
            accumulated_input_tokens=0,
            accumulated_output_tokens=0,
            accumulated_total_tokens=0,
            accumulated_extra_tokens={},
        )

        middleware.after_model(state, MagicMock())

        assert middleware._ctx.persisted_log_id is None
        assert middleware._ctx.persisted_cost_usd is None
        assert middleware._ctx.persisted_model_name is None
        assert middleware._ctx.persisted_fragment is None

    def test_fallback_to_legacy_when_no_persisted(self, middleware):
        """When _ctx has no persisted data, after_model falls back to DB write."""
        middleware._ctx = _MiddlewareContext()
        # No persisted_* set → fallback path

        ai_msg = _make_ai_message()
        state = CostTrackingState(
            messages=[ai_msg],
            model_call_count=0,
            accumulated_input_tokens=0,
            accumulated_output_tokens=0,
            accumulated_total_tokens=0,
            accumulated_extra_tokens={},
        )

        with patch(
            "inference_core.agents.middleware.cost_tracking.UsageSession"
        ) as MockSession:
            mock_instance = MagicMock()
            MockSession.return_value = mock_instance

            middleware.after_model(state, MagicMock())

        MockSession.assert_called_once()
        mock_instance.finalize_sync.assert_called_once()

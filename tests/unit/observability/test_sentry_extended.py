"""Extended tests for inference_core.observability.sentry.

Covers error paths, log_operation_start/complete, capture_batch_message,
_is_sentry_enabled edge cases, and exception swallowing in every public method.
"""

from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from inference_core.observability.sentry import (
    BatchSentryIntegration,
    batch_sentry,
    get_batch_sentry,
)

# ============================================================================
# helpers
# ============================================================================


def _make_enabled_sentry() -> BatchSentryIntegration:
    """Build a BatchSentryIntegration with .enabled forced to True."""
    with patch.object(BatchSentryIntegration, "_is_sentry_enabled", return_value=True):
        return BatchSentryIntegration()


# ============================================================================
# _is_sentry_enabled edge cases
# ============================================================================


class TestIsSentryEnabled:
    """Additional edge cases for _is_sentry_enabled."""

    @patch("inference_core.observability.sentry.get_settings")
    def test_returns_false_when_settings_raises(self, mock_settings):
        """get_settings() exception → returns False."""
        mock_settings.side_effect = RuntimeError("no config")

        with patch("inference_core.observability.sentry.SENTRY_AVAILABLE", True):
            sentry = BatchSentryIntegration()
        assert sentry.enabled is False

    @patch("inference_core.observability.sentry.get_settings")
    def test_returns_true_when_dsn_present(self, mock_settings):
        mock_settings.return_value = Mock(sentry_dsn="https://abc@sentry.io/1")
        with patch("inference_core.observability.sentry.SENTRY_AVAILABLE", True):
            sentry = BatchSentryIntegration()
        assert sentry.enabled is True


# ============================================================================
# add_breadcrumb
# ============================================================================


class TestAddBreadcrumb:
    """add_breadcrumb exception path and disabled path."""

    def test_noop_when_disabled(self):
        sentry = BatchSentryIntegration.__new__(BatchSentryIntegration)
        sentry.enabled = False
        # Should not raise
        sentry.add_breadcrumb("test")

    @patch("inference_core.observability.sentry.sentry_sdk")
    def test_swallows_exception(self, mock_sdk):
        """Exception in sentry_sdk.add_breadcrumb → logs warning, no raise."""
        mock_sdk.add_breadcrumb.side_effect = RuntimeError("sdk error")
        sentry = _make_enabled_sentry()
        # Should not raise
        sentry.add_breadcrumb("test msg")


# ============================================================================
# set_batch_context
# ============================================================================


class TestSetBatchContext:
    """set_batch_context exception swallowing."""

    @patch("inference_core.observability.sentry.set_context")
    @patch("inference_core.observability.sentry.set_tag")
    def test_swallows_exception(self, mock_tag, mock_ctx):
        """Exception during set_tag → logs warning, no raise."""
        mock_tag.side_effect = RuntimeError("tag error")
        sentry = _make_enabled_sentry()
        sentry.set_batch_context(job_id="j1", provider="openai")


# ============================================================================
# capture_batch_error
# ============================================================================


class TestCaptureBatchError:
    """capture_batch_error with enriched error attributes and error paths."""

    def test_returns_none_when_disabled(self):
        sentry = BatchSentryIntegration.__new__(BatchSentryIntegration)
        sentry.enabled = False
        assert sentry.capture_batch_error(ValueError("x")) is None

    @patch("inference_core.observability.sentry.capture_exception")
    @patch("inference_core.observability.sentry.set_context")
    def test_includes_provider_and_details(self, mock_ctx, mock_capture):
        """Error with .provider and .details attributes enriches context."""
        mock_capture.return_value = "evt-1"

        class RichError(Exception):
            provider = "anthropic"
            details = {"code": 429}

        sentry = _make_enabled_sentry()
        with patch.object(sentry, "set_batch_context"):
            event_id = sentry.capture_batch_error(RichError("rate limited"))

        assert event_id == "evt-1"
        # set_context should be called with error_details containing extra fields
        ctx_call = mock_ctx.call_args
        assert ctx_call[0][0] == "error_details"
        error_ctx = ctx_call[0][1]
        assert error_ctx["error_provider"] == "anthropic"
        assert error_ctx["error_details"] == {"code": 429}

    @patch("inference_core.observability.sentry.capture_exception")
    @patch("inference_core.observability.sentry.set_context")
    def test_swallows_exception_returns_none(self, mock_ctx, mock_capture):
        """Internal exception → returns None (no raise)."""
        mock_capture.side_effect = RuntimeError("capture failed")
        sentry = _make_enabled_sentry()
        with patch.object(sentry, "set_batch_context"):
            result = sentry.capture_batch_error(ValueError("x"))
        assert result is None


# ============================================================================
# log_operation_start
# ============================================================================


class TestLogOperationStart:
    """log_operation_start happy path and disabled path."""

    def test_noop_when_disabled(self):
        sentry = BatchSentryIntegration.__new__(BatchSentryIntegration)
        sentry.enabled = False
        sentry.log_operation_start("submit")  # no raise

    def test_calls_add_breadcrumb(self):
        sentry = _make_enabled_sentry()
        job_id = str(uuid4())
        with patch.object(sentry, "add_breadcrumb") as mock_bc:
            sentry.log_operation_start(
                "submit",
                job_id=job_id,
                provider="openai",
                provider_batch_id="b1",
                additional_data={"extra": True},
            )
            mock_bc.assert_called_once()
            kwargs = mock_bc.call_args.kwargs
            assert kwargs["category"] == "batch.operation"
            assert kwargs["data"]["operation"] == "submit"
            assert kwargs["data"]["stage"] == "start"
            assert kwargs["data"]["job_id"] == job_id
            assert kwargs["data"]["provider"] == "openai"
            assert kwargs["data"]["extra"] is True


# ============================================================================
# log_operation_complete
# ============================================================================


class TestLogOperationComplete:
    """log_operation_complete level routing and duration handling."""

    def test_noop_when_disabled(self):
        sentry = BatchSentryIntegration.__new__(BatchSentryIntegration)
        sentry.enabled = False
        sentry.log_operation_complete("fetch")  # no raise

    def test_success_uses_info_level(self):
        sentry = _make_enabled_sentry()
        with patch.object(sentry, "add_breadcrumb") as mock_bc:
            sentry.log_operation_complete("fetch", success=True)
            assert mock_bc.call_args.kwargs["level"] == "info"
            assert "completed" in mock_bc.call_args.kwargs["message"]

    def test_failure_uses_warning_level(self):
        sentry = _make_enabled_sentry()
        with patch.object(sentry, "add_breadcrumb") as mock_bc:
            sentry.log_operation_complete("fetch", success=False)
            assert mock_bc.call_args.kwargs["level"] == "warning"
            assert "failed" in mock_bc.call_args.kwargs["message"]

    def test_includes_duration(self):
        sentry = _make_enabled_sentry()
        with patch.object(sentry, "add_breadcrumb") as mock_bc:
            sentry.log_operation_complete("poll", duration_seconds=12.5)
            assert mock_bc.call_args.kwargs["data"]["duration_seconds"] == 12.5

    def test_includes_all_optional_fields(self):
        sentry = _make_enabled_sentry()
        with patch.object(sentry, "add_breadcrumb") as mock_bc:
            sentry.log_operation_complete(
                "poll",
                job_id="j1",
                provider="openai",
                provider_batch_id="b1",
                additional_data={"k": "v"},
            )
            data = mock_bc.call_args.kwargs["data"]
            assert data["job_id"] == "j1"
            assert data["provider"] == "openai"
            assert data["provider_batch_id"] == "b1"
            assert data["k"] == "v"


# ============================================================================
# capture_batch_message
# ============================================================================


class TestCaptureBatchMessage:
    """capture_batch_message happy, disabled, and error paths."""

    def test_returns_none_when_disabled(self):
        sentry = BatchSentryIntegration.__new__(BatchSentryIntegration)
        sentry.enabled = False
        assert sentry.capture_batch_message("test") is None

    @patch("inference_core.observability.sentry.capture_message")
    def test_happy_path(self, mock_capture):
        mock_capture.return_value = "msg-evt-1"
        sentry = _make_enabled_sentry()
        with patch.object(sentry, "set_batch_context"):
            event_id = sentry.capture_batch_message(
                "Important msg",
                level="warning",
                job_id="j1",
                provider="anthropic",
            )
        assert event_id == "msg-evt-1"
        mock_capture.assert_called_once_with("Important msg", "warning")

    @patch("inference_core.observability.sentry.capture_message")
    def test_swallows_exception(self, mock_capture):
        mock_capture.side_effect = RuntimeError("msg capture fail")
        sentry = _make_enabled_sentry()
        with patch.object(sentry, "set_batch_context"):
            result = sentry.capture_batch_message("test")
        assert result is None


# ============================================================================
# Singleton
# ============================================================================


class TestGetBatchSentry:
    """get_batch_sentry returns the global singleton."""

    def test_returns_same_instance(self):
        assert get_batch_sentry() is batch_sentry

    def test_is_batch_sentry_integration(self):
        assert isinstance(get_batch_sentry(), BatchSentryIntegration)

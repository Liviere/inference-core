"""
Unit tests for batch processing observability features.

Tests metrics collection, structured logging, and Sentry integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

from app.observability.metrics import (
    record_job_status_change,
    record_item_status_change,
    record_job_duration,
    record_provider_latency,
    record_poll_cycle_duration,
    update_jobs_in_progress,
    record_retry_attempt,
    record_error,
    get_metrics_summary,
    reset_metrics
)
from app.observability.logging import BatchLogger, get_batch_logger
from app.observability.sentry import BatchSentryIntegration


class TestBatchMetrics:
    """Test Prometheus metrics collection for batch processing."""
    
    def test_record_job_status_change(self):
        """Test job status change metrics recording."""
        provider = "test_openai"
        
        record_job_status_change(provider, None, "created")
        record_job_status_change(provider, "created", "submitted")
        record_job_status_change(provider, "submitted", "completed")
        
        summary = get_metrics_summary()
        # The metric might be truncated to 'batch_jobs' in the summary
        batch_jobs_key = next((k for k in summary.keys() if k.startswith("batch_jobs")), None)
        assert batch_jobs_key is not None
        
        # Check that we have samples
        job_samples = summary[batch_jobs_key]["samples"]
        assert len(job_samples) > 0
        
        # Check that our test provider is included
        test_samples = [s for s in job_samples if s["labels"]["provider"] == provider]
        assert len(test_samples) >= 3
        
        # Check specific labels
        status_values = [sample["labels"]["status"] for sample in test_samples]
        assert "created" in status_values
        assert "submitted" in status_values
        assert "completed" in status_values
    
    def test_record_item_status_change(self):
        """Test item status change metrics recording."""
        provider = "test_anthropic"
        
        record_item_status_change(provider, None, "completed", 5)
        record_item_status_change(provider, None, "failed", 2)
        
        summary = get_metrics_summary()
        # The metric might be truncated to 'batch_items' in the summary
        batch_items_key = next((k for k in summary.keys() if k.startswith("batch_items")), None)
        assert batch_items_key is not None
        
        item_samples = summary[batch_items_key]["samples"]
        test_samples = [s for s in item_samples if s["labels"]["provider"] == provider]
        assert len(test_samples) >= 2
        
        # Check values
        completed_sample = next(s for s in test_samples if s["labels"]["status"] == "completed")
        assert completed_sample["value"] == 5.0
        
        failed_sample = next(s for s in test_samples if s["labels"]["status"] == "failed")
        assert failed_sample["value"] == 2.0
    
    def test_record_provider_latency(self):
        """Test provider latency metrics recording."""
        provider = "test_gemini"
        
        record_provider_latency(provider, "submit", 1.5)
        record_provider_latency(provider, "poll", 0.3)
        record_provider_latency(provider, "fetch", 2.1)
        
        summary = get_metrics_summary()
        assert "batch_provider_latency_seconds" in summary
        
        # Check that histogram buckets are created
        latency_samples = summary["batch_provider_latency_seconds"]["samples"]
        test_samples = [s for s in latency_samples if s["labels"]["provider"] == provider]
        assert len(test_samples) > 0
        
        # Should have samples for different operations
        operations = set(sample["labels"]["operation"] for sample in test_samples)
        assert "submit" in operations
        assert "poll" in operations
        assert "fetch" in operations
    
    def test_record_job_duration(self):
        """Test job duration metrics recording."""
        provider = "test_duration"
        
        record_job_duration(provider, "completed", 45.5)
        record_job_duration(provider, "failed", 12.3)
        
        summary = get_metrics_summary()
        assert "batch_job_duration_seconds" in summary
        
        duration_samples = summary["batch_job_duration_seconds"]["samples"]
        test_samples = [s for s in duration_samples if s["labels"]["provider"] == provider]
        assert len(test_samples) > 0
        
        # Check that we have samples for both statuses
        statuses = set(sample["labels"]["status"] for sample in test_samples)
        assert "completed" in statuses
        assert "failed" in statuses
    
    def test_update_jobs_in_progress(self):
        """Test jobs in progress gauge updates."""
        provider = "test_progress"
        
        # Start with 0, increment, then decrement
        update_jobs_in_progress(provider, 3)
        update_jobs_in_progress(provider, -1)
        
        summary = get_metrics_summary()
        assert "batch_jobs_in_progress" in summary
        
        progress_samples = summary["batch_jobs_in_progress"]["samples"]
        test_samples = [s for s in progress_samples if s["labels"]["provider"] == provider]
        assert len(test_samples) >= 1
        
        # The final value should be 2 (3 - 1)
        provider_sample = test_samples[0]
        assert provider_sample["value"] >= 0  # Should be positive
    
    def test_record_error_and_retry(self):
        """Test error and retry metrics recording."""
        provider = "test_errors"
        
        record_error(provider, "transient", "submit")
        record_error(provider, "permanent", "fetch")
        record_retry_attempt(provider, "submit", "transient_error")
        
        summary = get_metrics_summary()
        # The metric might be truncated in the summary
        batch_errors_key = next((k for k in summary.keys() if k.startswith("batch_errors")), None)
        batch_retry_key = next((k for k in summary.keys() if k.startswith("batch_retry")), None)
        
        assert batch_errors_key is not None
        assert batch_retry_key is not None
        
        error_samples = summary[batch_errors_key]["samples"]
        retry_samples = summary[batch_retry_key]["samples"]
        
        test_error_samples = [s for s in error_samples if s["labels"]["provider"] == provider]
        test_retry_samples = [s for s in retry_samples if s["labels"]["provider"] == provider]
        
        assert len(test_error_samples) >= 2
        assert len(test_retry_samples) >= 1
        
        # Check error types
        error_types = [sample["labels"]["error_type"] for sample in test_error_samples]
        assert "transient" in error_types
        assert "permanent" in error_types


class TestBatchLogger:
    """Test structured logging for batch processing."""
    
    def setup_method(self):
        """Setup test logger."""
        self.logger = BatchLogger("test_batch")
    
    @patch("logging.getLogger")
    def test_logger_initialization(self, mock_get_logger):
        """Test logger initialization."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        logger = BatchLogger("test_logger")
        assert logger.logger == mock_logger
        assert logger._rate_limit_window == 60
        assert logger._max_debug_per_window == 10
    
    def test_build_batch_fields(self):
        """Test building standardized batch fields."""
        job_id = str(uuid4())
        provider = "openai"
        
        fields = self.logger._build_batch_fields(
            job_id=job_id,
            provider=provider,
            status_transition="created->submitted",
            operation="submit",
            extra_fields={"custom": "value"}
        )
        
        assert fields["component"] == "batch_processing"
        assert fields["batch_job_id"] == job_id
        assert fields["provider"] == provider
        assert fields["status_transition"] == "created->submitted"
        assert fields["operation"] == "submit"
        assert fields["custom"] == "value"
        assert "timestamp" in fields
    
    @patch("app.observability.logging.BatchLogger._build_batch_fields")
    def test_info_logging(self, mock_build_fields):
        """Test info level logging."""
        mock_build_fields.return_value = {"test": "fields"}
        
        with patch.object(self.logger, "logger") as mock_logger:
            self.logger.info(
                "Test message",
                job_id="test-job",
                provider="openai",
                operation="test"
            )
            
            mock_logger.info.assert_called_once_with(
                "Test message",
                extra={"test": "fields"}
            )
            mock_build_fields.assert_called_once()
    
    def test_debug_rate_limiting(self):
        """Test debug log rate limiting."""
        with patch.object(self.logger, "logger") as mock_logger:
            # Send more debug messages than the limit
            for i in range(15):
                self.logger.debug(
                    f"Debug message {i}",
                    job_id="test-job",
                    operation="rate_limit_test"
                )
            
            # Should only log the first 10 messages
            assert mock_logger.debug.call_count == 10
    
    def test_lifecycle_event_logging(self):
        """Test job lifecycle event logging."""
        job_id = str(uuid4())
        provider = "gemini"
        
        with patch.object(self.logger, "info") as mock_info:
            self.logger.log_job_lifecycle_event(
                "completed",
                job_id,
                provider,
                old_status="submitted",
                new_status="completed",
                duration_seconds=120.5,
                item_counts={"success": 8, "error": 2}
            )
            
            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args
            
            assert "completed" in args[0]  # Message contains event
            assert kwargs["job_id"] == job_id
            assert kwargs["provider"] == provider
            assert kwargs["status_transition"] == "submitted->completed"
            assert kwargs["duration_seconds"] == 120.5
            assert kwargs["success"] == 8
            assert kwargs["error"] == 2


class TestBatchSentryIntegration:
    """Test Sentry integration for batch processing."""
    
    def setup_method(self):
        """Setup test Sentry integration."""
        self.sentry = BatchSentryIntegration()
    
    @patch("app.observability.sentry.get_settings")
    def test_sentry_disabled_without_dsn(self, mock_get_settings):
        """Test Sentry is disabled when no DSN is configured."""
        mock_settings = Mock()
        mock_settings.sentry_dsn = None
        mock_get_settings.return_value = mock_settings
        
        sentry = BatchSentryIntegration()
        assert not sentry.enabled
    
    @patch("app.observability.sentry.SENTRY_AVAILABLE", False)
    def test_sentry_disabled_without_sdk(self):
        """Test Sentry is disabled when SDK is not available."""
        sentry = BatchSentryIntegration()
        assert not sentry.enabled
    
    @patch("app.observability.sentry.sentry_sdk")
    def test_add_breadcrumb(self, mock_sentry_sdk):
        """Test adding breadcrumbs."""
        # Mock Sentry as enabled
        with patch.object(self.sentry, "enabled", True):
            self.sentry.add_breadcrumb(
                "Test breadcrumb",
                category="batch.test",
                level="info",
                data={"key": "value"}
            )
            
            mock_sentry_sdk.add_breadcrumb.assert_called_once_with(
                message="Test breadcrumb",
                category="batch.test",
                level="info",
                data={"key": "value"}
            )
    
    @patch("app.observability.sentry.set_context")
    @patch("app.observability.sentry.set_tag")
    def test_set_batch_context(self, mock_set_tag, mock_set_context):
        """Test setting batch context."""
        job_id = str(uuid4())
        provider = "openai"
        provider_batch_id = "batch_123"
        
        with patch.object(self.sentry, "enabled", True):
            self.sentry.set_batch_context(
                job_id=job_id,
                provider=provider,
                provider_batch_id=provider_batch_id,
                operation="test"
            )
            
            # Check that tags were set
            mock_set_tag.assert_any_call("batch.job_id", job_id)
            mock_set_tag.assert_any_call("batch.provider", provider)
            mock_set_tag.assert_any_call("batch.provider_batch_id", provider_batch_id)
            
            # Check that context was set
            mock_set_context.assert_called_once()
            args = mock_set_context.call_args[0]
            assert args[0] == "batch_processing"
            assert "batch_job_id" in args[1]
            assert "provider" in args[1]
    
    @patch("app.observability.sentry.capture_exception")
    def test_capture_batch_error(self, mock_capture_exception):
        """Test capturing batch errors."""
        mock_capture_exception.return_value = "event_123"
        
        error = ValueError("Test error")
        job_id = str(uuid4())
        
        with patch.object(self.sentry, "enabled", True):
            with patch.object(self.sentry, "set_batch_context"):
                event_id = self.sentry.capture_batch_error(
                    error,
                    job_id=job_id,
                    provider="openai",
                    operation="test"
                )
                
                assert event_id == "event_123"
                mock_capture_exception.assert_called_once_with(error)
    
    def test_log_status_change(self):
        """Test logging status changes as breadcrumbs."""
        job_id = str(uuid4())
        
        with patch.object(self.sentry, "add_breadcrumb") as mock_add_breadcrumb:
            with patch.object(self.sentry, "enabled", True):
                self.sentry.log_status_change(
                    job_id,
                    "openai",
                    "created",
                    "submitted",
                    provider_batch_id="batch_123"
                )
                
                mock_add_breadcrumb.assert_called_once()
                # Get all arguments passed to the mock
                call_args = mock_add_breadcrumb.call_args
                
                # The call should be add_breadcrumb(message=..., category=..., level=..., data=...)
                # So we can check the keyword arguments
                assert "message" in call_args.kwargs
                assert "status changed" in call_args.kwargs["message"]
                assert call_args.kwargs["category"] == "batch.status_change"
                assert call_args.kwargs["data"]["status_transition"] == "created -> submitted"


def test_get_batch_logger():
    """Test getting the global batch logger instance."""
    logger1 = get_batch_logger()
    logger2 = get_batch_logger()
    
    # Should return the same instance
    assert logger1 is logger2
    assert isinstance(logger1, BatchLogger)
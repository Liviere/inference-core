"""
Structured logging for batch processing operations.

Provides a specialized logger with standardized fields for batch operations,
rate limiting for debug messages, and integration with existing JSON logging.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
from uuid import UUID

from inference_core.core.config import get_settings


class BatchLogger:
    """Specialized logger for batch processing with standardized fields."""

    def __init__(self, logger_name: str = "batch_processing"):
        self.logger = logging.getLogger(logger_name)
        self._debug_rate_limiter = {}
        self._rate_limit_window = 60  # 1 minute
        self._max_debug_per_window = 10  # Max debug logs per item per minute

    def _should_log_debug(self, rate_limit_key: str) -> bool:
        """Check if debug log should be emitted based on rate limiting.

        Args:
            rate_limit_key: Key to use for rate limiting (e.g., job_id + operation)

        Returns:
            True if the log should be emitted, False if rate limited
        """
        current_time = time.time()

        # Clean up old entries
        expired_keys = [
            key
            for key, (count, window_start) in self._debug_rate_limiter.items()
            if current_time - window_start > self._rate_limit_window
        ]
        for key in expired_keys:
            del self._debug_rate_limiter[key]

        # Check current key
        if rate_limit_key not in self._debug_rate_limiter:
            self._debug_rate_limiter[rate_limit_key] = (1, current_time)
            return True

        count, window_start = self._debug_rate_limiter[rate_limit_key]

        # If we're in a new window, reset
        if current_time - window_start > self._rate_limit_window:
            self._debug_rate_limiter[rate_limit_key] = (1, current_time)
            return True

        # Check if we've exceeded the limit
        if count >= self._max_debug_per_window:
            return False

        # Increment and allow
        self._debug_rate_limiter[rate_limit_key] = (count + 1, window_start)
        return True

    def _build_batch_fields(
        self,
        job_id: Optional[Union[str, UUID]] = None,
        provider: Optional[str] = None,
        status_transition: Optional[str] = None,
        operation: Optional[str] = None,
        item_id: Optional[Union[str, UUID]] = None,
        provider_batch_id: Optional[str] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build standardized batch logging fields.

        Args:
            job_id: Batch job UUID
            provider: LLM provider name
            status_transition: Status change description (e.g., "created->submitted")
            operation: Operation being performed (submit, poll, fetch, etc.)
            item_id: Batch item UUID (for item-level logs)
            provider_batch_id: External provider's batch ID
            extra_fields: Additional custom fields

        Returns:
            Dictionary of structured logging fields
        """
        fields = {
            "component": "batch_processing",
            "batch_job": True,  # Add explicit filter tag for batch job logs
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if job_id is not None:
            fields["batch_job_id"] = str(job_id)

        if provider is not None:
            fields["provider"] = provider

        if status_transition is not None:
            fields["status_transition"] = status_transition

        if operation is not None:
            fields["operation"] = operation

        if item_id is not None:
            fields["batch_item_id"] = str(item_id)

        if provider_batch_id is not None:
            fields["provider_batch_id"] = provider_batch_id

        if extra_fields:
            fields.update(extra_fields)

        return fields

    def info(
        self,
        message: str,
        job_id: Optional[Union[str, UUID]] = None,
        provider: Optional[str] = None,
        status_transition: Optional[str] = None,
        operation: Optional[str] = None,
        item_id: Optional[Union[str, UUID]] = None,
        provider_batch_id: Optional[str] = None,
        **extra_fields,
    ):
        """Log an info message with batch context.

        Args:
            message: Log message
            job_id: Batch job UUID
            provider: LLM provider name
            status_transition: Status change description
            operation: Operation being performed
            item_id: Batch item UUID
            provider_batch_id: External provider's batch ID
            **extra_fields: Additional fields to include
        """
        fields = self._build_batch_fields(
            job_id=job_id,
            provider=provider,
            status_transition=status_transition,
            operation=operation,
            item_id=item_id,
            provider_batch_id=provider_batch_id,
            extra_fields=extra_fields,
        )

        self.logger.info(message, extra=fields)

    def warning(
        self,
        message: str,
        job_id: Optional[Union[str, UUID]] = None,
        provider: Optional[str] = None,
        status_transition: Optional[str] = None,
        operation: Optional[str] = None,
        item_id: Optional[Union[str, UUID]] = None,
        provider_batch_id: Optional[str] = None,
        **extra_fields,
    ):
        """Log a warning message with batch context."""
        fields = self._build_batch_fields(
            job_id=job_id,
            provider=provider,
            status_transition=status_transition,
            operation=operation,
            item_id=item_id,
            provider_batch_id=provider_batch_id,
            extra_fields=extra_fields,
        )

        self.logger.warning(message, extra=fields)

    def error(
        self,
        message: str,
        job_id: Optional[Union[str, UUID]] = None,
        provider: Optional[str] = None,
        status_transition: Optional[str] = None,
        operation: Optional[str] = None,
        item_id: Optional[Union[str, UUID]] = None,
        provider_batch_id: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
        **extra_fields,
    ):
        """Log an error message with batch context.

        Args:
            message: Error message
            job_id: Batch job UUID
            provider: LLM provider name
            status_transition: Status change description
            operation: Operation that failed
            item_id: Batch item UUID
            provider_batch_id: External provider's batch ID
            error_details: Additional error information
            **extra_fields: Additional fields to include
        """
        fields = self._build_batch_fields(
            job_id=job_id,
            provider=provider,
            status_transition=status_transition,
            operation=operation,
            item_id=item_id,
            provider_batch_id=provider_batch_id,
            extra_fields=extra_fields,
        )

        if error_details:
            fields["error_details"] = error_details

        self.logger.error(message, extra=fields)

    def debug(
        self,
        message: str,
        job_id: Optional[Union[str, UUID]] = None,
        provider: Optional[str] = None,
        operation: Optional[str] = None,
        item_id: Optional[Union[str, UUID]] = None,
        provider_batch_id: Optional[str] = None,
        rate_limit_key: Optional[str] = None,
        **extra_fields,
    ):
        """Log a debug message with batch context and optional rate limiting.

        Args:
            message: Debug message
            job_id: Batch job UUID
            provider: LLM provider name
            operation: Operation being performed
            item_id: Batch item UUID
            provider_batch_id: External provider's batch ID
            rate_limit_key: Key for rate limiting (if None, uses job_id + operation)
            **extra_fields: Additional fields to include
        """
        # Build rate limit key
        if rate_limit_key is None:
            key_parts = []
            if job_id:
                key_parts.append(str(job_id))
            if operation:
                key_parts.append(operation)
            if item_id:
                key_parts.append(str(item_id))
            rate_limit_key = ":".join(key_parts) if key_parts else "default"

        # Check rate limiting for debug logs
        if not self._should_log_debug(rate_limit_key):
            return

        fields = self._build_batch_fields(
            job_id=job_id,
            provider=provider,
            operation=operation,
            item_id=item_id,
            provider_batch_id=provider_batch_id,
            extra_fields=extra_fields,
        )

        self.logger.debug(message, extra=fields)

    def log_job_lifecycle_event(
        self,
        event: str,
        job_id: Union[str, UUID],
        provider: str,
        old_status: Optional[str] = None,
        new_status: Optional[str] = None,
        provider_batch_id: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        item_counts: Optional[Dict[str, int]] = None,
        **extra_fields,
    ):
        """Log a batch job lifecycle event with standardized fields.

        Args:
            event: Event name (submitted, status_changed, completed, failed)
            job_id: Batch job UUID
            provider: LLM provider name
            old_status: Previous status (for status changes)
            new_status: New status (for status changes)
            provider_batch_id: External provider's batch ID
            duration_seconds: Duration for completed jobs
            item_counts: Item count breakdown (success, error, pending)
            **extra_fields: Additional fields
        """
        status_transition = None
        if old_status and new_status:
            status_transition = f"{old_status}->{new_status}"
        elif new_status:
            status_transition = f"init->{new_status}"

        log_fields = extra_fields.copy()
        log_fields["event"] = event

        if duration_seconds is not None:
            log_fields["duration_seconds"] = duration_seconds

        if item_counts:
            log_fields.update(item_counts)

        self.info(
            f"Batch job {event}: {job_id}",
            job_id=job_id,
            provider=provider,
            status_transition=status_transition,
            operation="lifecycle",
            provider_batch_id=provider_batch_id,
            **log_fields,
        )

    def log_item_batch_update(
        self,
        job_id: Union[str, UUID],
        provider: str,
        success_count: int,
        error_count: int,
        total_count: int,
        provider_batch_id: Optional[str] = None,
        **extra_fields,
    ):
        """Log batch item updates with counts.

        Args:
            job_id: Batch job UUID
            provider: LLM provider name
            success_count: Number of successful items
            error_count: Number of failed items
            total_count: Total number of items processed
            provider_batch_id: External provider's batch ID
            **extra_fields: Additional fields
        """
        log_fields = extra_fields.copy()
        log_fields.update(
            {
                "success_count": success_count,
                "error_count": error_count,
                "total_count": total_count,
                "success_rate": (
                    (success_count / total_count * 100) if total_count > 0 else 0
                ),
            }
        )

        self.info(
            f"Batch items updated: {success_count} success, {error_count} error out of {total_count}",
            job_id=job_id,
            provider=provider,
            operation="item_update",
            provider_batch_id=provider_batch_id,
            **log_fields,
        )


# Global batch logger instance
batch_logger = BatchLogger()


def get_batch_logger() -> BatchLogger:
    """Get the global batch logger instance.

    Returns:
        BatchLogger instance configured for the application
    """
    return batch_logger

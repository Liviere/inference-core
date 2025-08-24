"""
Sentry integration for batch processing observability.

Provides optional Sentry breadcrumbs and error capture with enriched context
for batch processing operations.
"""

import logging
from typing import Any, Dict, Optional, Union
from uuid import UUID

try:
    import sentry_sdk
    from sentry_sdk import capture_exception, capture_message, set_context, set_tag

    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

from inference_core.core.config import get_settings

logger = logging.getLogger(__name__)


class BatchSentryIntegration:
    """Sentry integration for batch processing operations."""

    def __init__(self):
        self.enabled = self._is_sentry_enabled()
        if not self.enabled:
            logger.debug(
                "Sentry integration disabled - no DSN configured or sentry_sdk not available"
            )

    def _is_sentry_enabled(self) -> bool:
        """Check if Sentry is enabled and configured."""
        if not SENTRY_AVAILABLE:
            return False

        try:
            settings = get_settings()
            return bool(settings.sentry_dsn)
        except Exception:
            return False

    def add_breadcrumb(
        self,
        message: str,
        category: str = "batch",
        level: str = "info",
        data: Optional[Dict[str, Any]] = None,
    ):
        """Add a breadcrumb to Sentry for batch operations.

        Args:
            message: Breadcrumb message
            category: Breadcrumb category (defaults to "batch")
            level: Log level (info, warning, error)
            data: Additional data to include
        """
        if not self.enabled:
            return

        try:
            sentry_sdk.add_breadcrumb(
                message=message, category=category, level=level, data=data or {}
            )
        except Exception as e:
            logger.warning(f"Failed to add Sentry breadcrumb: {e}")

    def set_batch_context(
        self,
        job_id: Optional[Union[str, UUID]] = None,
        provider: Optional[str] = None,
        provider_batch_id: Optional[str] = None,
        item_id: Optional[Union[str, UUID]] = None,
        operation: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ):
        """Set batch-specific context in Sentry.

        Args:
            job_id: Batch job UUID
            provider: LLM provider name
            provider_batch_id: External provider's batch ID
            item_id: Batch item UUID
            operation: Current operation
            additional_context: Additional context data
        """
        if not self.enabled:
            return

        try:
            context = {}

            if job_id is not None:
                context["batch_job_id"] = str(job_id)
                set_tag("batch.job_id", str(job_id))

            if provider is not None:
                context["provider"] = provider
                set_tag("batch.provider", provider)

            if provider_batch_id is not None:
                context["provider_batch_id"] = provider_batch_id
                set_tag("batch.provider_batch_id", provider_batch_id)

            if item_id is not None:
                context["batch_item_id"] = str(item_id)
                set_tag("batch.item_id", str(item_id))

            if operation is not None:
                context["operation"] = operation
                set_tag("batch.operation", operation)

            if additional_context:
                context.update(additional_context)

            set_context("batch_processing", context)

        except Exception as e:
            logger.warning(f"Failed to set Sentry batch context: {e}")

    def capture_batch_error(
        self,
        error: Exception,
        job_id: Optional[Union[str, UUID]] = None,
        provider: Optional[str] = None,
        provider_batch_id: Optional[str] = None,
        item_id: Optional[Union[str, UUID]] = None,
        operation: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Capture an error with batch context.

        Args:
            error: Exception to capture
            job_id: Batch job UUID
            provider: LLM provider name
            provider_batch_id: External provider's batch ID
            item_id: Batch item UUID
            operation: Operation that failed
            additional_context: Additional context data

        Returns:
            Sentry event ID if captured, None otherwise
        """
        if not self.enabled:
            return None

        try:
            # Set batch context
            self.set_batch_context(
                job_id=job_id,
                provider=provider,
                provider_batch_id=provider_batch_id,
                item_id=item_id,
                operation=operation,
                additional_context=additional_context,
            )

            # Add additional error context
            error_context = {
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

            if hasattr(error, "provider"):
                error_context["error_provider"] = error.provider

            if hasattr(error, "details"):
                error_context["error_details"] = error.details

            set_context("error_details", error_context)

            # Capture the exception
            event_id = capture_exception(error)
            return event_id

        except Exception as e:
            logger.warning(f"Failed to capture error in Sentry: {e}")
            return None

    def log_status_change(
        self,
        job_id: Union[str, UUID],
        provider: str,
        old_status: Optional[str],
        new_status: str,
        provider_batch_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """Log a batch job status change as a Sentry breadcrumb.

        Args:
            job_id: Batch job UUID
            provider: LLM provider name
            old_status: Previous status
            new_status: New status
            provider_batch_id: External provider's batch ID
            additional_data: Additional data to include
        """
        if not self.enabled:
            return

        status_transition = f"{old_status or 'init'} -> {new_status}"

        breadcrumb_data = {
            "job_id": str(job_id),
            "provider": provider,
            "status_transition": status_transition,
            "old_status": old_status,
            "new_status": new_status,
        }

        if provider_batch_id:
            breadcrumb_data["provider_batch_id"] = provider_batch_id

        if additional_data:
            breadcrumb_data.update(additional_data)

        self.add_breadcrumb(
            message=f"Batch job status changed: {status_transition}",
            category="batch.status_change",
            level="info",
            data=breadcrumb_data,
        )

    def log_operation_start(
        self,
        operation: str,
        job_id: Optional[Union[str, UUID]] = None,
        provider: Optional[str] = None,
        provider_batch_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """Log the start of a batch operation.

        Args:
            operation: Operation name (submit, poll, fetch, etc.)
            job_id: Batch job UUID
            provider: LLM provider name
            provider_batch_id: External provider's batch ID
            additional_data: Additional data to include
        """
        if not self.enabled:
            return

        breadcrumb_data = {"operation": operation, "stage": "start"}

        if job_id:
            breadcrumb_data["job_id"] = str(job_id)

        if provider:
            breadcrumb_data["provider"] = provider

        if provider_batch_id:
            breadcrumb_data["provider_batch_id"] = provider_batch_id

        if additional_data:
            breadcrumb_data.update(additional_data)

        self.add_breadcrumb(
            message=f"Batch operation started: {operation}",
            category="batch.operation",
            level="info",
            data=breadcrumb_data,
        )

    def log_operation_complete(
        self,
        operation: str,
        job_id: Optional[Union[str, UUID]] = None,
        provider: Optional[str] = None,
        provider_batch_id: Optional[str] = None,
        success: bool = True,
        duration_seconds: Optional[float] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """Log the completion of a batch operation.

        Args:
            operation: Operation name
            job_id: Batch job UUID
            provider: LLM provider name
            provider_batch_id: External provider's batch ID
            success: Whether the operation succeeded
            duration_seconds: Operation duration
            additional_data: Additional data to include
        """
        if not self.enabled:
            return

        breadcrumb_data = {
            "operation": operation,
            "stage": "complete",
            "success": success,
        }

        if job_id:
            breadcrumb_data["job_id"] = str(job_id)

        if provider:
            breadcrumb_data["provider"] = provider

        if provider_batch_id:
            breadcrumb_data["provider_batch_id"] = provider_batch_id

        if duration_seconds is not None:
            breadcrumb_data["duration_seconds"] = duration_seconds

        if additional_data:
            breadcrumb_data.update(additional_data)

        level = "info" if success else "warning"
        status = "completed" if success else "failed"

        self.add_breadcrumb(
            message=f"Batch operation {status}: {operation}",
            category="batch.operation",
            level=level,
            data=breadcrumb_data,
        )

    def capture_batch_message(
        self,
        message: str,
        level: str = "info",
        job_id: Optional[Union[str, UUID]] = None,
        provider: Optional[str] = None,
        provider_batch_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Capture a message with batch context.

        Args:
            message: Message to capture
            level: Message level (info, warning, error)
            job_id: Batch job UUID
            provider: LLM provider name
            provider_batch_id: External provider's batch ID
            additional_context: Additional context data

        Returns:
            Sentry event ID if captured, None otherwise
        """
        if not self.enabled:
            return None

        try:
            # Set batch context
            self.set_batch_context(
                job_id=job_id,
                provider=provider,
                provider_batch_id=provider_batch_id,
                additional_context=additional_context,
            )

            # Capture the message
            event_id = capture_message(message, level)
            return event_id

        except Exception as e:
            logger.warning(f"Failed to capture message in Sentry: {e}")
            return None


# Global Sentry integration instance
batch_sentry = BatchSentryIntegration()


def get_batch_sentry() -> BatchSentryIntegration:
    """Get the global batch Sentry integration instance.

    Returns:
        BatchSentryIntegration instance
    """
    return batch_sentry

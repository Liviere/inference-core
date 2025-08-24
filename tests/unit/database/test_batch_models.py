"""
Unit Tests for Batch Models

Tests for BatchJob, BatchItem, and BatchEvent models including
CRUD operations, relationships, and computed properties.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from inference_core.database.sql.models.batch import (
    BatchEvent,
    BatchEventType,
    BatchItem,
    BatchItemStatus,
    BatchJob,
    BatchJobStatus,
)


class TestBatchJobModel:
    """Test BatchJob model functionality"""

    def test_batch_job_creation(self):
        """Test creating a BatchJob instance"""
        job = BatchJob(
            provider="openai",
            model="gpt-4",
            status=BatchJobStatus.CREATED,
            mode="chat",
            request_count=10,
        )

        assert job.provider == "openai"
        assert job.model == "gpt-4"
        assert job.status == BatchJobStatus.CREATED
        assert job.mode == "chat"
        assert job.request_count == 10
        assert job.success_count == 0
        assert job.error_count == 0

    def test_batch_job_computed_properties(self):
        """Test BatchJob computed properties"""
        job = BatchJob(
            provider="openai",
            model="gpt-4",
            status=BatchJobStatus.IN_PROGRESS,
            mode="chat",
            request_count=10,
            success_count=7,
            error_count=2,
        )

        # Test completion rate: (7 + 2) / 10 * 100 = 90%
        assert job.completion_rate == 90.0

        # Test success rate: 7 / (7 + 2) * 100 = 77.78%
        assert abs(job.success_rate - 77.77777777777779) < 0.001

        # Test pending count: 10 - 7 - 2 = 1
        assert job.pending_count == 1

        # Test is_complete
        assert not job.is_complete

        job.status = BatchJobStatus.COMPLETED
        assert job.is_complete

    def test_batch_job_completion_rate_edge_cases(self):
        """Test completion rate with edge cases"""
        # No requests
        job = BatchJob(
            provider="openai",
            model="gpt-4",
            status=BatchJobStatus.CREATED,
            mode="chat",
            request_count=0,
        )
        assert job.completion_rate == 0.0
        assert job.success_rate == 0.0

    def test_batch_job_success_rate_edge_cases(self):
        """Test success rate with edge cases"""
        # No completed items
        job = BatchJob(
            provider="openai",
            model="gpt-4",
            status=BatchJobStatus.CREATED,
            mode="chat",
            request_count=10,
            success_count=0,
            error_count=0,
        )
        assert job.success_rate == 0.0

    def test_batch_job_update_counts(self):
        """Test updating success and error counts"""
        job = BatchJob(
            provider="openai",
            model="gpt-4",
            status=BatchJobStatus.IN_PROGRESS,
            mode="chat",
            request_count=10,
            success_count=5,
            error_count=1,
        )

        job.update_counts(success_delta=2, error_delta=1)

        assert job.success_count == 7
        assert job.error_count == 2

    def test_batch_job_repr(self):
        """Test BatchJob string representation"""
        job_id = uuid4()
        job = BatchJob(
            id=job_id,
            provider="openai",
            model="gpt-4",
            status=BatchJobStatus.CREATED,
            mode="chat",
            request_count=10,
        )

        expected = f"<BatchJob(id={job_id}, provider=openai, status=BatchJobStatus.CREATED, requests=10)>"
        assert repr(job) == expected


class TestBatchItemModel:
    """Test BatchItem model functionality"""

    def test_batch_item_creation(self):
        """Test creating a BatchItem instance"""
        job_id = uuid4()
        item = BatchItem(
            batch_job_id=job_id,
            sequence_index=1,
            custom_external_id="ext-123",
            input_payload={"message": "test"},
            status=BatchItemStatus.QUEUED,
        )

        assert item.batch_job_id == job_id
        assert item.sequence_index == 1
        assert item.custom_external_id == "ext-123"
        assert item.input_payload == {"message": "test"}
        assert item.status == BatchItemStatus.QUEUED

    def test_batch_item_properties(self):
        """Test BatchItem computed properties"""
        job_id = uuid4()

        # Queued item
        item = BatchItem(
            batch_job_id=job_id,
            sequence_index=1,
            status=BatchItemStatus.QUEUED,
        )
        assert not item.is_completed
        assert not item.is_successful

        # Completed item
        item.status = BatchItemStatus.COMPLETED
        assert item.is_completed
        assert item.is_successful

        # Failed item
        item.status = BatchItemStatus.FAILED
        assert item.is_completed
        assert not item.is_successful

    def test_batch_item_repr(self):
        """Test BatchItem string representation"""
        item_id = uuid4()
        job_id = uuid4()
        item = BatchItem(
            id=item_id,
            batch_job_id=job_id,
            sequence_index=1,
            status=BatchItemStatus.QUEUED,
        )

        expected = f"<BatchItem(id={item_id}, batch_job_id={job_id}, sequence=1, status=BatchItemStatus.QUEUED)>"
        assert repr(item) == expected


class TestBatchEventModel:
    """Test BatchEvent model functionality"""

    def test_batch_event_creation(self):
        """Test creating a BatchEvent instance"""
        job_id = uuid4()
        event_time = datetime.now(timezone.utc)

        event = BatchEvent(
            batch_job_id=job_id,
            event_type=BatchEventType.STATUS_CHANGE,
            old_status="created",
            new_status="submitted",
            event_timestamp=event_time,
            event_data={"reason": "manual submission"},
        )

        assert event.batch_job_id == job_id
        assert event.event_type == BatchEventType.STATUS_CHANGE
        assert event.old_status == "created"
        assert event.new_status == "submitted"
        assert event.event_timestamp == event_time
        assert event.event_data == {"reason": "manual submission"}

    def test_batch_event_repr(self):
        """Test BatchEvent string representation"""
        event_id = uuid4()
        job_id = uuid4()
        event_time = datetime.now(timezone.utc)

        event = BatchEvent(
            id=event_id,
            batch_job_id=job_id,
            event_type=BatchEventType.STATUS_CHANGE,
            event_timestamp=event_time,
        )

        expected = f"<BatchEvent(id={event_id}, batch_job_id={job_id}, type=BatchEventType.STATUS_CHANGE, timestamp={event_time})>"
        assert repr(event) == expected


class TestBatchModelEnums:
    """Test batch model enumerations"""

    def test_batch_job_status_enum(self):
        """Test BatchJobStatus enum values"""
        assert BatchJobStatus.CREATED == "created"
        assert BatchJobStatus.SUBMITTED == "submitted"
        assert BatchJobStatus.IN_PROGRESS == "in_progress"
        assert BatchJobStatus.COMPLETED == "completed"
        assert BatchJobStatus.FAILED == "failed"
        assert BatchJobStatus.CANCELLED == "cancelled"

    def test_batch_item_status_enum(self):
        """Test BatchItemStatus enum values"""
        assert BatchItemStatus.QUEUED == "queued"
        assert BatchItemStatus.SENT == "sent"
        assert BatchItemStatus.COMPLETED == "completed"
        assert BatchItemStatus.FAILED == "failed"

    def test_batch_event_type_enum(self):
        """Test BatchEventType enum values"""
        assert BatchEventType.STATUS_CHANGE == "status_change"
        assert BatchEventType.ITEM_UPDATE == "item_update"
        assert BatchEventType.ERROR == "error"
        assert BatchEventType.PROGRESS_UPDATE == "progress_update"


class TestBatchModelDefaults:
    """Test default values for batch models"""

    def test_batch_job_defaults(self):
        """Test BatchJob default values"""
        job = BatchJob(
            provider="test",
            model="test-model",
            mode="test",
        )

        # These should have defaults from the column definitions
        assert job.request_count == 0
        assert job.success_count == 0
        assert job.error_count == 0

    def test_batch_item_defaults(self):
        """Test BatchItem default values"""
        job_id = uuid4()
        item = BatchItem(
            batch_job_id=job_id,
            sequence_index=1,
        )

        # Status should default to QUEUED
        assert item.status == BatchItemStatus.QUEUED


# Integration tests would go here when testing with actual database
# For now, these test the model logic without database persistence

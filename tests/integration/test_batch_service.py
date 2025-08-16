"""
Integration Tests for Batch Service

Tests for BatchService with actual database operations.
Tests CRUD operations, relationships, and business logic.
"""

from datetime import UTC, datetime

import pytest

from app.database.sql.models.batch import (
    BatchEventType,
    BatchItemStatus,
    BatchJobStatus,
)
from app.schemas.batch import (
    BatchEventCreate,
    BatchItemCreate,
    BatchItemUpdate,
    BatchJobCreate,
    BatchJobQuery,
    BatchJobUpdate,
)
from app.services.batch_service import BatchService


@pytest.mark.integration
class TestBatchServiceCRUD:
    """Test basic CRUD operations for batch service"""

    @pytest.mark.asyncio
    async def test_create_batch_job(self, async_session_with_engine):
        """Test creating a batch job"""
        async_session, _ = async_session_with_engine
        service = BatchService(async_session)

        job_data = BatchJobCreate(
            provider="openai",
            model="gpt-4",
            mode="chat",
            request_count=10,
            config_json={"temperature": 0.7},
        )

        job = await service.create_batch_job(job_data)

        assert job.id is not None
        assert job.provider == "openai"
        assert job.model == "gpt-4"
        assert job.mode == "chat"
        assert job.status == BatchJobStatus.CREATED
        assert job.request_count == 10
        assert job.config_json == {"temperature": 0.7}
        assert job.success_count == 0
        assert job.error_count == 0

    @pytest.mark.asyncio
    async def test_get_batch_job(self, async_session_with_engine):
        """Test retrieving a batch job"""
        async_session, _ = async_session_with_engine
        service = BatchService(async_session)

        # Create job
        job_data = BatchJobCreate(
            provider="anthropic", model="claude-3", mode="chat", request_count=5
        )
        created_job = await service.create_batch_job(job_data)

        # Retrieve job
        retrieved_job = await service.get_batch_job(created_job.id)

        assert retrieved_job is not None
        assert retrieved_job.id == created_job.id
        assert retrieved_job.provider == "anthropic"
        assert retrieved_job.model == "claude-3"

    @pytest.mark.asyncio
    async def test_update_batch_job(self, async_session_with_engine):
        """Test updating a batch job"""
        async_session, _ = async_session_with_engine
        service = BatchService(async_session)

        # Create job
        job_data = BatchJobCreate(
            provider="openai", model="gpt-4", mode="chat", request_count=10
        )
        job = await service.create_batch_job(job_data)

        # Update job
        update_data = BatchJobUpdate(
            status=BatchJobStatus.SUBMITTED,
            provider_batch_id="ext-batch-123",
            submitted_at=datetime.now(tz=UTC),
            success_count=3,
            error_count=1,
        )

        updated_job = await service.update_batch_job(job.id, update_data)

        assert updated_job is not None
        assert updated_job.status == BatchJobStatus.SUBMITTED
        assert updated_job.provider_batch_id == "ext-batch-123"
        assert updated_job.submitted_at is not None
        assert updated_job.success_count == 3
        assert updated_job.error_count == 1

    @pytest.mark.asyncio
    async def test_delete_batch_job(self, async_session_with_engine):
        """Test soft deleting a batch job"""
        async_session, _ = async_session_with_engine
        service = BatchService(async_session)

        # Create job
        job_data = BatchJobCreate(
            provider="openai", model="gpt-4", mode="chat", request_count=10
        )
        job = await service.create_batch_job(job_data)

        # Delete job
        deleted = await service.delete_batch_job(job.id)
        assert deleted is True

        # Verify job is soft deleted
        retrieved_job = await service.get_batch_job(job.id)
        assert retrieved_job is None

    @pytest.mark.asyncio
    async def test_query_batch_jobs(self, async_session_with_engine):
        """Test querying batch jobs with filters"""
        async_session, _ = async_session_with_engine
        service = BatchService(async_session)

        # Create multiple jobs
        jobs_data = [
            BatchJobCreate(
                provider="openai", model="gpt-4", mode="chat", request_count=10
            ),
            BatchJobCreate(
                provider="anthropic",
                model="claude-3",
                mode="completion",
                request_count=5,
            ),
            BatchJobCreate(
                provider="openai", model="gpt-3.5", mode="chat", request_count=8
            ),
        ]

        created_jobs = []
        for job_data in jobs_data:
            job = await service.create_batch_job(job_data)
            created_jobs.append(job)

        # Query by provider
        query = BatchJobQuery(provider="openai", limit=10, offset=0)
        openai_jobs = await service.query_batch_jobs(query)
        assert len(openai_jobs) == 2
        assert all(job.provider == "openai" for job in openai_jobs)

        # Query by mode
        query = BatchJobQuery(mode="chat", limit=10, offset=0)
        chat_jobs = await service.query_batch_jobs(query)
        assert len(chat_jobs) == 2
        assert all(job.mode == "chat" for job in chat_jobs)

    @pytest.mark.asyncio
    async def test_get_pending_jobs(self, async_session_with_engine):
        """Test retrieving pending jobs"""
        async_session, _ = async_session_with_engine
        service = BatchService(async_session)

        # Create jobs with different statuses
        job1_data = BatchJobCreate(
            provider="openai", model="gpt-4", mode="chat", request_count=10
        )
        job1 = await service.create_batch_job(job1_data)

        job2_data = BatchJobCreate(
            provider="openai", model="gpt-4", mode="chat", request_count=5
        )
        job2 = await service.create_batch_job(job2_data)

        # Update one to completed
        await service.update_batch_job(
            job2.id, BatchJobUpdate(status=BatchJobStatus.COMPLETED)
        )

        # Get pending jobs
        pending_jobs = await service.get_pending_jobs()

        # Should only return job1 (CREATED status)
        assert len(pending_jobs) == 1
        assert pending_jobs[0].id == job1.id
        assert pending_jobs[0].status == BatchJobStatus.CREATED


@pytest.mark.integration
class TestBatchItemOperations:
    """Test batch item operations"""

    @pytest.mark.asyncio
    async def test_create_batch_items(self, async_session_with_engine):
        """Test creating batch items"""
        async_session, _ = async_session_with_engine
        service = BatchService(async_session)

        # Create parent job
        job_data = BatchJobCreate(
            provider="openai", model="gpt-4", mode="chat", request_count=0
        )
        job = await service.create_batch_job(job_data)

        # Create items
        items_data = [
            BatchItemCreate(
                sequence_index=0,
                custom_external_id="item-1",
                input_payload={"message": "Hello 1"},
            ),
            BatchItemCreate(
                sequence_index=1,
                custom_external_id="item-2",
                input_payload={"message": "Hello 2"},
            ),
        ]

        items = await service.create_batch_items(job.id, items_data)

        assert len(items) == 2
        assert items[0].sequence_index == 0
        assert items[0].custom_external_id == "item-1"
        assert items[0].input_payload == {"message": "Hello 1"}
        assert items[0].status == BatchItemStatus.QUEUED

        # Verify job request count was updated
        updated_job = await service.get_batch_job(job.id)
        assert updated_job.request_count == 2

    @pytest.mark.asyncio
    async def test_get_batch_items(self, async_session_with_engine):
        """Test retrieving batch items"""
        async_session, _ = async_session_with_engine
        service = BatchService(async_session)

        # Create job and items
        job_data = BatchJobCreate(
            provider="openai", model="gpt-4", mode="chat", request_count=0
        )
        job = await service.create_batch_job(job_data)

        items_data = [
            BatchItemCreate(sequence_index=0, input_payload={"message": "Test 1"}),
            BatchItemCreate(sequence_index=1, input_payload={"message": "Test 2"}),
        ]
        await service.create_batch_items(job.id, items_data)

        # Get all items
        items = await service.get_batch_items(job.id)
        assert len(items) == 2
        assert items[0].sequence_index == 0  # Should be ordered by sequence
        assert items[1].sequence_index == 1

        # Get items by status
        queued_items = await service.get_batch_items(
            job.id, status=BatchItemStatus.QUEUED
        )
        assert len(queued_items) == 2

    @pytest.mark.asyncio
    async def test_update_batch_item_and_job_counts(self, async_session_with_engine):
        """Test updating batch item and propagating to job counts"""
        async_session, _ = async_session_with_engine
        service = BatchService(async_session)

        # Create job and items
        job_data = BatchJobCreate(
            provider="openai", model="gpt-4", mode="chat", request_count=0
        )
        job = await service.create_batch_job(job_data)

        items_data = [
            BatchItemCreate(sequence_index=0, input_payload={"message": "Test 1"}),
            BatchItemCreate(sequence_index=1, input_payload={"message": "Test 2"}),
        ]
        items = await service.create_batch_items(job.id, items_data)

        # Update first item to completed
        item_update = BatchItemUpdate(
            status=BatchItemStatus.COMPLETED, output_payload={"response": "Success"}
        )
        updated_item = await service.update_batch_item(items[0].id, item_update)

        assert updated_item.status == BatchItemStatus.COMPLETED
        assert updated_item.output_payload == {"response": "Success"}

        # Check job counts were updated
        updated_job = await service.get_batch_job(job.id)
        assert updated_job.success_count == 1
        assert updated_job.error_count == 0

        # Update second item to failed
        item_update = BatchItemUpdate(
            status=BatchItemStatus.FAILED, error_detail="Processing failed"
        )
        await service.update_batch_item(items[1].id, item_update)

        # Check job counts again
        updated_job = await service.get_batch_job(job.id)
        assert updated_job.success_count == 1
        assert updated_job.error_count == 1
        assert updated_job.status == BatchJobStatus.COMPLETED  # All items processed


@pytest.mark.integration
class TestBatchEventOperations:
    """Test batch event operations"""

    @pytest.mark.asyncio
    async def test_create_batch_event(self, async_session_with_engine):
        """Test creating batch events"""
        async_session, _ = async_session_with_engine
        service = BatchService(async_session)

        # Create job
        job_data = BatchJobCreate(
            provider="openai", model="gpt-4", mode="chat", request_count=10
        )
        job = await service.create_batch_job(job_data)

        # Create event
        event_data = BatchEventCreate(
            event_type=BatchEventType.PROGRESS_UPDATE,
            event_data={"progress": 50, "message": "Halfway done"},
        )

        event = await service.create_batch_event(job.id, event_data)

        assert event.id is not None
        assert event.batch_job_id == job.id
        assert event.event_type == BatchEventType.PROGRESS_UPDATE
        assert event.event_data == {"progress": 50, "message": "Halfway done"}

    @pytest.mark.asyncio
    async def test_get_batch_events(self, async_session_with_engine):
        """Test retrieving batch events"""
        async_session, _ = async_session_with_engine
        service = BatchService(async_session)

        # Create job
        job_data = BatchJobCreate(
            provider="openai", model="gpt-4", mode="chat", request_count=10
        )
        job = await service.create_batch_job(job_data)

        # Create multiple events
        events_data = [
            BatchEventCreate(
                event_type=BatchEventType.PROGRESS_UPDATE, event_data={"progress": 25}
            ),
            BatchEventCreate(
                event_type=BatchEventType.ERROR, event_data={"error": "Network timeout"}
            ),
            BatchEventCreate(
                event_type=BatchEventType.PROGRESS_UPDATE, event_data={"progress": 75}
            ),
        ]

        for event_data in events_data:
            await service.create_batch_event(job.id, event_data)

        # Get all events (should include initial status change event from job creation)
        all_events = await service.get_batch_events(job.id)
        assert len(all_events) >= 4  # 3 created + 1 from job creation

        # Get progress events only
        progress_events = await service.get_batch_events(
            job.id, event_type=BatchEventType.PROGRESS_UPDATE
        )
        assert len(progress_events) == 2
        assert all(
            event.event_type == BatchEventType.PROGRESS_UPDATE
            for event in progress_events
        )


@pytest.mark.integration
class TestBatchServiceStatistics:
    """Test batch statistics functionality"""

    @pytest.mark.asyncio
    async def test_get_batch_stats(self, async_session_with_engine):
        """Test retrieving batch statistics"""
        async_session, _ = async_session_with_engine
        service = BatchService(async_session)

        # Create jobs with different providers and statuses
        jobs_data = [
            BatchJobCreate(
                provider="openai", model="gpt-4", mode="chat", request_count=10
            ),
            BatchJobCreate(
                provider="anthropic",
                model="claude-3",
                mode="completion",
                request_count=5,
            ),
            BatchJobCreate(
                provider="openai", model="gpt-3.5", mode="chat", request_count=8
            ),
        ]

        jobs = []
        for job_data in jobs_data:
            job = await service.create_batch_job(job_data)
            jobs.append(job)

        # Update some jobs with success/error counts
        await service.update_batch_job(
            jobs[0].id,
            BatchJobUpdate(
                success_count=8, error_count=2, status=BatchJobStatus.COMPLETED
            ),
        )
        await service.update_batch_job(
            jobs[1].id,
            BatchJobUpdate(
                success_count=5, error_count=0, status=BatchJobStatus.COMPLETED
            ),
        )

        # Get statistics
        stats = await service.get_batch_stats()

        assert stats.total_jobs == 3
        assert stats.jobs_by_provider["openai"] == 2
        assert stats.jobs_by_provider["anthropic"] == 1
        assert stats.total_requests == 23  # 10 + 5 + 8
        assert stats.total_successes == 13  # 8 + 5
        assert stats.total_errors == 2

        # Success rate should be 13/15 * 100 = 86.67%
        assert abs(stats.average_success_rate - 86.66666666666667) < 0.001


@pytest.mark.integration
class TestBatchServiceWithItems:
    """Test batch service operations with items"""

    @pytest.mark.asyncio
    async def test_get_batch_job_with_items(self, async_session_with_engine):
        """Test retrieving batch job with all its items"""
        async_session, _ = async_session_with_engine
        service = BatchService(async_session)

        # Create job
        job_data = BatchJobCreate(
            provider="openai", model="gpt-4", mode="chat", request_count=0
        )
        job = await service.create_batch_job(job_data)

        # Create items
        items_data = [
            BatchItemCreate(sequence_index=0, input_payload={"message": "Test 1"}),
            BatchItemCreate(sequence_index=1, input_payload={"message": "Test 2"}),
        ]
        await service.create_batch_items(job.id, items_data)

        # Get job with items
        job_with_items = await service.get_batch_job_with_items(job.id)

        assert job_with_items is not None
        assert len(job_with_items.items) == 2
        # Items should be accessible via the relationship
        items_list = list(job_with_items.items)
        assert len(items_list) == 2

    @pytest.mark.asyncio
    async def test_bulk_update_items_status(self, async_session_with_engine):
        """Test bulk updating item statuses"""
        async_session, _ = async_session_with_engine
        service = BatchService(async_session)

        # Create job and items
        job_data = BatchJobCreate(
            provider="openai", model="gpt-4", mode="chat", request_count=0
        )
        job = await service.create_batch_job(job_data)

        items_data = [
            BatchItemCreate(sequence_index=0, input_payload={"message": "Test 1"}),
            BatchItemCreate(sequence_index=1, input_payload={"message": "Test 2"}),
            BatchItemCreate(sequence_index=2, input_payload={"message": "Test 3"}),
        ]
        items = await service.create_batch_items(job.id, items_data)

        # Bulk update statuses
        status_updates = [
            {
                "item_id": items[0].id,
                "status": BatchItemStatus.COMPLETED,
                "output_payload": {"response": "Success 1"},
            },
            {
                "item_id": items[1].id,
                "status": BatchItemStatus.FAILED,
                "error_detail": "Processing failed",
            },
            {
                "item_id": items[2].id,
                "status": BatchItemStatus.COMPLETED,
                "output_payload": {"response": "Success 3"},
            },
        ]

        updated_count = await service.update_batch_items_status(status_updates)
        assert updated_count == 3

        # Verify job counts
        updated_job = await service.get_batch_job(job.id)
        assert updated_job.success_count == 2
        assert updated_job.error_count == 1
        assert updated_job.status == BatchJobStatus.COMPLETED

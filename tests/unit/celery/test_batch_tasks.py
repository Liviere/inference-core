"""
Tests for batch lifecycle Celery tasks
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from celery.exceptions import Retry

from inference_core.celery.tasks.batch_tasks import (
    batch_fetch,
    batch_poll,
    batch_retry_failed,
    batch_submit,
)
from inference_core.database.sql.models.batch import BatchItemStatus, BatchJobStatus


class TestBatchSubmitTask:
    """Tests for batch_submit task"""

    @patch("inference_core.celery.tasks.batch_tasks.get_async_session")
    @patch("inference_core.celery.tasks.batch_tasks.registry")
    def test_batch_submit_success(self, mock_registry, mock_get_session):
        """Test successful batch submission"""
        # Setup mocks
        job_id = str(uuid.uuid4())
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        mock_batch_service = AsyncMock()
        with patch(
            "inference_core.celery.tasks.batch_tasks.BatchService",
            return_value=mock_batch_service,
        ):
            mock_job = MagicMock()
            mock_job.id = uuid.UUID(job_id)
            mock_job.status = BatchJobStatus.CREATED
            mock_job.provider = "openai"
            mock_job.model = "gpt-4"
            mock_job.mode = "chat"
            mock_job.config_json = {}

            mock_batch_service.get_batch_job.return_value = mock_job
            mock_batch_service.get_batch_items.return_value = [
                MagicMock(
                    id=uuid.uuid4(), sequence_index=0, input_payload={"test": "data"}
                )
            ]

            mock_provider = MagicMock()
            mock_provider.prepare_payloads.return_value = MagicMock()
            mock_provider.submit.return_value = MagicMock(
                provider_batch_id="batch_123", submitted_at=datetime.now(), item_count=1
            )
            mock_registry.create_provider.return_value = mock_provider

            # Execute task
            result = batch_submit(job_id)

            # Verify result
            assert result["job_id"] == job_id
            assert result["status"] == "submitted"
            assert "duration" in result


class TestBatchPollTask:
    """Tests for batch_poll task"""

    @patch("inference_core.celery.tasks.batch_tasks.get_sync_redis")
    @patch("inference_core.celery.tasks.batch_tasks.get_async_session")
    @patch("inference_core.celery.tasks.batch_tasks.select")
    def test_batch_poll_with_lock(self, mock_select, mock_get_session, mock_get_redis):
        """Test batch poll with Redis lock"""
        # Setup Redis mock
        mock_redis = MagicMock()
        mock_redis.set.return_value = True  # Lock acquired
        mock_get_redis.return_value = mock_redis

        # Setup session mock
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        # Setup query mock - need to properly mock the SQLAlchemy result
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars

        # Make session.execute return the mocked result directly (not a coroutine)
        async def mock_execute(query):
            return mock_result

        mock_session.execute = mock_execute

        # Execute task
        result = batch_poll()

        # Verify lock was acquired and released
        mock_redis.set.assert_called_once()
        mock_redis.delete.assert_called_once()
        assert result["status"] == "completed"
        assert result["jobs_polled"] == 0

    @patch("inference_core.celery.tasks.batch_tasks.get_sync_redis")
    def test_batch_poll_lock_contention(self, mock_get_redis):
        """Test batch poll when lock is already held"""
        # Setup Redis mock - lock not acquired
        mock_redis = MagicMock()
        mock_redis.set.return_value = False  # Lock not acquired
        mock_get_redis.return_value = mock_redis

        # Execute task
        result = batch_poll()

        # Verify task was skipped
        assert result["status"] == "skipped"
        assert "Another poll process is running" in result["reason"]


class TestBatchFetchTask:
    """Tests for batch_fetch task"""

    @patch("inference_core.celery.tasks.batch_tasks.get_async_session")
    @patch("inference_core.celery.tasks.batch_tasks.registry")
    def test_batch_fetch_success(self, mock_registry, mock_get_session):
        """Test successful batch fetch"""
        # Setup mocks
        job_id = str(uuid.uuid4())
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        mock_batch_service = AsyncMock()
        with patch(
            "inference_core.celery.tasks.batch_tasks.BatchService",
            return_value=mock_batch_service,
        ):
            mock_job = MagicMock()
            mock_job.id = uuid.UUID(job_id)
            mock_job.status = BatchJobStatus.COMPLETED
            mock_job.provider = "openai"

            mock_items = [MagicMock(id=uuid.uuid4(), status=BatchItemStatus.QUEUED)]

            mock_batch_service.get_batch_job.return_value = mock_job
            mock_batch_service.get_batch_items.return_value = mock_items

            mock_provider = MagicMock()
            mock_provider.fetch_results.return_value = [
                MagicMock(
                    custom_id=str(mock_items[0].id),
                    is_success=True,
                    output_data={"result": "success"},
                    output_text="Success text",
                )
            ]
            mock_registry.create_provider.return_value = mock_provider

            # Execute task
            result = batch_fetch(job_id)

            # Verify result
            assert result["job_id"] == job_id
            assert result["status"] == "completed"
            assert result["success_count"] == 1
            assert result["error_count"] == 0

    @patch("inference_core.celery.tasks.batch_tasks.get_async_session")
    def test_batch_fetch_not_completed(self, mock_get_session):
        """Test batch fetch when job is not completed"""
        job_id = str(uuid.uuid4())
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        mock_batch_service = AsyncMock()
        with patch(
            "inference_core.celery.tasks.batch_tasks.BatchService",
            return_value=mock_batch_service,
        ):
            mock_job = MagicMock()
            mock_job.status = BatchJobStatus.IN_PROGRESS  # Not completed
            mock_batch_service.get_batch_job.return_value = mock_job

            # Execute task
            result = batch_fetch(job_id)

            # Verify result
            assert result["job_id"] == job_id
            assert result["status"] == "in_progress"
            assert "Job not completed" in result["message"]


class TestBatchRetryFailedTask:
    """Tests for batch_retry_failed task"""

    @patch("inference_core.celery.tasks.batch_tasks.get_async_session")
    @patch("inference_core.celery.tasks.batch_tasks.batch_submit")
    def test_batch_retry_failed_success(self, mock_batch_submit, mock_get_session):
        """Test successful retry of failed items"""
        job_id = str(uuid.uuid4())
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        mock_batch_service = AsyncMock()
        with patch(
            "inference_core.celery.tasks.batch_tasks.BatchService",
            return_value=mock_batch_service,
        ):
            mock_original_job = MagicMock()
            mock_original_job.provider = "openai"
            mock_original_job.model = "gpt-4"
            mock_original_job.mode = "chat"
            mock_original_job.config_json = {}

            mock_failed_items = [
                MagicMock(
                    status=BatchItemStatus.FAILED,
                    input_payload={"test": "data"},
                    custom_external_id="item1",
                )
            ]

            mock_retry_job = MagicMock()
            mock_retry_job.id = uuid.uuid4()

            mock_batch_service.get_batch_job.return_value = mock_original_job
            mock_batch_service.get_batch_items.return_value = mock_failed_items
            mock_batch_service.create_batch_job.return_value = mock_retry_job

            # Execute task
            result = batch_retry_failed(job_id)

            # Verify result
            assert result["original_job_id"] == job_id
            assert result["status"] == "retry_job_created"
            assert result["failed_items_count"] == 1
            mock_batch_submit.delay.assert_called_once()

    @patch("inference_core.celery.tasks.batch_tasks.get_async_session")
    def test_batch_retry_failed_no_failed_items(self, mock_get_session):
        """Test retry when there are no failed items"""
        job_id = str(uuid.uuid4())
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        mock_batch_service = AsyncMock()
        with patch(
            "inference_core.celery.tasks.batch_tasks.BatchService",
            return_value=mock_batch_service,
        ):
            mock_original_job = MagicMock()
            mock_batch_service.get_batch_job.return_value = mock_original_job
            mock_batch_service.get_batch_items.return_value = []  # No failed items

            # Execute task
            result = batch_retry_failed(job_id)

            # Verify result
            assert result["job_id"] == job_id
            assert result["status"] == "no_failed_items"
            assert "No failed items to retry" in result["message"]


class TestBatchTasksIntegration:
    """Integration tests for batch tasks"""

    def test_all_tasks_are_registered(self):
        """Test that all batch tasks are properly registered with Celery"""
        from inference_core.celery.celery_main import celery_app

        registered_tasks = celery_app.tasks.keys()

        # Check that our batch tasks are registered
        assert "batch.submit" in registered_tasks
        assert "batch.poll" in registered_tasks
        assert "batch.fetch" in registered_tasks
        assert "batch.retry_failed" in registered_tasks

    def test_task_retry_configuration(self):
        """Test that tasks have proper retry configuration"""
        # Check retry configuration for batch_submit
        assert batch_submit.max_retries == 5
        assert batch_submit.autoretry_for is not None
        assert batch_submit.retry_backoff is True
        assert batch_submit.retry_backoff_max == 300
        assert batch_submit.retry_jitter is True

        # Check retry configuration for batch_poll
        assert batch_poll.max_retries == 3
        assert batch_poll.autoretry_for is not None
        assert batch_poll.retry_backoff is True
        assert batch_poll.retry_backoff_max == 120

        # Check retry configuration for batch_fetch
        assert batch_fetch.max_retries == 5
        assert batch_fetch.retry_backoff is True
        assert batch_fetch.retry_backoff_max == 300

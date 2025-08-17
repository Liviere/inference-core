"""
Celery tasks for batch processing lifecycle

Orchestrates batch job submission, polling, fetching results, and retry operations.
Includes concurrency controls and error handling with exponential backoff.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import select

from app.celery.celery_main import celery_app
from app.core.redis_client import get_sync_redis
from app.database.sql.connection import get_async_session
from app.database.sql.models.batch import BatchItemStatus, BatchJob, BatchJobStatus
from app.llm.batch.exceptions import ProviderPermanentError, ProviderTransientError
from app.llm.batch.registry import registry
from app.llm.config import llm_config
from app.schemas.batch import (
    BatchItemCreate,
    BatchItemUpdate,
    BatchJobCreate,
    BatchJobUpdate,
)
from app.services.batch_service import BatchService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async event loop management
# ---------------------------------------------------------------------------
# Problem: Using asyncio.run() inside Celery (prefork) tasks creates a brand new
# event loop for every invocation and closes it afterwards. SQLAlchemy async
# engine / asyncpg connections become bound to the original loop they were
# created on. Reusing pooled connections across new loops then triggers errors:
#   "Future <...> attached to a different loop" or "Event loop is closed".
# Solution: Maintain a persistent event loop per worker process and run all
# coroutines via that loop (run_until_complete). Each Celery worker process
# executes tasks sequentially, so this is safe. If concurrency changes to allow
# parallel execution in the same process, a more robust queue/loop thread would
# be needed.

_loop: asyncio.AbstractEventLoop | None = None


def _run_async(coro):
    """Run an async coroutine on a persistent event loop.

    Ensures a single event loop per process preventing cross-loop attachment
    issues with asyncpg / SQLAlchemy.
    """
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop.run_until_complete(coro)


# Redis lock keys
BATCH_POLL_LOCK_KEY = "batch_poll:lock"
BATCH_POLL_LOCK_TIMEOUT = 300  # 5 minutes

# Dispatch lock to avoid multiple workers dispatching simultaneously (short TTL)
BATCH_DISPATCH_LOCK_KEY = "batch_dispatch:lock"
BATCH_DISPATCH_LOCK_TIMEOUT = 60


@celery_app.task(
    bind=True,
    name="batch.dispatch",
)
def batch_dispatch(self) -> Dict[str, Any]:
    """Dispatch (submit) all jobs in CREATED status.

    Runs periodically via beat; finds jobs still in CREATED and enqueues a submit task
    per job. Lightweight – real submission logic stays in batch_submit (idempotent).
    """
    start_time = time.time()
    redis_client = get_sync_redis()
    lock_acquired = redis_client.set(
        BATCH_DISPATCH_LOCK_KEY, "1", nx=True, ex=BATCH_DISPATCH_LOCK_TIMEOUT
    )
    if not lock_acquired:
        logger.debug("Dispatch already running, skipping")
        return {
            "status": "skipped",
            "reason": "lock",
            "duration": time.time() - start_time,
        }

    dispatched = 0
    errors = 0
    try:

        async def _dispatch():
            nonlocal dispatched, errors
            async with get_async_session() as session:
                batch_service = BatchService(session)
                # Select jobs in CREATED
                result = await session.execute(
                    select(BatchJob).where(BatchJob.status == BatchJobStatus.CREATED)
                )
                jobs = result.scalars().all()
                for job in jobs:
                    try:
                        batch_submit.delay(str(job.id))
                        dispatched += 1
                    except Exception as e:
                        logger.error(f"Failed to enqueue submit for job {job.id}: {e}")
                        errors += 1

        _run_async(_dispatch())
    finally:
        redis_client.delete(BATCH_DISPATCH_LOCK_KEY)

    return {
        "status": "completed",
        "jobs_dispatched": dispatched,
        "errors": errors,
        "duration": time.time() - start_time,
    }


@celery_app.task(
    bind=True,
    name="batch.submit",
    autoretry_for=(ProviderTransientError,),
    max_retries=5,
    retry_backoff=True,
    retry_backoff_max=300,  # 5 minutes max
    retry_jitter=True,
)
def batch_submit(self, job_id: str) -> Dict[str, Any]:
    """
    Submit a batch job to the provider.

    Args:
        job_id: UUID of the batch job to submit

    Returns:
        Dict containing submission results and metadata
    """
    start_time = time.time()
    job_uuid = UUID(job_id)

    try:
        logger.info(f"Starting batch submission for job {job_id}")

        # Get database session and batch service
        async def _submit():
            async with get_async_session() as session:
                batch_service = BatchService(session)

                # Get batch job with items
                job = await batch_service.get_batch_job(job_uuid)
                if not job:
                    raise ValueError(f"Batch job {job_id} not found")

                # Check if already submitted
                if job.status != BatchJobStatus.CREATED:
                    logger.warning(
                        f"Job {job_id} already submitted (status: {job.status})"
                    )
                    return {
                        "job_id": job_id,
                        "status": job.status.value,
                        "message": "Job already submitted",
                        "duration": time.time() - start_time,
                    }

                # Get items for submission
                items = await batch_service.get_batch_items(job_uuid)
                if not items:
                    raise ValueError(f"No items found for batch job {job_id}")

                # Build provider runtime config (merge general provider + batch model info)
                provider_instance = registry.create_provider(
                    job.provider,
                    config=llm_config.get_provider_runtime_config(job.provider),
                )

                # Prepare submission
                item_data = [
                    {
                        # Provider prepare_payloads expects an 'id' field
                        "id": str(item.id),
                        "sequence_index": item.sequence_index,
                        "input_payload": item.input_payload,
                    }
                    for item in items
                ]

                prepared_submission = provider_instance.prepare_payloads(
                    batch_items=item_data,
                    model=job.model,
                    mode=job.mode,
                    config=job.config_json,
                )

                # Submit to provider
                submit_result = provider_instance.submit(prepared_submission)

                # Update job with provider batch ID and status
                await batch_service.update_batch_job(
                    job_uuid,
                    BatchJobUpdate(
                        status=BatchJobStatus.SUBMITTED,
                        provider_batch_id=submit_result.provider_batch_id,
                        submitted_at=submit_result.submitted_at,
                    ),
                )

                logger.info(
                    f"Successfully submitted batch job {job_id} to provider {job.provider}"
                )

                return {
                    "job_id": job_id,
                    "provider_batch_id": submit_result.provider_batch_id,
                    "status": "submitted",
                    "item_count": submit_result.item_count,
                    "submitted_at": submit_result.submitted_at.isoformat(),
                    "duration": time.time() - start_time,
                }

        return _run_async(_submit())

    except ProviderPermanentError as e:
        logger.error(f"Permanent error submitting batch job {job_id}: {e}")

        # Update job status to failed
        async def _mark_failed():
            async with get_async_session() as session:
                batch_service = BatchService(session)
                await batch_service.update_batch_job(
                    job_uuid,
                    BatchJobUpdate(status=BatchJobStatus.FAILED, error_summary=str(e)),
                )

        _run_async(_mark_failed())

        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "duration": time.time() - start_time,
        }

    except ProviderTransientError as e:
        logger.warning(f"Transient error submitting batch job {job_id}: {e}")
        raise self.retry(countdown=min(2**self.request.retries, 300), exc=e)

    except Exception as e:
        logger.error(f"Unexpected error submitting batch job {job_id}: {e}")
        return {
            "job_id": job_id,
            "status": "error",
            "error": str(e),
            "duration": time.time() - start_time,
        }


@celery_app.task(
    bind=True,
    name="batch.poll",
    autoretry_for=(ProviderTransientError,),
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
)
def batch_poll(self) -> Dict[str, Any]:
    """
    Poll status of all in-progress batch jobs.

    Uses Redis lock to prevent concurrent polling.

    Returns:
        Dict containing polling results and statistics
    """
    start_time = time.time()
    redis_client = get_sync_redis()

    # Try to acquire lock
    lock_acquired = redis_client.set(
        BATCH_POLL_LOCK_KEY, "9", nx=True, ex=BATCH_POLL_LOCK_TIMEOUT
    )

    if not lock_acquired:
        logger.info("Batch poll already running, skipping")
        return {
            "status": "skipped",
            "reason": "Another poll process is running",
            "duration": time.time() - start_time,
        }

    try:
        logger.info("Starting batch poll")

        async def _poll():
            async with get_async_session() as session:
                batch_service = BatchService(session)

                # Get jobs that need polling
                query = select(BatchJob).where(
                    BatchJob.status.in_(
                        [BatchJobStatus.SUBMITTED, BatchJobStatus.IN_PROGRESS]
                    )
                )
                result = await session.execute(query)
                jobs = result.scalars().all()

                if not jobs:
                    logger.info("No jobs to poll")
                    return {
                        "status": "completed",
                        "jobs_polled": 0,
                        "status_changes": 0,
                        "duration": time.time() - start_time,
                    }

                status_changes = 0
                jobs_polled = 0

                for job in jobs:
                    try:
                        # Get provider instance
                        provider_instance = registry.create_provider(
                            job.provider,
                            config=llm_config.get_provider_runtime_config(job.provider),
                        )

                        # Poll provider status
                        provider_status = provider_instance.poll_status(
                            job.provider_batch_id
                        )
                        jobs_polled += 1

                        # Check if status changed
                        new_status = BatchJobStatus(provider_status.normalized_status)
                        if new_status != job.status:
                            status_changes += 1

                            # Update job status
                            update_data = {"status": new_status}

                            # Handle completion
                            if new_status in [
                                BatchJobStatus.COMPLETED,
                                BatchJobStatus.FAILED,
                            ]:
                                update_data["completed_at"] = datetime.now(timezone.utc)

                                # For completed jobs, schedule fetch task
                                if new_status == BatchJobStatus.COMPLETED:
                                    batch_fetch.delay(str(job.id))

                            await batch_service.update_batch_job(
                                job.id, BatchJobUpdate(**update_data)
                            )

                            logger.info(
                                f"Job {job.id} status changed: {job.status} -> {new_status}"
                            )

                    except ProviderPermanentError as e:
                        logger.error(f"Permanent error polling job {job.id}: {e}")
                        # Mark job as failed
                        await batch_service.update_batch_job(
                            job.id,
                            BatchJobUpdate(
                                status=BatchJobStatus.FAILED,
                                error_summary=str(e),
                                completed_at=datetime.now(timezone.utc),
                            ),
                        )
                        status_changes += 1

                    except ProviderTransientError as e:
                        logger.warning(f"Transient error polling job {job.id}: {e}")
                        # Continue with other jobs, don't fail the whole poll
                        continue

                    except Exception as e:
                        logger.error(f"Unexpected error polling job {job.id}: {e}")
                        continue

                logger.info(
                    f"Batch poll completed: {jobs_polled} jobs polled, {status_changes} status changes"
                )

                return {
                    "status": "completed",
                    "jobs_polled": jobs_polled,
                    "status_changes": status_changes,
                    "duration": time.time() - start_time,
                }

        return _run_async(_poll())

    except Exception as e:
        logger.error(f"Error in batch poll: {e}")
        return {
            "status": "error",
            "error": str(e),
            "duration": time.time() - start_time,
        }

    finally:
        # Release lock
        redis_client.delete(BATCH_POLL_LOCK_KEY)


@celery_app.task(
    bind=True,
    name="batch.fetch",
    autoretry_for=(ProviderTransientError,),
    max_retries=5,
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)
def batch_fetch(self, job_id: str) -> Dict[str, Any]:
    """
    Fetch results from a completed batch job.

    Args:
        job_id: UUID of the batch job to fetch results for

    Returns:
        Dict containing fetch results and statistics
    """
    start_time = time.time()
    job_uuid = UUID(job_id)

    try:
        logger.info(f"Starting batch fetch for job {job_id}")

        async def _fetch():
            async with get_async_session() as session:
                batch_service = BatchService(session)

                # Get batch job
                job = await batch_service.get_batch_job(job_uuid)
                if not job:
                    raise ValueError(f"Batch job {job_id} not found")

                # Check if job is completed
                if job.status != BatchJobStatus.COMPLETED:
                    logger.warning(f"Job {job_id} not completed (status: {job.status})")
                    return {
                        "job_id": job_id,
                        "status": job.status.value,
                        "message": "Job not completed",
                        "duration": time.time() - start_time,
                    }

                # Check if already fetched (idempotent)
                items = await batch_service.get_batch_items(job_uuid)
                completed_items = [
                    item
                    for item in items
                    if item.status
                    in [BatchItemStatus.COMPLETED, BatchItemStatus.FAILED]
                ]

                if len(completed_items) == len(items):
                    logger.info(f"Job {job_id} already has all results fetched")
                    return {
                        "job_id": job_id,
                        "status": "already_fetched",
                        "message": "All results already fetched",
                        "duration": time.time() - start_time,
                    }

                # Get provider instance
                provider_instance = registry.create_provider(
                    job.provider,
                    config=llm_config.get_provider_runtime_config(job.provider),
                )

                # Fetch results from provider
                results = provider_instance.fetch_results(job.provider_batch_id)

                # Map results to items by custom_id
                results_map = {result.custom_id: result for result in results}

                success_count = 0
                error_count = 0

                # Update items with results
                for item in items:
                    item_id_str = str(item.id)
                    if item_id_str in results_map:
                        result = results_map[item_id_str]

                        if result.is_success:
                            await batch_service.update_batch_item(
                                item.id,
                                BatchItemUpdate(
                                    status=BatchItemStatus.COMPLETED,
                                    output_payload=result.output_data
                                    or {"text": result.output_text},
                                ),
                            )
                            success_count += 1
                        else:
                            await batch_service.update_batch_item(
                                item.id,
                                BatchItemUpdate(
                                    status=BatchItemStatus.FAILED,
                                    error_detail=result.error_message,
                                ),
                            )
                            error_count += 1

                # Update job counts
                await batch_service.update_batch_job(
                    job_uuid,
                    BatchJobUpdate(
                        success_count=success_count, error_count=error_count
                    ),
                )

                logger.info(
                    f"Fetched results for job {job_id}: {success_count} successful, {error_count} failed"
                )

                return {
                    "job_id": job_id,
                    "status": "completed",
                    "success_count": success_count,
                    "error_count": error_count,
                    "total_items": len(items),
                    "duration": time.time() - start_time,
                }

        return _run_async(_fetch())

    except ProviderPermanentError as e:
        logger.error(f"Permanent error fetching batch job {job_id}: {e}")
        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "duration": time.time() - start_time,
        }

    except ProviderTransientError as e:
        logger.warning(f"Transient error fetching batch job {job_id}: {e}")
        raise self.retry(countdown=min(2**self.request.retries, 300), exc=e)

    except Exception as e:
        logger.error(f"Unexpected error fetching batch job {job_id}: {e}")
        return {
            "job_id": job_id,
            "status": "error",
            "error": str(e),
            "duration": time.time() - start_time,
        }


@celery_app.task(
    bind=True,
    name="batch.retry_failed",
    autoretry_for=(ProviderTransientError,),
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
)
def batch_retry_failed(self, job_id: str) -> Dict[str, Any]:
    """
    Retry failed items from a batch job by creating a new job.

    Args:
        job_id: UUID of the batch job to retry failed items from

    Returns:
        Dict containing retry results and new job information
    """
    start_time = time.time()
    job_uuid = UUID(job_id)

    try:
        logger.info(f"Starting batch retry for failed items in job {job_id}")

        async def _retry():
            async with get_async_session() as session:
                batch_service = BatchService(session)

                # Get original batch job
                original_job = await batch_service.get_batch_job(job_uuid)
                if not original_job:
                    raise ValueError(f"Batch job {job_id} not found")

                # Get failed items
                items = await batch_service.get_batch_items(job_uuid)
                failed_items = [
                    item for item in items if item.status == BatchItemStatus.FAILED
                ]

                if not failed_items:
                    logger.info(f"No failed items found in job {job_id}")
                    return {
                        "job_id": job_id,
                        "status": "no_failed_items",
                        "message": "No failed items to retry",
                        "duration": time.time() - start_time,
                    }

                # Create new batch job for retry
                retry_job_data = BatchJobCreate(
                    provider=original_job.provider,
                    model=original_job.model,
                    mode=original_job.mode,
                    request_count=len(failed_items),
                    config_json=original_job.config_json,
                )

                retry_job = await batch_service.create_batch_job(retry_job_data)

                # Create items for retry job
                retry_items_data = []
                for i, failed_item in enumerate(failed_items):
                    retry_items_data.append(
                        BatchItemCreate(
                            sequence_index=i,
                            custom_external_id=failed_item.custom_external_id,
                            input_payload=failed_item.input_payload,
                        )
                    )

                await batch_service.create_batch_items(retry_job.id, retry_items_data)

                # Submit retry job
                batch_submit.delay(str(retry_job.id))

                logger.info(
                    f"Created retry job {retry_job.id} for {len(failed_items)} failed items from {job_id}"
                )

                return {
                    "original_job_id": job_id,
                    "retry_job_id": str(retry_job.id),
                    "status": "retry_job_created",
                    "failed_items_count": len(failed_items),
                    "duration": time.time() - start_time,
                }

        return _run_async(_retry())

    except Exception as e:
        logger.error(f"Error retrying failed items from job {job_id}: {e}")
        return {
            "job_id": job_id,
            "status": "error",
            "error": str(e),
            "duration": time.time() - start_time,
        }

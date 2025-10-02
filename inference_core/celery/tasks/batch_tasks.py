"""
Celery tasks for batch processing lifecycle

Orchestrates batch job submission, polling, fetching results, and retry operations.
Includes concurrency controls and error handling with exponential backoff.
Enhanced with observability metrics, structured logging, and optional Sentry integration.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import select

from inference_core.celery.celery_main import celery_app
from inference_core.core.redis_client import get_sync_redis
from inference_core.database.sql.connection import get_async_session
from inference_core.database.sql.models.batch import (
    BatchItemStatus,
    BatchJob,
    BatchJobStatus,
)
from inference_core.llm.batch.exceptions import (
    ProviderPermanentError,
    ProviderTransientError,
)
from inference_core.llm.batch.registry import registry
from inference_core.llm.config import llm_config
from inference_core.observability.logging import get_batch_logger
from inference_core.observability.metrics import (
    record_error,
    record_item_status_change,
    record_job_duration,
    record_job_status_change,
    record_poll_cycle_duration,
    record_provider_latency,
    record_retry_attempt,
    time_provider_operation,
    update_jobs_in_progress,
)
from inference_core.observability.sentry import get_batch_sentry
from inference_core.schemas.batch import (
    BatchItemCreate,
    BatchItemUpdate,
    BatchJobCreate,
    BatchJobUpdate,
)
from inference_core.services.batch_service import BatchService

logger = logging.getLogger(__name__)
batch_logger = get_batch_logger()
batch_sentry = get_batch_sentry()


# Async execution helper (shared worker event loop)
from inference_core.celery.async_utils import run_in_worker_loop

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

    # Log operation start
    batch_logger.info(
        "Starting batch dispatch cycle",
        operation="dispatch",
        dispatch_cycle_start=start_time,
    )
    batch_sentry.log_operation_start("dispatch")

    redis_client = get_sync_redis()
    lock_acquired = redis_client.set(
        BATCH_DISPATCH_LOCK_KEY, "1", nx=True, ex=BATCH_DISPATCH_LOCK_TIMEOUT
    )
    if not lock_acquired:
        batch_logger.debug("Dispatch already running, skipping", operation="dispatch")
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

                batch_logger.info(
                    f"Found {len(jobs)} jobs to dispatch",
                    operation="dispatch",
                    jobs_found=len(jobs),
                )

                for job in jobs:
                    try:
                        # Set Sentry context for this job
                        batch_sentry.set_batch_context(
                            job_id=job.id, provider=job.provider, operation="dispatch"
                        )

                        batch_submit.delay(str(job.id))
                        dispatched += 1

                        batch_logger.debug(
                            f"Dispatched submit task for job {job.id}",
                            job_id=job.id,
                            provider=job.provider,
                            operation="dispatch",
                        )

                    except Exception as e:
                        errors += 1
                        batch_logger.error(
                            f"Failed to enqueue submit for job {job.id}: {e}",
                            job_id=job.id,
                            provider=job.provider if hasattr(job, "provider") else None,
                            operation="dispatch",
                            error_details={
                                "error": str(e),
                                "error_type": type(e).__name__,
                            },
                        )

                        # Capture error in Sentry
                        batch_sentry.capture_batch_error(
                            e,
                            job_id=job.id,
                            provider=job.provider if hasattr(job, "provider") else None,
                            operation="dispatch",
                        )

        # Execute async dispatch logic in worker loop
        run_in_worker_loop(_dispatch())
    finally:
        redis_client.delete(BATCH_DISPATCH_LOCK_KEY)

    duration = time.time() - start_time

    # Log completion with metrics
    batch_logger.info(
        f"Batch dispatch completed: {dispatched} dispatched, {errors} errors",
        operation="dispatch",
        jobs_dispatched=dispatched,
        errors=errors,
        duration_seconds=duration,
    )

    batch_sentry.log_operation_complete(
        "dispatch",
        success=(errors == 0),
        duration_seconds=duration,
        additional_data={"jobs_dispatched": dispatched, "errors": errors},
    )

    return {
        "status": "completed",
        "jobs_dispatched": dispatched,
        "errors": errors,
        "duration": duration,
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
    provider = None  # Will be set once we load the job

    try:
        batch_logger.info(
            f"Starting batch submission for job {job_id}",
            job_id=job_id,
            operation="submit",
        )
        batch_sentry.log_operation_start("submit", job_id=job_id)

        # Get database session and batch service
        async def _submit():
            nonlocal provider
            async with get_async_session() as session:
                batch_service = BatchService(session)

                # Get batch job with items
                job = await batch_service.get_batch_job(job_uuid)
                if not job:
                    raise ValueError(f"Batch job {job_id} not found")

                provider = job.provider

                # Set Sentry context with provider info
                batch_sentry.set_batch_context(
                    job_id=job.id, provider=provider, operation="submit"
                )

                # Check if already submitted
                if job.status != BatchJobStatus.CREATED:
                    batch_logger.warning(
                        f"Job {job_id} already submitted (status: {job.status})",
                        job_id=job_id,
                        provider=provider,
                        operation="submit",
                        status_transition=f"{job.status}->skipped",
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

                batch_logger.debug(
                    f"Preparing submission for {len(items)} items",
                    job_id=job_id,
                    provider=provider,
                    operation="submit",
                    item_count=len(items),
                )

                # Build provider runtime config (merge general provider + batch model info)
                provider_start_time = time.time()
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

                # Submit to provider using decorator for automatic timing
                @time_provider_operation(provider, "submit")
                def submit_batch():
                    return provider_instance.submit(prepared_submission)

                submit_result = submit_batch()

                batch_logger.info(
                    f"Provider submission completed for job {job_id}",
                    job_id=job_id,
                    provider=provider,
                    operation="submit",
                    provider_batch_id=submit_result.provider_batch_id,
                    item_count=submit_result.item_count,
                )

                # Update job with provider batch ID and status
                # Persist submission metadata (e.g. custom_id_mapping) into job metadata_json
                try:
                    metadata = (
                        job.get_metadata() if hasattr(job, "get_metadata") else {}
                    )
                except Exception:
                    metadata = {}
                if submit_result.submission_metadata:
                    # Namespace to avoid clashes with other providers/keys
                    metadata.setdefault("submission_metadata", {})
                    metadata["submission_metadata"][
                        str(job.id)
                    ] = submit_result.submission_metadata
                    if hasattr(job, "set_metadata"):
                        job.set_metadata(metadata)

                await batch_service.update_batch_job(
                    job_uuid,
                    BatchJobUpdate(
                        status=BatchJobStatus.SUBMITTED,
                        provider_batch_id=submit_result.provider_batch_id,
                        submitted_at=submit_result.submitted_at,
                    ),
                )

                # Record metrics for status change
                record_job_status_change(provider, "submitted")
                update_jobs_in_progress(provider, 1)  # Job is now in progress

                # Log structured event
                batch_logger.log_job_lifecycle_event(
                    "submitted",
                    job.id,
                    provider,
                    old_status="created",
                    new_status="submitted",
                    provider_batch_id=submit_result.provider_batch_id,
                    item_counts={"item_count": submit_result.item_count},
                )

                # Sentry breadcrumb for status change
                batch_sentry.log_status_change(
                    job.id,
                    provider,
                    "created",
                    "submitted",
                    provider_batch_id=submit_result.provider_batch_id,
                    additional_data={"item_count": submit_result.item_count},
                )

                # Create semantic event for submission (separate from status change)
                try:
                    from inference_core.database.sql.models.batch import BatchEventType

                    await batch_service.create_semantic_event(
                        job_uuid,
                        BatchEventType.SUBMITTED,
                        event_data={
                            "provider_batch_id": submit_result.provider_batch_id,
                            "item_count": submit_result.item_count,
                        },
                    )
                except Exception as evt_err:  # pragma: no cover - non critical
                    batch_logger.warning(
                        f"Failed to create SUBMITTED event for job {job_id}: {evt_err}",
                        job_id=job_id,
                        provider=provider,
                        operation="submit",
                    )

                batch_logger.info(
                    f"Successfully submitted batch job {job_id} to provider {job.provider}",
                    job_id=job_id,
                    provider=provider,
                    operation="submit",
                    provider_batch_id=submit_result.provider_batch_id,
                )

                return {
                    "job_id": job_id,
                    "provider_batch_id": submit_result.provider_batch_id,
                    "status": "submitted",
                    "item_count": submit_result.item_count,
                    "submitted_at": submit_result.submitted_at.isoformat(),
                    "duration": time.time() - start_time,
                }

        # Execute async submission logic and capture result
        result = run_in_worker_loop(_submit())
        duration = time.time() - start_time

        batch_sentry.log_operation_complete(
            "submit",
            job_id=job_id,
            provider=provider,
            success=True,
            duration_seconds=duration,
        )

        return result

    except ProviderPermanentError as e:
        duration = time.time() - start_time

        batch_logger.error(
            f"Permanent error submitting batch job {job_id}: {e}",
            job_id=job_id,
            provider=provider,
            operation="submit",
            error_details={"error": str(e), "error_type": "permanent"},
        )

        # Record error metrics
        if provider:
            record_error(provider, "permanent", "submit")

        # Capture in Sentry
        batch_sentry.capture_batch_error(
            e, job_id=job_id, provider=provider, operation="submit"
        )

        # Update job status to failed
        async def _mark_failed():
            async with get_async_session() as session:
                batch_service = BatchService(session)
                await batch_service.update_batch_job(
                    job_uuid,
                    BatchJobUpdate(status=BatchJobStatus.FAILED, error_summary=str(e)),
                )

                # Record status change
                if provider:
                    record_job_status_change(provider, "failed")

        # Persist failed state asynchronously
        run_in_worker_loop(_mark_failed())

        batch_sentry.log_operation_complete(
            "submit",
            job_id=job_id,
            provider=provider,
            success=False,
            duration_seconds=duration,
        )

        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "duration": duration,
        }

    except ProviderTransientError as e:
        duration = time.time() - start_time

        batch_logger.warning(
            f"Transient error submitting batch job {job_id}: {e}",
            job_id=job_id,
            provider=provider,
            operation="submit",
            error_details={
                "error": str(e),
                "error_type": "transient",
                "retry_count": self.request.retries,
            },
        )

        # Record retry attempt
        if provider:
            record_retry_attempt(provider, "submit", "transient_error")
            record_error(provider, "transient", "submit")

        # Capture in Sentry
        batch_sentry.capture_batch_error(
            e,
            job_id=job_id,
            provider=provider,
            operation="submit",
            additional_context={"retry_count": self.request.retries},
        )

        raise self.retry(countdown=min(2**self.request.retries, 300), exc=e)

    except Exception as e:
        duration = time.time() - start_time

        batch_logger.error(
            f"Unexpected error submitting batch job {job_id}: {e}",
            job_id=job_id,
            provider=provider,
            operation="submit",
            error_details={"error": str(e), "error_type": "unexpected"},
        )

        # Record error metrics
        if provider:
            record_error(provider, "unexpected", "submit")

        # Capture in Sentry
        batch_sentry.capture_batch_error(
            e, job_id=job_id, provider=provider, operation="submit"
        )

        batch_sentry.log_operation_complete(
            "submit",
            job_id=job_id,
            provider=provider,
            success=False,
            duration_seconds=duration,
        )

        return {
            "job_id": job_id,
            "status": "error",
            "error": str(e),
            "duration": duration,
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

    # Log polling cycle start
    batch_logger.info(
        "Starting batch poll cycle", operation="poll", poll_cycle_start=start_time
    )
    batch_sentry.log_operation_start("poll")

    # Try to acquire lock
    lock_acquired = redis_client.set(
        BATCH_POLL_LOCK_KEY, "1", nx=True, ex=BATCH_POLL_LOCK_TIMEOUT
    )

    if not lock_acquired:
        batch_logger.info(
            "Batch poll already running, skipping",
            operation="poll",
            skip_reason="lock_held",
        )
        return {
            "status": "skipped",
            "reason": "Another poll process is running",
            "duration": time.time() - start_time,
        }

    try:
        batch_logger.info("Starting batch poll", operation="poll")

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
                    batch_logger.info("No jobs to poll", operation="poll", jobs_found=0)
                    return {
                        "status": "completed",
                        "jobs_polled": 0,
                        "status_changes": 0,
                        "duration": time.time() - start_time,
                    }

                batch_logger.info(
                    f"Found {len(jobs)} jobs to poll",
                    operation="poll",
                    jobs_found=len(jobs),
                )

                status_changes = 0
                jobs_polled = 0
                provider_stats = {}  # Track per-provider statistics

                for job in jobs:
                    provider = job.provider

                    # Initialize provider stats
                    if provider not in provider_stats:
                        provider_stats[provider] = {
                            "polled": 0,
                            "status_changes": 0,
                            "errors": 0,
                        }

                    try:
                        # Set Sentry context
                        batch_sentry.set_batch_context(
                            job_id=job.id,
                            provider=provider,
                            provider_batch_id=job.provider_batch_id,
                            operation="poll",
                        )

                        # Get provider instance
                        provider_instance = registry.create_provider(
                            job.provider,
                            config=llm_config.get_provider_runtime_config(job.provider),
                        )

                        # Poll provider status using decorator for automatic timing
                        @time_provider_operation(provider, "poll")
                        def poll_batch_status():
                            return provider_instance.poll_status(job.provider_batch_id)

                        provider_status = poll_batch_status()

                        jobs_polled += 1
                        provider_stats[provider]["polled"] += 1

                        batch_logger.debug(
                            f"Polled job {job.id} status",
                            job_id=job.id,
                            provider=provider,
                            provider_batch_id=job.provider_batch_id,
                            operation="poll",
                            provider_status=provider_status.normalized_status,
                            current_status=job.status.value,
                        )

                        # Check if status changed
                        new_status = BatchJobStatus(provider_status.normalized_status)
                        if new_status != job.status:
                            status_changes += 1
                            provider_stats[provider]["status_changes"] += 1

                            old_status = job.status.value

                            # Update job status
                            update_data = {"status": new_status}

                            # Handle completion
                            if new_status in [
                                BatchJobStatus.COMPLETED,
                                BatchJobStatus.FAILED,
                            ]:
                                completed_at = datetime.now(timezone.utc)
                                update_data["completed_at"] = completed_at

                                # Calculate and record job duration
                                job_start_time = job.submitted_at or job.created_at
                                if job_start_time:
                                    duration = (
                                        completed_at - job_start_time
                                    ).total_seconds()
                                    record_job_duration(
                                        provider, new_status.value, duration
                                    )

                                # Update in-progress counter
                                update_jobs_in_progress(provider, -1)

                                # For completed jobs, schedule fetch task
                                if new_status == BatchJobStatus.COMPLETED:
                                    batch_fetch.delay(str(job.id))

                                    batch_logger.debug(
                                        f"Scheduled fetch task for completed job {job.id}",
                                        job_id=job.id,
                                        provider=provider,
                                        provider_batch_id=job.provider_batch_id,
                                        operation="poll",
                                    )

                            await batch_service.update_batch_job(
                                job.id, BatchJobUpdate(**update_data)
                            )

                            # Record metrics
                            record_job_status_change(provider, new_status.value)

                            # Log structured status change
                            batch_logger.log_job_lifecycle_event(
                                "status_changed",
                                job.id,
                                provider,
                                old_status=old_status,
                                new_status=new_status.value,
                                provider_batch_id=job.provider_batch_id,
                            )

                            # Sentry breadcrumb
                            batch_sentry.log_status_change(
                                job.id,
                                provider,
                                old_status,
                                new_status.value,
                                provider_batch_id=job.provider_batch_id,
                            )

                            batch_logger.info(
                                f"Job {job.id} status changed: {job.status} -> {new_status}",
                                job_id=job.id,
                                provider=provider,
                                provider_batch_id=job.provider_batch_id,
                                operation="poll",
                                status_transition=f"{old_status}->{new_status.value}",
                            )

                    except ProviderPermanentError as e:
                        provider_stats[provider]["errors"] += 1

                        batch_logger.error(
                            f"Permanent error polling job {job.id}: {e}",
                            job_id=job.id,
                            provider=provider,
                            provider_batch_id=job.provider_batch_id,
                            operation="poll",
                            error_details={"error": str(e), "error_type": "permanent"},
                        )

                        # Record error metrics
                        record_error(provider, "permanent", "poll")

                        # Capture in Sentry
                        batch_sentry.capture_batch_error(
                            e,
                            job_id=job.id,
                            provider=provider,
                            provider_batch_id=job.provider_batch_id,
                            operation="poll",
                        )

                        # Mark job as failed
                        await batch_service.update_batch_job(
                            job.id,
                            BatchJobUpdate(
                                status=BatchJobStatus.FAILED,
                                error_summary=str(e),
                                completed_at=datetime.now(timezone.utc),
                            ),
                        )

                        # Record status change and update counters
                        record_job_status_change(provider, "failed")
                        update_jobs_in_progress(provider, -1)
                        status_changes += 1

                    except ProviderTransientError as e:
                        provider_stats[provider]["errors"] += 1

                        batch_logger.warning(
                            f"Transient error polling job {job.id}: {e}",
                            job_id=job.id,
                            provider=provider,
                            provider_batch_id=job.provider_batch_id,
                            operation="poll",
                            error_details={"error": str(e), "error_type": "transient"},
                        )

                        # Record error metrics
                        record_error(provider, "transient", "poll")

                        # Continue with other jobs, don't fail the whole poll
                        continue

                    except Exception as e:
                        provider_stats[provider]["errors"] += 1

                        batch_logger.error(
                            f"Unexpected error polling job {job.id}: {e}",
                            job_id=job.id,
                            provider=provider,
                            provider_batch_id=job.provider_batch_id,
                            operation="poll",
                            error_details={"error": str(e), "error_type": "unexpected"},
                        )

                        # Record error metrics
                        record_error(provider, "unexpected", "poll")

                        # Capture in Sentry
                        batch_sentry.capture_batch_error(
                            e,
                            job_id=job.id,
                            provider=provider,
                            provider_batch_id=job.provider_batch_id,
                            operation="poll",
                        )

                        continue

                batch_logger.info(
                    f"Batch poll completed: {jobs_polled} jobs polled, {status_changes} status changes",
                    operation="poll",
                    jobs_polled=jobs_polled,
                    status_changes=status_changes,
                    provider_stats=provider_stats,
                )

                return {
                    "status": "completed",
                    "jobs_polled": jobs_polled,
                    "status_changes": status_changes,
                    "provider_stats": provider_stats,
                    "duration": time.time() - start_time,
                }

        # Execute async polling logic
        result = run_in_worker_loop(_poll())
        duration = time.time() - start_time

        record_poll_cycle_duration(duration)

        batch_sentry.log_operation_complete(
            "poll",
            success=True,
            duration_seconds=duration,
            additional_data={
                "jobs_polled": result.get("jobs_polled", 0),
                "status_changes": result.get("status_changes", 0),
            },
        )

        return result

    except Exception as e:
        duration = time.time() - start_time

        batch_logger.error(
            f"Error in batch poll: {e}",
            operation="poll",
            error_details={"error": str(e), "error_type": "unexpected"},
        )

        # Capture in Sentry
        batch_sentry.capture_batch_error(e, operation="poll")

        batch_sentry.log_operation_complete(
            "poll", success=False, duration_seconds=duration
        )

        return {
            "status": "error",
            "error": str(e),
            "duration": duration,
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
    provider = None

    try:
        batch_logger.info(
            f"Starting batch fetch for job {job_id}", job_id=job_id, operation="fetch"
        )
        batch_sentry.log_operation_start("fetch", job_id=job_id)

        async def _fetch():
            nonlocal provider
            async with get_async_session() as session:
                batch_service = BatchService(session)

                # Get batch job
                job = await batch_service.get_batch_job(job_uuid)
                if not job:
                    raise ValueError(f"Batch job {job_id} not found")

                provider = job.provider

                # Set Sentry context
                batch_sentry.set_batch_context(
                    job_id=job.id,
                    provider=provider,
                    provider_batch_id=job.provider_batch_id,
                    operation="fetch",
                )

                # Check if job is completed
                if job.status != BatchJobStatus.COMPLETED:
                    batch_logger.warning(
                        f"Job {job_id} not completed (status: {job.status})",
                        job_id=job_id,
                        provider=provider,
                        provider_batch_id=job.provider_batch_id,
                        operation="fetch",
                        current_status=job.status.value,
                    )
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
                    batch_logger.info(
                        f"Job {job_id} already has all results fetched",
                        job_id=job_id,
                        provider=provider,
                        provider_batch_id=job.provider_batch_id,
                        operation="fetch",
                        total_items=len(items),
                        completed_items=len(completed_items),
                    )
                    return {
                        "job_id": job_id,
                        "status": "already_fetched",
                        "message": "All results already fetched",
                        "duration": time.time() - start_time,
                    }

                batch_logger.debug(
                    f"Fetching results for {len(items) - len(completed_items)} items",
                    job_id=job_id,
                    provider=provider,
                    provider_batch_id=job.provider_batch_id,
                    operation="fetch",
                    total_items=len(items),
                    completed_items=len(completed_items),
                    pending_items=len(items) - len(completed_items),
                )

                # Get provider instance
                provider_instance = registry.create_provider(
                    job.provider,
                    config=llm_config.get_provider_runtime_config(job.provider),
                )

                # Fetch results from provider using decorator for automatic timing
                @time_provider_operation(provider, "fetch")
                def fetch_batch_results():
                    return provider_instance.fetch_results(job.provider_batch_id)

                results = fetch_batch_results()

                batch_logger.info(
                    f"Fetched {len(results)} results from provider",
                    job_id=job_id,
                    provider=provider,
                    provider_batch_id=job.provider_batch_id,
                    operation="fetch",
                    results_count=len(results),
                )

                # If provider is gemini, attempt to remap synthetic item_X ids to real UUIDs using submission_metadata
                if job.provider == "gemini":
                    try:
                        meta = (
                            job.get_metadata() if hasattr(job, "get_metadata") else {}
                        )
                        submission_meta = meta.get("submission_metadata", {}).get(
                            str(job.id), {}
                        )
                        custom_map = (
                            submission_meta.get("custom_id_mapping")
                            if isinstance(submission_meta, dict)
                            else None
                        )
                        # custom_map shape: {index: original_uuid}
                        if custom_map:
                            remapped = []
                            remapped_count = 0
                            for r in results:
                                if r.custom_id.startswith("item_"):
                                    try:
                                        idx = r.custom_id.split("_", 1)[1]
                                        original = custom_map.get(idx)
                                        if original:
                                            r.custom_id = str(original)
                                            remapped_count += 1
                                    except Exception:
                                        pass
                                remapped.append(r)
                            results = remapped

                            batch_logger.debug(
                                f"Remapped {remapped_count} Gemini custom IDs",
                                job_id=job_id,
                                provider=provider,
                                operation="fetch",
                                remapped_count=remapped_count,
                            )

                    except Exception as remap_err:  # pragma: no cover (defensive)
                        batch_logger.warning(
                            f"Failed to remap Gemini custom_ids for job {job.id}: {remap_err}",
                            job_id=job_id,
                            provider=provider,
                            operation="fetch",
                            error_details={"error": str(remap_err)},
                        )

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

                # Record item metrics
                record_item_status_change(provider, None, "completed", success_count)
                record_item_status_change(provider, None, "failed", error_count)

                # Update job counts
                await batch_service.update_batch_job(
                    job_uuid,
                    BatchJobUpdate(
                        success_count=success_count, error_count=error_count
                    ),
                )

                # Log batch item update
                batch_logger.log_item_batch_update(
                    job.id,
                    provider,
                    success_count,
                    error_count,
                    len(items),
                    provider_batch_id=job.provider_batch_id,
                )

                # Emit semantic fetch completed event
                try:
                    from inference_core.database.sql.models.batch import BatchEventType

                    await batch_service.create_semantic_event(
                        job_uuid,
                        BatchEventType.FETCH_COMPLETED,
                        event_data={
                            "success_count": success_count,
                            "error_count": error_count,
                            "total_items": len(items),
                        },
                    )
                except Exception as evt_err:  # pragma: no cover
                    batch_logger.warning(
                        f"Failed to create FETCH_COMPLETED event for job {job_id}: {evt_err}",
                        job_id=job_id,
                        provider=provider,
                        operation="fetch",
                    )

                batch_logger.info(
                    f"Fetched results for job {job_id}: {success_count} successful, {error_count} failed",
                    job_id=job_id,
                    provider=provider,
                    provider_batch_id=job.provider_batch_id,
                    operation="fetch",
                    success_count=success_count,
                    error_count=error_count,
                    total_items=len(items),
                )

                return {
                    "job_id": job_id,
                    "status": "completed",
                    "success_count": success_count,
                    "error_count": error_count,
                    "total_items": len(items),
                    "duration": time.time() - start_time,
                }

        # Execute async fetch logic
        result = run_in_worker_loop(_fetch())
        duration = time.time() - start_time

        batch_sentry.log_operation_complete(
            "fetch",
            job_id=job_id,
            provider=provider,
            success=True,
            duration_seconds=duration,
            additional_data={
                "success_count": result.get("success_count", 0),
                "error_count": result.get("error_count", 0),
                "total_items": result.get("total_items", 0),
            },
        )

        return result

    except ProviderPermanentError as e:
        duration = time.time() - start_time

        batch_logger.error(
            f"Permanent error fetching batch job {job_id}: {e}",
            job_id=job_id,
            provider=provider,
            operation="fetch",
            error_details={"error": str(e), "error_type": "permanent"},
        )

        # Record error metrics
        if provider:
            record_error(provider, "permanent", "fetch")

        # Capture in Sentry
        batch_sentry.capture_batch_error(
            e, job_id=job_id, provider=provider, operation="fetch"
        )

        batch_sentry.log_operation_complete(
            "fetch",
            job_id=job_id,
            provider=provider,
            success=False,
            duration_seconds=duration,
        )

        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "duration": duration,
        }

    except ProviderTransientError as e:
        duration = time.time() - start_time

        batch_logger.warning(
            f"Transient error fetching batch job {job_id}: {e}",
            job_id=job_id,
            provider=provider,
            operation="fetch",
            error_details={
                "error": str(e),
                "error_type": "transient",
                "retry_count": self.request.retries,
            },
        )

        # Record retry attempt and error
        if provider:
            record_retry_attempt(provider, "fetch", "transient_error")
            record_error(provider, "transient", "fetch")

        # Capture in Sentry
        batch_sentry.capture_batch_error(
            e,
            job_id=job_id,
            provider=provider,
            operation="fetch",
            additional_context={"retry_count": self.request.retries},
        )

        raise self.retry(countdown=min(2**self.request.retries, 300), exc=e)

    except Exception as e:
        duration = time.time() - start_time

        batch_logger.error(
            f"Unexpected error fetching batch job {job_id}: {e}",
            job_id=job_id,
            provider=provider,
            operation="fetch",
            error_details={"error": str(e), "error_type": "unexpected"},
        )

        # Record error metrics
        if provider:
            record_error(provider, "unexpected", "fetch")

        # Capture in Sentry
        batch_sentry.capture_batch_error(
            e, job_id=job_id, provider=provider, operation="fetch"
        )

        batch_sentry.log_operation_complete(
            "fetch",
            job_id=job_id,
            provider=provider,
            success=False,
            duration_seconds=duration,
        )

        return {
            "job_id": job_id,
            "status": "error",
            "error": str(e),
            "duration": duration,
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

        return run_in_worker_loop(_retry())

    except Exception as e:
        logger.error(f"Error retrying failed items from job {job_id}: {e}")
        return {
            "job_id": job_id,
            "status": "error",
            "error": str(e),
            "duration": time.time() - start_time,
        }

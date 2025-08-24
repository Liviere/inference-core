"""
Batch Service

Service layer for batch processing operations.
Provides CRUD operations and business logic for BatchJob, BatchItem, and BatchEvent.
"""

import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from inference_core.database.sql.models.batch import (
    BatchEvent,
    BatchEventType,
    BatchItem,
    BatchItemStatus,
    BatchJob,
    BatchJobStatus,
)
from inference_core.schemas.batch import (
    BatchEventCreate,
    BatchItemCreate,
    BatchItemUpdate,
    BatchJobCreate,
    BatchJobQuery,
    BatchJobStats,
    BatchJobUpdate,
)

logger = logging.getLogger(__name__)


class BatchService:
    """Service for batch processing operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    # BatchJob operations
    async def create_batch_job(
        self, job_data: BatchJobCreate, created_by: Optional[UUID] = None
    ) -> BatchJob:
        """Create a new batch job"""
        job = BatchJob(
            provider=job_data.provider,
            model=job_data.model,
            mode=job_data.mode,
            status=BatchJobStatus.CREATED,
            request_count=job_data.request_count,
            config_json=job_data.config_json,
            created_by=created_by,
        )

        self.session.add(job)
        await self.session.commit()
        await self.session.refresh(job)

        # Create initial event
        await self._create_status_change_event(
            job.id, None, BatchJobStatus.CREATED, created_by
        )

        logger.info(f"Created batch job {job.id} for provider {job.provider}")
        return job

    async def get_batch_job(self, job_id: UUID) -> Optional[BatchJob]:
        """Get batch job by ID"""
        result = await self.session.execute(
            select(BatchJob).where(
                and_(BatchJob.id == job_id, BatchJob.is_deleted == False)
            )
        )
        return result.scalar_one_or_none()

    async def get_batch_job_with_items(self, job_id: UUID) -> Optional[BatchJob]:
        """Get batch job with its items"""
        result = await self.session.execute(
            select(BatchJob)
            .options(selectinload(BatchJob.items))
            .where(and_(BatchJob.id == job_id, BatchJob.is_deleted == False))
        )
        return result.scalar_one_or_none()

    async def update_batch_job(
        self,
        job_id: UUID,
        job_data: BatchJobUpdate,
        updated_by: Optional[UUID] = None,
    ) -> Optional[BatchJob]:
        """Update batch job"""
        job = await self.get_batch_job(job_id)
        if not job:
            return None

        old_status = job.status

        # Update fields
        update_data = job_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            if hasattr(job, field):
                setattr(job, field, value)

        job.updated_by = updated_by

        await self.session.commit()
        await self.session.refresh(job)

        # Create status change event if status changed
        if job_data.status and job_data.status != old_status:
            await self._create_status_change_event(
                job.id, old_status, job_data.status, updated_by
            )

        logger.info(f"Updated batch job {job.id}")
        return job

    async def query_batch_jobs(self, query: BatchJobQuery) -> List[BatchJob]:
        """Query batch jobs with filters"""
        stmt = select(BatchJob).where(BatchJob.is_deleted == False)

        if query.provider:
            stmt = stmt.where(BatchJob.provider == query.provider)
        if query.status:
            stmt = stmt.where(BatchJob.status == query.status)
        if query.mode:
            stmt = stmt.where(BatchJob.mode == query.mode)
        if query.submitted_after:
            stmt = stmt.where(BatchJob.submitted_at >= query.submitted_after)
        if query.submitted_before:
            stmt = stmt.where(BatchJob.submitted_at <= query.submitted_before)

        stmt = stmt.order_by(desc(BatchJob.created_at))
        stmt = stmt.offset(query.offset).limit(query.limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_pending_jobs(self, provider: Optional[str] = None) -> List[BatchJob]:
        """Get jobs that are pending processing"""
        stmt = select(BatchJob).where(
            and_(
                BatchJob.is_deleted == False,
                BatchJob.status.in_(
                    [
                        BatchJobStatus.CREATED,
                        BatchJobStatus.SUBMITTED,
                        BatchJobStatus.IN_PROGRESS,
                    ]
                ),
            )
        )

        if provider:
            stmt = stmt.where(BatchJob.provider == provider)

        stmt = stmt.order_by(BatchJob.created_at)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def delete_batch_job(self, job_id: UUID) -> bool:
        """Soft delete batch job"""
        job = await self.get_batch_job(job_id)
        if not job:
            return False

        job.is_deleted = True
        job.deleted_at = datetime.now(UTC)

        await self.session.commit()
        logger.info(f"Deleted batch job {job.id}")
        return True

    # BatchItem operations
    async def create_batch_items(
        self,
        job_id: UUID,
        items_data: List[BatchItemCreate],
        created_by: Optional[UUID] = None,
    ) -> List[BatchItem]:
        """Create multiple batch items for a job"""
        job = await self.get_batch_job(job_id)
        if not job:
            raise ValueError(f"Batch job {job_id} not found")

        items = []
        for item_data in items_data:
            item = BatchItem(
                batch_job_id=job_id,
                sequence_index=item_data.sequence_index,
                custom_external_id=item_data.custom_external_id,
                input_payload=item_data.input_payload,
                status=BatchItemStatus.QUEUED,
                created_by=created_by,
            )
            items.append(item)
            self.session.add(item)

        # Update job request count
        job.request_count = len(items_data)

        await self.session.commit()

        # Refresh all items
        for item in items:
            await self.session.refresh(item)

        logger.info(f"Created {len(items)} batch items for job {job_id}")
        return items

    async def get_batch_items(
        self, job_id: UUID, status: Optional[BatchItemStatus] = None
    ) -> List[BatchItem]:
        """Get batch items for a job"""
        stmt = select(BatchItem).where(
            and_(BatchItem.batch_job_id == job_id, BatchItem.is_deleted == False)
        )

        if status:
            stmt = stmt.where(BatchItem.status == status)

        stmt = stmt.order_by(BatchItem.sequence_index)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_batch_item(
        self,
        item_id: UUID,
        item_data: BatchItemUpdate,
        updated_by: Optional[UUID] = None,
    ) -> Optional[BatchItem]:
        """Update batch item and propagate status changes to job"""
        result = await self.session.execute(
            select(BatchItem).where(
                and_(BatchItem.id == item_id, BatchItem.is_deleted == False)
            )
        )
        item = result.scalar_one_or_none()
        if not item:
            return None

        old_status = item.status

        # Update fields
        update_data = item_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            if hasattr(item, field):
                setattr(item, field, value)

        item.updated_by = updated_by

        await self.session.commit()
        await self.session.refresh(item)

        # Update job counts if status changed
        if item_data.status and item_data.status != old_status:
            await self._update_job_counts_from_item_status_change(
                item.batch_job_id, old_status, item_data.status
            )

        return item

    async def update_batch_items_status(
        self,
        items_status_updates: List[Dict[str, Any]],
        updated_by: Optional[UUID] = None,
    ) -> int:
        """Bulk update batch items status"""
        updated_count = 0

        for update in items_status_updates:
            item_id = update.get("item_id")
            new_status = update.get("status")

            if item_id and new_status:
                item_update = BatchItemUpdate(
                    status=new_status,
                    output_payload=update.get("output_payload"),
                    error_detail=update.get("error_detail"),
                )

                if await self.update_batch_item(item_id, item_update, updated_by):
                    updated_count += 1

        return updated_count

    # BatchEvent operations
    async def create_batch_event(
        self,
        job_id: UUID,
        event_data: BatchEventCreate,
        created_by: Optional[UUID] = None,
    ) -> BatchEvent:
        """Create a batch event"""
        event = BatchEvent(
            batch_job_id=job_id,
            event_type=event_data.event_type,
            old_status=event_data.old_status,
            new_status=event_data.new_status,
            event_data=event_data.event_data,
            created_by=created_by,
        )

        self.session.add(event)
        await self.session.commit()
        await self.session.refresh(event)

        return event

    async def create_semantic_event(
        self,
        job_id: UUID,
        event_type: BatchEventType,
        event_data: Optional[Dict[str, Any]] = None,
        created_by: Optional[UUID] = None,
    ) -> BatchEvent:
        """Create a semantic (non-status-change) event.

        Args:
            job_id: Target batch job
            event_type: Semantic event type (submitted, fetch_completed, ...)
            event_data: Additional metadata
            created_by: Optional user id
        """
        be = BatchEventCreate(
            event_type=event_type,
            old_status=None,
            new_status=None,
            event_data=event_data or {},
        )
        return await self.create_batch_event(job_id, be, created_by)

    async def get_batch_events(
        self, job_id: UUID, event_type: Optional[BatchEventType] = None
    ) -> List[BatchEvent]:
        """Get batch events for a job"""
        stmt = select(BatchEvent).where(BatchEvent.batch_job_id == job_id)

        if event_type:
            stmt = stmt.where(BatchEvent.event_type == event_type)
        # Ascending order -> natural chronological timeline
        stmt = stmt.order_by(
            BatchEvent.event_timestamp, BatchEvent.created_at, BatchEvent.id
        )

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # Statistics and reporting
    async def get_batch_stats(self) -> BatchJobStats:
        """Get batch processing statistics"""
        # Total jobs
        total_jobs_result = await self.session.execute(
            select(func.count(BatchJob.id)).where(BatchJob.is_deleted == False)
        )
        total_jobs = total_jobs_result.scalar() or 0

        # Jobs by status
        status_result = await self.session.execute(
            select(BatchJob.status, func.count(BatchJob.id))
            .where(BatchJob.is_deleted == False)
            .group_by(BatchJob.status)
        )
        jobs_by_status = {status: count for status, count in status_result.fetchall()}

        # Jobs by provider
        provider_result = await self.session.execute(
            select(BatchJob.provider, func.count(BatchJob.id))
            .where(BatchJob.is_deleted == False)
            .group_by(BatchJob.provider)
        )
        jobs_by_provider = {
            provider: count for provider, count in provider_result.fetchall()
        }

        # Request statistics
        stats_result = await self.session.execute(
            select(
                func.sum(BatchJob.request_count),
                func.sum(BatchJob.success_count),
                func.sum(BatchJob.error_count),
            ).where(BatchJob.is_deleted == False)
        )
        total_requests, total_successes, total_errors = stats_result.fetchone() or (
            0,
            0,
            0,
        )

        # Average success rate
        avg_success_rate = 0.0
        if total_requests and total_requests > 0:
            completed_requests = (total_successes or 0) + (total_errors or 0)
            if completed_requests > 0:
                avg_success_rate = (total_successes or 0) / completed_requests * 100

        return BatchJobStats(
            total_jobs=total_jobs,
            jobs_by_status=jobs_by_status,
            jobs_by_provider=jobs_by_provider,
            total_requests=total_requests or 0,
            total_successes=total_successes or 0,
            total_errors=total_errors or 0,
            average_success_rate=avg_success_rate,
        )

    # Private helper methods
    async def _create_status_change_event(
        self,
        job_id: UUID,
        old_status: Optional[BatchJobStatus],
        new_status: BatchJobStatus,
        created_by: Optional[UUID] = None,
    ):
        """Create a status change event"""
        # Normalize old_status/new_status in case underlying ORM returned raw strings
        if isinstance(old_status, str):
            try:
                old_status_enum = BatchJobStatus(old_status)
            except ValueError:
                old_status_enum = None
        else:
            old_status_enum = old_status
        if isinstance(new_status, str):
            try:
                new_status_enum = BatchJobStatus(new_status)
            except ValueError:
                new_status_enum = None
        else:
            new_status_enum = new_status
        event_data = BatchEventCreate(
            event_type=BatchEventType.STATUS_CHANGE,
            old_status=old_status_enum.value if old_status_enum else None,
            new_status=new_status_enum.value if new_status_enum else None,
        )
        await self.create_batch_event(job_id, event_data, created_by)

    async def _update_job_counts_from_item_status_change(
        self,
        job_id: UUID,
        old_status: BatchItemStatus,
        new_status: BatchItemStatus,
    ):
        """Update job success/error counts based on item status change"""
        job = await self.get_batch_job(job_id)
        if not job:
            return

        # Calculate deltas
        success_delta = 0
        error_delta = 0

        # Remove from old status count
        if old_status == BatchItemStatus.COMPLETED:
            success_delta -= 1
        elif old_status == BatchItemStatus.FAILED:
            error_delta -= 1

        # Add to new status count
        if new_status == BatchItemStatus.COMPLETED:
            success_delta += 1
        elif new_status == BatchItemStatus.FAILED:
            error_delta += 1

        # Update job counts
        if success_delta != 0 or error_delta != 0:
            job.update_counts(success_delta, error_delta)
            await self.session.commit()

            # Check if job is complete and update status
            if job.success_count + job.error_count >= job.request_count:
                if job.error_count == 0:
                    job.status = BatchJobStatus.COMPLETED
                else:
                    job.status = (
                        BatchJobStatus.COMPLETED
                    )  # or FAILED based on error threshold
                job.completed_at = datetime.now(UTC)
                await self.session.commit()

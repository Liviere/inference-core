"""
Batch Processing API Endpoints

FastAPI endpoints for batch processing operations.
Provides endpoints to create, query, list items, and cancel batch jobs.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from inference_core.core.dependecies import get_current_active_user, get_db
from inference_core.database.sql.models.batch import BatchItemStatus, BatchJobStatus
from inference_core.llm.config import ModelProvider, llm_config
from inference_core.schemas.batch import (
    BatchCancelResponse,
    BatchItemListResponse,
    BatchJobCreateRequest,
    BatchJobCreateResponse,
    BatchJobDetailResponse,
)
from inference_core.services.batch_service import BatchService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/llm/batch",
    tags=["Batch Processing"],
)


async def get_batch_service(db: AsyncSession = Depends(get_db)) -> BatchService:
    """Dependency to get BatchService instance"""
    return BatchService(db)


@router.post(
    "/", response_model=BatchJobCreateResponse, status_code=status.HTTP_201_CREATED
)
async def create_batch_job(
    request: BatchJobCreateRequest,
    current_user: dict = Depends(get_current_active_user),
    batch_service: BatchService = Depends(get_batch_service),
) -> BatchJobCreateResponse:
    """
    Create a new batch processing job.

    Creates a batch job with the provided items and starts processing.
    Returns job ID and initial status immediately.

    Args:
        request: Batch job creation request with provider, model, and items
        current_user: Authenticated user
        batch_service: Batch service instance

    Returns:
        BatchJobCreateResponse with job_id and initial status

    Raises:
        HTTPException: If validation fails or creation error occurs
    """
    try:
        # --- Validation against LLM configuration ---
        provider = request.provider
        model_name = request.model

        # 1. Provider exists
        if provider not in llm_config.providers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown provider '{provider}'",
            )

        # 2. Model exists
        model_cfg = llm_config.get_model_config(model_name)
        if not model_cfg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown model '{model_name}'",
            )

        # 3. Model/provider consistency
        # model_cfg.provider may already be a string because of pydantic use_enum_values
        provider_value = (
            model_cfg.provider.value
            if hasattr(model_cfg.provider, "value")
            else str(model_cfg.provider)
        )
        if provider_value != provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Model '{model_name}' belongs to provider '{provider_value}',"
                    f" not '{provider}'"
                ),
            )

        # 4. Batch provider enabled
        if not llm_config.batch_config.is_provider_enabled(provider):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider '{provider}' is not enabled for batch processing",
            )

        # 5. Model batch-enabled for provider
        batch_model_cfg = llm_config.batch_config.get_model_config(provider, model_name)
        if not batch_model_cfg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Model '{model_name}' is not configured for batch processing under provider '{provider}'"
                ),
            )

        # Determine processing mode based on the input structure
        # For simplicity, default to "chat" mode unless specified in params
        mode = "chat"
        if request.params and "mode" in request.params:
            mode = request.params["mode"]

        # 6. Mode matches configured batch model mode
        if batch_model_cfg.mode != mode:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Mode '{mode}' not allowed for model '{model_name}'. "
                    f"Expected '{batch_model_cfg.mode}'"
                ),
            )

        # Create the batch job
        from inference_core.schemas.batch import BatchJobCreate

        job_data = BatchJobCreate(
            provider=request.provider,
            model=request.model,
            mode=mode,
            request_count=len(request.items),
            config_json=request.params,
        )

        user_id = UUID(current_user["id"])
        job = await batch_service.create_batch_job(job_data, created_by=user_id)

        # Create batch items
        from inference_core.schemas.batch import BatchItemCreate

        items_data = []
        for idx, item in enumerate(request.items):
            item_data = BatchItemCreate(
                sequence_index=idx,
                custom_external_id=item.custom_id,
                input_payload=item.input,
            )
            items_data.append(item_data)

        await batch_service.create_batch_items(job.id, items_data, created_by=user_id)

        logger.info(
            f"Created batch job {job.id} with {len(request.items)} items for user {user_id}"
        )

        return BatchJobCreateResponse(
            job_id=job.id,
            status=job.status,
            message="Batch job created successfully",
            item_count=len(request.items),
        )

    except HTTPException:
        # Re-raise validation HTTP errors so they propagate correctly
        raise
    except ValueError as e:
        logger.error(f"Validation error creating batch job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error creating batch job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create batch job: {str(e)}",
        )


@router.get("/{job_id}", response_model=BatchJobDetailResponse)
async def get_batch_job(
    job_id: UUID,
    current_user: dict = Depends(get_current_active_user),
    batch_service: BatchService = Depends(get_batch_service),
) -> BatchJobDetailResponse:
    """
    Get detailed information about a batch job.

    Returns job details including aggregated counts and status timeline.

    Args:
        job_id: Unique batch job identifier
        current_user: Authenticated user
        batch_service: Batch service instance

    Returns:
        Detailed batch job information with events

    Raises:
        HTTPException: If job not found
    """
    try:
        job = await batch_service.get_batch_job(job_id)
        user_id = UUID(current_user["id"])
        if not job or job.created_by != user_id:
            # Return 404 to avoid leaking existence of other users' jobs
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch job {job_id} not found",
            )

        # Get job events for the timeline
        events = await batch_service.get_batch_events(job_id)

        # Convert to response format
        from inference_core.schemas.batch import (
            BatchEventResponse,
            BatchJobDetailResponse,
        )

        # Calculate computed properties
        completion_rate = 0.0
        success_rate = 0.0
        is_complete = job.status in [
            BatchJobStatus.COMPLETED,
            BatchJobStatus.FAILED,
            BatchJobStatus.CANCELLED,
        ]
        pending_count = max(0, job.request_count - job.success_count - job.error_count)

        if job.request_count > 0:
            completed = job.success_count + job.error_count
            completion_rate = (completed / job.request_count) * 100
            if completed > 0:
                success_rate = (job.success_count / completed) * 100

        return BatchJobDetailResponse(
            id=job.id,
            provider=job.provider,
            model=job.model,
            mode=job.mode,
            status=job.status,
            provider_batch_id=job.provider_batch_id,
            submitted_at=job.submitted_at,
            completed_at=job.completed_at,
            request_count=job.request_count,
            success_count=job.success_count,
            error_count=job.error_count,
            error_summary=job.error_summary,
            result_uri=job.result_uri,
            config_json=job.config_json,
            created_at=job.created_at,
            updated_at=job.updated_at,
            created_by=job.created_by,
            updated_by=job.updated_by,
            completion_rate=completion_rate,
            success_rate=success_rate,
            is_complete=is_complete,
            pending_count=pending_count,
            events=[
                BatchEventResponse(
                    id=event.id,
                    batch_job_id=event.batch_job_id,
                    event_type=event.event_type,
                    old_status=event.old_status,
                    new_status=event.new_status,
                    event_timestamp=event.event_timestamp,
                    event_data=event.event_data,
                    created_at=event.created_at,
                )
                for event in events
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get batch job: {str(e)}",
        )


@router.get("/{job_id}/items", response_model=BatchItemListResponse)
async def get_batch_items(
    job_id: UUID,
    item_status: Optional[BatchItemStatus] = Query(
        None, description="Filter by item status"
    ),
    limit: int = Query(100, description="Maximum number of results", ge=1, le=1000),
    offset: int = Query(0, description="Number of results to skip", ge=0),
    current_user: dict = Depends(get_current_active_user),
    batch_service: BatchService = Depends(get_batch_service),
) -> BatchItemListResponse:
    """
    Get paginated list of items for a batch job.

    Supports filtering by status and pagination.

    Args:
        job_id: Unique batch job identifier
        item_status: Optional filter by item status
        limit: Maximum number of items to return
        offset: Number of items to skip
        current_user: Authenticated user
        batch_service: Batch service instance

    Returns:
        Paginated list of batch items

    Raises:
        HTTPException: If job not found
    """
    try:
        # Verify job exists
        job = await batch_service.get_batch_job(job_id)
        user_id = UUID(current_user["id"])
        if not job or job.created_by != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch job {job_id} not found",
            )

        # Get items with filtering
        items = await batch_service.get_batch_items(job_id, status=item_status)

        # Apply pagination
        total = len(items)
        paginated_items = items[offset : offset + limit]
        has_more = offset + limit < total

        # Convert to response format
        from inference_core.schemas.batch import BatchItemResponse

        item_responses = []
        for item in paginated_items:
            is_completed = item.status in [
                BatchItemStatus.COMPLETED,
                BatchItemStatus.FAILED,
            ]
            is_successful = item.status == BatchItemStatus.COMPLETED

            item_responses.append(
                BatchItemResponse(
                    id=item.id,
                    batch_job_id=item.batch_job_id,
                    sequence_index=item.sequence_index,
                    custom_external_id=item.custom_external_id,
                    input_payload=item.input_payload,
                    output_payload=item.output_payload,
                    status=item.status,
                    error_detail=item.error_detail,
                    created_at=item.created_at,
                    updated_at=item.updated_at,
                    is_completed=is_completed,
                    is_successful=is_successful,
                )
            )

        return BatchItemListResponse(
            items=item_responses,
            total=total,
            limit=limit,
            offset=offset,
            has_more=has_more,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch items for job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get batch items: {str(e)}",
        )


@router.post("/{job_id}/cancel", response_model=BatchCancelResponse)
async def cancel_batch_job(
    job_id: UUID,
    current_user: dict = Depends(get_current_active_user),
    batch_service: BatchService = Depends(get_batch_service),
) -> BatchCancelResponse:
    """
    Cancel a batch processing job.

    Attempts to cancel the job both with the provider and locally.
    Updates job status appropriately based on provider support.

    Args:
        job_id: Unique batch job identifier
        current_user: Authenticated user
        batch_service: Batch service instance

    Returns:
        Cancellation result with updated status

    Raises:
        HTTPException: If job not found or cannot be cancelled
    """
    try:
        # Get the job
        job = await batch_service.get_batch_job(job_id)
        user_id = UUID(current_user["id"])
        if not job or job.created_by != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch job {job_id} not found",
            )

        # Check if job can be cancelled
        if job.status in [
            BatchJobStatus.COMPLETED,
            BatchJobStatus.FAILED,
            BatchJobStatus.CANCELLED,
        ]:
            return BatchCancelResponse(
                job_id=job_id,
                status=job.status,
                message=f"Job is already {job.status.value} and cannot be cancelled",
                cancelled=False,
            )

        # Try to cancel with provider if submitted
        cancelled_with_provider = False
        if job.provider_batch_id and job.status in [
            BatchJobStatus.SUBMITTED,
            BatchJobStatus.IN_PROGRESS,
        ]:
            try:
                # Import provider registry to get provider
                from inference_core.llm.batch.registry import (
                    registry as batch_provider_registry,
                )

                registry = batch_provider_registry
                provider = registry.create_provider(job.provider)
                cancelled_with_provider = provider.cancel(job.provider_batch_id)

                logger.info(
                    f"Provider cancellation for job {job_id}: {cancelled_with_provider}"
                )

            except Exception as e:
                logger.warning(f"Failed to cancel job {job_id} with provider: {str(e)}")
                cancelled_with_provider = False

        # Update job status locally
        # user_id already defined above
        from inference_core.schemas.batch import BatchJobUpdate

        update_data = BatchJobUpdate(status=BatchJobStatus.CANCELLED)
        updated_job = await batch_service.update_batch_job(
            job_id, update_data, updated_by=user_id
        )

        if not updated_job:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update job status",
            )

        message = "Job cancelled successfully"
        if cancelled_with_provider:
            message += " (provider cancellation succeeded)"
        elif job.provider_batch_id:
            message += " (local cancellation only - provider cancellation failed)"
        else:
            message += " (local cancellation - job not yet submitted to provider)"

        logger.info(f"Cancelled batch job {job_id} for user {user_id}")

        return BatchCancelResponse(
            job_id=job_id,
            status=BatchJobStatus.CANCELLED,
            message=message,
            cancelled=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling batch job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel batch job: {str(e)}",
        )

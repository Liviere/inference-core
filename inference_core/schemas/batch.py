"""
Batch Processing Schemas

Pydantic schemas for batch processing API operations.
Includes request/response schemas for BatchJob, BatchItem, and BatchEvent.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from inference_core.database.sql.models.batch import (
    BatchEventType,
    BatchItemStatus,
    BatchJobStatus,
)


class BatchJobBase(BaseModel):
    """Base schema for batch job"""

    provider: str = Field(..., description="LLM provider name", max_length=50)
    model: str = Field(..., description="Model name", max_length=100)
    mode: str = Field(
        ...,
        description="Processing mode (chat, embedding, completion, custom)",
        max_length=20,
    )
    config_json: Optional[Dict[str, Any]] = Field(
        None, description="Batch configuration as JSON"
    )


class BatchJobCreate(BatchJobBase):
    """Schema for creating a batch job"""

    request_count: int = Field(0, description="Total number of requests", ge=0)


class BatchJobUpdate(BaseModel):
    """Schema for updating a batch job"""

    status: Optional[BatchJobStatus] = Field(None, description="Job status")
    provider_batch_id: Optional[str] = Field(
        None, description="External provider batch ID", max_length=255
    )
    submitted_at: Optional[datetime] = Field(
        None, description="When batch was submitted"
    )
    completed_at: Optional[datetime] = Field(None, description="When batch completed")
    success_count: Optional[int] = Field(
        None, description="Number of successful requests", ge=0
    )
    error_count: Optional[int] = Field(
        None, description="Number of failed requests", ge=0
    )
    error_summary: Optional[str] = Field(None, description="Summary of errors")
    result_uri: Optional[str] = Field(
        None, description="Result download URI", max_length=500
    )
    config_json: Optional[Dict[str, Any]] = Field(
        None, description="Updated batch configuration"
    )


class BatchJobResponse(BatchJobBase):
    """Schema for batch job response"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., description="Unique job identifier")
    status: BatchJobStatus = Field(..., description="Current job status")
    provider_batch_id: Optional[str] = Field(
        None, description="External provider batch ID (added after submission)"
    )
    submitted_at: Optional[datetime] = Field(
        None, description="When batch was submitted"
    )
    completed_at: Optional[datetime] = Field(None, description="When batch completed")
    request_count: int = Field(..., description="Total number of requests")
    success_count: int = Field(..., description="Number of successful requests")
    error_count: int = Field(..., description="Number of failed requests")
    error_summary: Optional[str] = Field(None, description="Summary of errors")
    result_uri: Optional[str] = Field(None, description="Result download URI")

    # Audit fields
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    created_by: Optional[UUID] = Field(None, description="Creator user ID")
    updated_by: Optional[UUID] = Field(None, description="Last updater user ID")

    # Computed properties
    completion_rate: float = Field(..., description="Completion rate percentage")
    success_rate: float = Field(..., description="Success rate percentage")
    is_complete: bool = Field(..., description="Whether batch is complete")
    pending_count: int = Field(..., description="Number of pending requests")


class BatchJobList(BaseModel):
    """Schema for batch job list response"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., description="Unique job identifier")
    provider: str = Field(..., description="LLM provider name")
    model: str = Field(..., description="Model name")
    status: BatchJobStatus = Field(..., description="Current job status")
    mode: str = Field(..., description="Processing mode")
    request_count: int = Field(..., description="Total number of requests")
    success_count: int = Field(..., description="Number of successful requests")
    error_count: int = Field(..., description="Number of failed requests")
    completion_rate: float = Field(..., description="Completion rate percentage")
    created_at: datetime = Field(..., description="Creation timestamp")
    submitted_at: Optional[datetime] = Field(
        None, description="When batch was submitted"
    )
    completed_at: Optional[datetime] = Field(None, description="When batch completed")


# Batch Item Schemas
class BatchItemBase(BaseModel):
    """Base schema for batch item"""

    sequence_index: int = Field(..., description="Order within the batch", ge=0)
    custom_external_id: Optional[str] = Field(
        None, description="Optional external identifier", max_length=255
    )
    input_payload: Optional[Dict[str, Any]] = Field(
        None, description="Request data as JSON"
    )


class BatchItemCreate(BatchItemBase):
    """Schema for creating a batch item"""

    pass


class BatchItemUpdate(BaseModel):
    """Schema for updating a batch item"""

    status: Optional[BatchItemStatus] = Field(None, description="Item status")
    output_payload: Optional[Dict[str, Any]] = Field(
        None, description="Response data as JSON"
    )
    error_detail: Optional[str] = Field(
        None, description="Error details if processing failed"
    )


class BatchItemResponse(BatchItemBase):
    """Schema for batch item response"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., description="Unique item identifier")
    batch_job_id: UUID = Field(..., description="Parent batch job ID")
    status: BatchItemStatus = Field(..., description="Current item status")
    output_payload: Optional[Dict[str, Any]] = Field(
        None, description="Response data as JSON"
    )
    error_detail: Optional[str] = Field(None, description="Error details")

    # Audit fields
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    # Computed properties
    is_completed: bool = Field(..., description="Whether item processing is complete")
    is_successful: bool = Field(..., description="Whether item completed successfully")


class BatchItemList(BaseModel):
    """Schema for batch item list response"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., description="Unique item identifier")
    sequence_index: int = Field(..., description="Order within the batch")
    custom_external_id: Optional[str] = Field(None, description="External identifier")
    status: BatchItemStatus = Field(..., description="Current item status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    is_completed: bool = Field(..., description="Whether item processing is complete")
    is_successful: bool = Field(..., description="Whether item completed successfully")


# Batch Event Schemas
class BatchEventBase(BaseModel):
    """Base schema for batch event"""

    event_type: BatchEventType = Field(..., description="Type of event")
    old_status: Optional[str] = Field(
        None, description="Previous status", max_length=20
    )
    new_status: Optional[str] = Field(None, description="New status", max_length=20)
    event_data: Optional[Dict[str, Any]] = Field(
        None, description="Additional event data"
    )


class BatchEventCreate(BatchEventBase):
    """Schema for creating a batch event"""

    pass


class BatchEventResponse(BatchEventBase):
    """Schema for batch event response"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., description="Unique event identifier")
    batch_job_id: UUID = Field(..., description="Parent batch job ID")
    event_timestamp: datetime = Field(..., description="When the event occurred")

    # Audit fields
    created_at: datetime = Field(..., description="Creation timestamp")


# Bulk operation schemas
class BatchItemBulkCreate(BaseModel):
    """Schema for bulk creating batch items"""

    items: List[BatchItemCreate] = Field(..., description="List of items to create")


class BatchJobWithItems(BatchJobResponse):
    """Schema for batch job with its items"""

    items: List[BatchItemResponse] = Field(
        default_factory=list, description="Batch items"
    )


class BatchJobStats(BaseModel):
    """Schema for batch job statistics"""

    total_jobs: int = Field(..., description="Total number of jobs")
    jobs_by_status: Dict[str, int] = Field(..., description="Jobs grouped by status")
    jobs_by_provider: Dict[str, int] = Field(
        ..., description="Jobs grouped by provider"
    )
    total_requests: int = Field(
        ..., description="Total number of requests across all jobs"
    )
    total_successes: int = Field(..., description="Total successful requests")
    total_errors: int = Field(..., description="Total failed requests")
    average_success_rate: float = Field(
        ..., description="Average success rate across all jobs"
    )


class BatchJobQuery(BaseModel):
    """Schema for batch job query parameters"""

    provider: Optional[str] = Field(None, description="Filter by provider")
    status: Optional[BatchJobStatus] = Field(None, description="Filter by status")
    mode: Optional[str] = Field(None, description="Filter by processing mode")
    submitted_after: Optional[datetime] = Field(
        None, description="Filter by submission date"
    )
    submitted_before: Optional[datetime] = Field(
        None, description="Filter by submission date"
    )
    limit: int = Field(100, description="Maximum number of results", ge=1, le=1000)
    offset: int = Field(0, description="Number of results to skip", ge=0)


class BatchItemQuery(BaseModel):
    """Schema for batch item query parameters"""

    batch_job_id: Optional[UUID] = Field(None, description="Filter by batch job")
    status: Optional[BatchItemStatus] = Field(None, description="Filter by status")
    has_error: Optional[bool] = Field(
        None, description="Filter items with/without errors"
    )
    limit: int = Field(100, description="Maximum number of results", ge=1, le=1000)
    offset: int = Field(0, description="Number of results to skip", ge=0)


# API-specific request schemas
class BatchItemInput(BaseModel):
    """Schema for batch item input in API requests"""

    input: Dict[str, Any] = Field(..., description="Input data for processing")
    custom_id: Optional[str] = Field(
        None, description="Optional custom identifier", max_length=255
    )


class BatchJobCreateRequest(BaseModel):
    """Schema for creating a batch job via API"""

    provider: str = Field(..., description="LLM provider name", max_length=50)
    model: str = Field(..., description="Model name", max_length=100)
    items: List[BatchItemInput] = Field(
        ..., description="List of items to process", min_length=1, max_length=1000
    )
    params: Optional[Dict[str, Any]] = Field(
        None, description="Additional parameters for processing"
    )


class BatchJobCreateResponse(BaseModel):
    """Schema for batch job creation response"""

    job_id: UUID = Field(..., description="Unique job identifier")
    status: BatchJobStatus = Field(..., description="Initial job status")
    message: str = Field(..., description="Success message")
    item_count: int = Field(..., description="Number of items in the batch")


class BatchJobDetailResponse(BatchJobResponse):
    """Schema for detailed batch job response with events"""

    events: List[BatchEventResponse] = Field(
        default_factory=list, description="Job events"
    )


class BatchItemListResponse(BaseModel):
    """Schema for paginated batch items response"""

    items: List[BatchItemResponse] = Field(..., description="List of batch items")
    total: int = Field(..., description="Total number of items")
    limit: int = Field(..., description="Items per page")
    offset: int = Field(..., description="Items skipped")
    has_more: bool = Field(..., description="Whether there are more items")


class BatchCancelResponse(BaseModel):
    """Schema for batch cancellation response"""

    job_id: UUID = Field(..., description="Unique job identifier")
    status: BatchJobStatus = Field(..., description="Updated job status")
    message: str = Field(..., description="Cancellation result message")
    cancelled: bool = Field(..., description="Whether cancellation was successful")

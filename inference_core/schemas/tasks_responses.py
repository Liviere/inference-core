"""
Async task response schemas
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TaskResponse(BaseModel):
    """Response model for async task submission"""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(default="PENDING", description="Current task status")
    message: str = Field(
        default="Task submitted successfully", description="Status message"
    )
    estimated_time: Optional[int] = Field(
        default=None, description="Estimated completion time in seconds"
    )


class TaskStatusResponse(BaseModel):
    """Response model for task status check"""

    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Current task status")
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Task result if completed"
    )
    info: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional task information"
    )
    traceback: Optional[str] = Field(
        default=None, description="Error traceback if task failed"
    )
    successful: Optional[bool] = Field(
        default=None, description="Whether task completed successfully"
    )
    failed: Optional[bool] = Field(default=None, description="Whether task failed")
    progress: Optional[Dict[str, Any]] = Field(
        default=None, description="Task progress information"
    )


class ActiveTasksResponse(BaseModel):
    """Response model for active tasks information"""

    active: Dict[str, Any] = Field(..., description="Currently active tasks")
    scheduled: Dict[str, Any] = Field(..., description="Scheduled tasks")
    reserved: Dict[str, Any] = Field(..., description="Reserved tasks")


class WorkerStatsResponse(BaseModel):
    """Response model for worker statistics"""

    stats: Dict[str, Any] = Field(..., description="Worker statistics")
    ping: Dict[str, Any] = Field(..., description="Worker ping responses")
    registered: Dict[str, Any] = Field(..., description="Registered tasks by worker")


class TaskCancelResponse(BaseModel):
    """Response model for task cancellation"""

    task_id: str = Field(..., description="Cancelled task identifier")
    cancelled: bool = Field(..., description="Whether cancellation was successful")
    message: str = Field(
        default="Task cancelled successfully", description="Cancellation message"
    )

"""
Common Response Schemas

Pydantic schemas for common API responses including
pagination, success messages, and error responses.
"""

from typing import Any, Dict, List, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar('T')


class BaseResponse(BaseModel):
    """Base response schema"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    timestamp: str = Field(..., description="Response timestamp")


class SuccessResponse(BaseResponse):
    """Success response schema"""
    success: bool = Field(default=True, description="Success flag")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")


class ErrorResponse(BaseResponse):
    """Error response schema"""
    success: bool = Field(default=False, description="Success flag")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response schema"""
    validation_errors: List[Dict[str, Any]] = Field(..., description="Validation errors")


class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(..., description="Check timestamp")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component statuses")


class MetricsResponse(BaseModel):
    """Metrics response schema"""
    metrics: Dict[str, Any] = Field(..., description="System metrics")
    timestamp: str = Field(..., description="Metrics timestamp")


class FileUploadResponse(BaseModel):
    """File upload response schema"""
    file_id: str = Field(..., description="Uploaded file ID")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="File content type")
    url: Optional[str] = Field(None, description="File access URL")
    upload_timestamp: str = Field(..., description="Upload timestamp")


class BulkOperationResponse(BaseModel):
    """Bulk operation response schema"""
    total_requested: int = Field(..., description="Total items requested for operation")
    successful: int = Field(..., description="Number of successful operations")
    failed: int = Field(..., description="Number of failed operations")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="List of errors")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Operation results")


class AsyncTaskResponse(BaseModel):
    """Async task response schema"""
    task_id: str = Field(..., description="Background task ID")
    status: str = Field(..., description="Initial task status")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")
    tracking_url: Optional[str] = Field(None, description="Task tracking URL")


class StatusResponse(BaseModel):
    """Generic status response schema"""
    status: str = Field(..., description="Current status")
    details: Optional[Dict[str, Any]] = Field(None, description="Status details")
    last_updated: str = Field(..., description="Last status update timestamp")


class ConfigResponse(BaseModel):
    """Configuration response schema"""
    config: Dict[str, Any] = Field(..., description="Configuration data")
    environment: str = Field(..., description="Current environment")
    last_updated: str = Field(..., description="Configuration last update timestamp")


class StatsResponse(BaseModel):
    """Statistics response schema"""
    stats: Dict[str, Any] = Field(..., description="Statistics data")
    period: str = Field(..., description="Statistics period")
    generated_at: str = Field(..., description="Statistics generation timestamp")

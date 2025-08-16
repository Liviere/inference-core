"""
Data Transfer Objects (DTOs) for batch provider operations.

These classes define the structure for data exchange between 
the batch processing system and provider implementations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .enums import BatchMode, BatchStatus


class UsageInfo(BaseModel):
    """Token usage information for LLM requests"""
    
    prompt_tokens: Optional[int] = Field(None, description="Number of tokens in the prompt")
    completion_tokens: Optional[int] = Field(None, description="Number of tokens in the completion")
    total_tokens: Optional[int] = Field(None, description="Total number of tokens used")
    
    # Additional usage metrics that may be provided by some providers
    cache_creation_input_tokens: Optional[int] = Field(None, description="Tokens used for cache creation")
    cache_read_input_tokens: Optional[int] = Field(None, description="Tokens read from cache")


class PreparedSubmission(BaseModel):
    """Represents a batch submission prepared for a provider"""
    
    batch_id: UUID = Field(..., description="Internal batch job ID")
    provider_name: str = Field(..., description="Target provider name")
    model: str = Field(..., description="Model to use for processing")
    mode: BatchMode = Field(..., description="Processing mode")
    payloads: List[Dict[str, Any]] = Field(..., description="List of prepared request payloads")
    config: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific configuration")
    
    # TODO: Issue #002 - Support chunking for large batches
    # This may need to return multiple PreparedSubmission objects or add chunking metadata
    chunk_index: Optional[int] = Field(None, description="Chunk index for large batches (future use)")
    total_chunks: Optional[int] = Field(None, description="Total number of chunks for large batches (future use)")
    
    
class ProviderSubmitResult(BaseModel):
    """Result of submitting a batch to a provider"""
    
    provider_batch_id: str = Field(..., description="Provider's unique identifier for this batch")
    status: BatchStatus = Field(..., description="Initial normalized status")
    raw_status: str = Field(..., description="Original status from provider")
    submitted_at: datetime = Field(..., description="When the batch was submitted")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional provider metadata")


class ProviderStatus(BaseModel):
    """Status information from a provider about a batch job"""
    
    provider_batch_id: str = Field(..., description="Provider's unique identifier for this batch")
    status: BatchStatus = Field(..., description="Current normalized status")
    raw_status: str = Field(..., description="Original status from provider")
    progress: Optional[Dict[str, Any]] = Field(None, description="Progress information if available")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp if finished")
    result_uri: Optional[str] = Field(None, description="URI to download results if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional provider metadata")


class ProviderResultRow(BaseModel):
    """Individual result row from a batch operation"""
    
    request_id: str = Field(..., description="Identifier for the original request")
    status: str = Field(..., description="Status of this individual request")
    response: Optional[Dict[str, Any]] = Field(None, description="Response data if successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    usage: Optional[UsageInfo] = Field(None, description="Token usage information for this request")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
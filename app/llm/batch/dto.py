"""
Data Transfer Objects (DTOs) for batch provider operations.

These classes define the structure for data exchange between 
the batch processing system and provider implementations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class PreparedSubmission(BaseModel):
    """Represents a batch submission prepared for a provider"""
    
    batch_id: UUID = Field(..., description="Internal batch job ID")
    provider_name: str = Field(..., description="Target provider name")
    model: str = Field(..., description="Model to use for processing")
    mode: str = Field(..., description="Processing mode (chat, embedding, completion, custom)")
    payloads: List[Dict[str, Any]] = Field(..., description="List of prepared request payloads")
    config: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific configuration")
    
    
class ProviderSubmitResult(BaseModel):
    """Result of submitting a batch to a provider"""
    
    provider_batch_id: str = Field(..., description="Provider's unique identifier for this batch")
    status: str = Field(..., description="Initial status from provider")
    submitted_at: datetime = Field(..., description="When the batch was submitted")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional provider metadata")


class ProviderStatus(BaseModel):
    """Status information from a provider about a batch job"""
    
    provider_batch_id: str = Field(..., description="Provider's unique identifier for this batch")
    status: str = Field(..., description="Current status from provider")
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
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
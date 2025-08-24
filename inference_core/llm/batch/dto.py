"""
Data Transfer Objects for Batch Processing

Minimal typed DTOs for provider communication and result handling.
Kept simple and extensible per technical requirements.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class PreparedSubmission(BaseModel):
    """
    Prepared batch submission ready for provider submission.
    
    Contains pre-processed batch items and metadata needed for submission.
    """
    
    batch_job_id: UUID = Field(..., description="Internal batch job identifier")
    provider: str = Field(..., description="Target provider name")
    model: str = Field(..., description="Model to use for processing")
    mode: str = Field(..., description="Processing mode (chat, embedding, completion, custom)")
    items: List[Dict[str, Any]] = Field(..., description="Prepared batch items for submission")
    config: Optional[Dict[str, Any]] = Field(None, description="Provider-specific configuration")
    
    def get_item_count(self) -> int:
        """Get number of items in the submission."""
        return len(self.items)


class ProviderStatus(BaseModel):
    """
    Provider-specific batch status information.
    
    Maps provider status to internal status representation.
    """
    
    provider_batch_id: str = Field(..., description="Provider's batch identifier")
    status: str = Field(..., description="Provider-specific status")
    normalized_status: str = Field(..., description="Normalized internal status")
    progress_info: Optional[Dict[str, Any]] = Field(None, description="Additional progress details")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time if available")
    

class ProviderResultRow(BaseModel):
    """
    Single result row from provider batch processing.
    
    Represents one completed item with its output and metadata.
    """
    
    custom_id: str = Field(..., description="Custom identifier linking to BatchItem")
    output_text: Optional[str] = Field(None, description="Primary text output")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Structured output data")
    raw_metadata: Optional[Dict[str, Any]] = Field(None, description="Raw provider metadata")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    is_success: bool = Field(..., description="Whether processing was successful")
    

class ProviderSubmitResult(BaseModel):
    """
    Result of batch submission to provider.
    
    Contains provider batch ID and submission metadata.
    """
    
    provider_batch_id: str = Field(..., description="Provider's assigned batch identifier")
    status: str = Field(..., description="Initial provider status")
    submitted_at: datetime = Field(..., description="When submission was completed")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time if provided")
    submission_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional submission details")
    item_count: int = Field(..., description="Number of items submitted", ge=0)
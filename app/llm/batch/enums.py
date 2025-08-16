"""
Enums for batch processing system.

Defines standardized enumerations for batch status and processing modes 
to prevent string drift and ensure consistency across the system.
"""

from enum import Enum


class BatchMode(str, Enum):
    """
    Batch processing modes.
    
    Defines the type of operation to perform on the batch requests.
    """
    CHAT = "chat"
    COMPLETION = "completion" 
    EMBEDDING = "embedding"
    CUSTOM = "custom"


class BatchStatus(str, Enum):
    """
    Standardized batch status values.
    
    Internal representation of batch job status, normalized from provider-specific
    status values. Used for consistent status handling across different providers.
    """
    # Submission states
    VALIDATING = "validating"      # Provider is validating the batch
    QUEUED = "queued"             # Batch is queued for processing
    IN_PROGRESS = "in_progress"   # Batch is currently being processed
    FINALIZING = "finalizing"     # Provider is finalizing results
    
    # Terminal states
    COMPLETED = "completed"       # Batch completed successfully
    FAILED = "failed"            # Batch failed with errors
    CANCELLED = "cancelled"      # Batch was cancelled
    EXPIRED = "expired"          # Batch expired before completion
    
    # Special states
    UNKNOWN = "unknown"          # Status cannot be determined


# Provider-specific status mappings
# TODO: Issue #002 - Move these mappings to llm_config.yaml for dynamic configuration
PROVIDER_STATUS_MAPPINGS = {
    "openai": {
        "validating": BatchStatus.VALIDATING,
        "in_progress": BatchStatus.IN_PROGRESS,
        "finalizing": BatchStatus.FINALIZING,
        "completed": BatchStatus.COMPLETED,
        "failed": BatchStatus.FAILED,
        "expired": BatchStatus.EXPIRED,
        "cancelled": BatchStatus.CANCELLED,
    },
    # Add other provider mappings as needed
    # "gemini": {...},
    # "claude": {...},
}


def normalize_provider_status(provider_name: str, provider_status: str) -> BatchStatus:
    """
    Normalize provider-specific status to internal BatchStatus enum.
    
    Args:
        provider_name: Name of the provider (e.g., "openai")
        provider_status: Raw status from the provider
        
    Returns:
        Normalized BatchStatus enum value
    """
    mapping = PROVIDER_STATUS_MAPPINGS.get(provider_name, {})
    return mapping.get(provider_status.lower(), BatchStatus.UNKNOWN)
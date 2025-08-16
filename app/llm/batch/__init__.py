"""
Batch processing module for LLM providers.

This module provides the infrastructure for batch processing operations
across different LLM providers with a common interface.
"""

from .base_provider import BaseBatchProvider
from .dto import PreparedSubmission, ProviderResultRow, ProviderStatus, ProviderSubmitResult, UsageInfo
from .enums import BatchMode, BatchStatus, normalize_provider_status
from .exceptions import ProviderError, ProviderPermanentError, ProviderTransientError
from .registry import BatchProviderRegistry, ProviderNotRegisteredError, batch_provider_registry

__all__ = [
    # Base provider interface
    "BaseBatchProvider",
    
    # DTOs
    "PreparedSubmission",
    "ProviderSubmitResult", 
    "ProviderStatus",
    "ProviderResultRow",
    "UsageInfo",
    
    # Enums and utilities
    "BatchMode",
    "BatchStatus", 
    "normalize_provider_status",
    
    # Exceptions
    "ProviderError",
    "ProviderTransientError",
    "ProviderPermanentError",
    
    # Registry
    "BatchProviderRegistry",
    "ProviderNotRegisteredError",
    "batch_provider_registry",
]
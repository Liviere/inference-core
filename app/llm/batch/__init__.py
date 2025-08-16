"""
Batch processing module for LLM providers.

This module provides the infrastructure for batch processing operations
across different LLM providers with a common interface.
"""

from .base_provider import BaseBatchProvider
from .dto import PreparedSubmission, ProviderResultRow, ProviderStatus, ProviderSubmitResult
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
    
    # Exceptions
    "ProviderError",
    "ProviderTransientError",
    "ProviderPermanentError",
    
    # Registry
    "BatchProviderRegistry",
    "ProviderNotRegisteredError",
    "batch_provider_registry",
]
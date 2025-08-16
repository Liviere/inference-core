"""
Batch Processing Module

Provides provider-agnostic batch processing capabilities for LLM operations.
Includes base provider interface, registry, and supporting utilities.
"""

from .dto import PreparedSubmission, ProviderSubmitResult, ProviderStatus, ProviderResultRow
from .exceptions import ProviderTransientError, ProviderPermanentError, ProviderNotFoundError
from .registry import BatchProviderRegistry, registry
from .providers.base import BaseBatchProvider

__all__ = [
    "PreparedSubmission",
    "ProviderSubmitResult", 
    "ProviderStatus",
    "ProviderResultRow",
    "ProviderTransientError",
    "ProviderPermanentError",
    "ProviderNotFoundError",
    "BatchProviderRegistry",
    "BaseBatchProvider",
    "registry",
]
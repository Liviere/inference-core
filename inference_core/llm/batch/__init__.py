"""
Batch Processing Module

Provides provider-agnostic batch processing capabilities for LLM operations.
Includes base provider interface, registry, and supporting utilities.
"""

from .dto import (
    PreparedSubmission,
    ProviderResultRow,
    ProviderStatus,
    ProviderSubmitResult,
)
from .exceptions import (
    ProviderNotFoundError,
    ProviderPermanentError,
    ProviderTransientError,
)
from .providers.base import BaseBatchProvider
from .registry import BatchProviderRegistry, get_global_registry


# Import and register providers
def _register_providers():
    """Register all available batch providers."""
    registry = get_global_registry()
    try:
        from .providers.openai_provider import OpenAIBatchProvider

        registry.register(OpenAIBatchProvider)
    except ImportError:
        pass  # OpenAI provider optional if dependencies missing

    try:
        from .providers.gemini_provider import GeminiBatchProvider

        registry.register(GeminiBatchProvider)
    except ImportError:
        pass  # Gemini provider optional if dependencies missing

    try:
        from .providers.claude_provider import ClaudeBatchProvider

        registry.register(ClaudeBatchProvider)
    except ImportError:
        pass  # Claude provider optional if dependencies missing


# Register providers on module import
_register_providers()

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
    "get_global_registry",
]

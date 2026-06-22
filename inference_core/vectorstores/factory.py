"""
Vector Store Factory

Factory functions for creating vector store provider instances
based on configuration settings.
"""

import logging
import threading
from typing import Any, Dict, Optional

from ..core.config import get_settings
from .base import BaseVectorStoreProvider, InMemoryVectorStoreProvider

logger = logging.getLogger(__name__)


def create_vector_store_provider(
    backend: str, config: Optional[Dict[str, Any]] = None
) -> BaseVectorStoreProvider:
    """
    Create a vector store provider instance.

    Args:
        backend: Backend type ("qdrant", "memory")
        config: Optional configuration dict (uses settings if None)

    Returns:
        Vector store provider instance

    Raises:
        ValueError: If backend is not supported
        ImportError: If required dependencies are missing
    """
    if config is None:
        config = _get_config_from_settings()

    if backend == "memory":
        return InMemoryVectorStoreProvider(config)

    elif backend == "qdrant":
        try:
            from .qdrant_provider import QdrantProvider

            return QdrantProvider(config)
        except ImportError as e:
            raise ImportError(
                f"Qdrant dependencies not available: {e}. "
                "Install with: pip install qdrant-client sentence-transformers langchain-qdrant"
            ) from e

    else:
        raise ValueError(
            f"Unsupported vector store backend: {backend}. "
            f"Supported backends: memory, qdrant"
        )


# Process-wide provider cache (replaces functools.lru_cache so we can control
# exactly what gets memoized). A successfully created provider is cached for the
# process lifetime (singleton). The "disabled" decision is also cached — it is a
# stable, config-driven state. A *transient* construction failure is deliberately
# NOT cached, so the next call retries instead of pinning the process into a
# permanently-unavailable state (root cause of ASSISTANTS-3).
_provider_instance: Optional[BaseVectorStoreProvider] = None
_provider_disabled: bool = False
_provider_lock = threading.Lock()


def get_vector_store_provider() -> Optional[BaseVectorStoreProvider]:
    """
    Get the configured vector store provider instance.

    A real provider is cached as a process-wide singleton. If the vector store
    is disabled, ``None`` is cached (stable config state). If provider
    construction raises (e.g. a transient dependency/config hiccup), ``None`` is
    returned WITHOUT caching so a later call can recover.

    Returns:
        Vector store provider instance, or None if disabled / construction failed
    """
    global _provider_instance, _provider_disabled
    if _provider_instance is not None:
        return _provider_instance
    if _provider_disabled:
        return None

    with _provider_lock:
        # Re-check under the lock (another thread may have just built it).
        if _provider_instance is not None:
            return _provider_instance
        if _provider_disabled:
            return None

        settings = get_settings()
        if not settings.is_vector_store_enabled:
            logger.info("Vector store is disabled")
            _provider_disabled = True
            return None

        backend = settings.vector_backend
        config = _get_config_from_settings()

        try:
            provider = create_vector_store_provider(backend, config)
            logger.info(f"Created vector store provider: {backend}")
            _provider_instance = provider
            return provider
        except Exception as e:
            # Do NOT cache a transient failure — let the next call retry.
            logger.error(f"Failed to create vector store provider '{backend}': {e}")
            return None


def _get_config_from_settings() -> Dict[str, Any]:
    """Extract vector store configuration from application settings"""
    settings = get_settings()

    config = {
        "default_collection": settings.vector_collection_default,
        "dimension": settings.vector_dim,
        "distance": settings.vector_distance,
        "embedding_model": settings.vector_embedding_model,
        "max_batch_size": settings.vector_ingest_max_batch_size,
    }

    # Add backend-specific configuration
    if settings.vector_backend == "qdrant":
        config.update(
            {
                "url": settings.qdrant_url,
                "api_key": settings.qdrant_api_key,
            }
        )

    return config


def clear_provider_cache():
    """Clear the cached provider instance (useful for testing)"""
    global _provider_instance, _provider_disabled
    with _provider_lock:
        _provider_instance = None
        _provider_disabled = False
    logger.debug("Cleared vector store provider cache")

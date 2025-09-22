"""
Vector Store Factory

Factory functions for creating vector store provider instances
based on configuration settings.
"""

import logging
from functools import lru_cache
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


@lru_cache(maxsize=1)
def get_vector_store_provider() -> Optional[BaseVectorStoreProvider]:
    """
    Get the configured vector store provider instance.
    
    This function is cached to ensure singleton behavior.
    
    Returns:
        Vector store provider instance or None if disabled
    """
    settings = get_settings()
    
    if not settings.is_vector_store_enabled:
        logger.info("Vector store is disabled")
        return None
    
    backend = settings.vector_backend
    config = _get_config_from_settings()
    
    try:
        provider = create_vector_store_provider(backend, config)
        logger.info(f"Created vector store provider: {backend}")
        return provider
    except Exception as e:
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
        config.update({
            "url": settings.qdrant_url,
            "api_key": settings.qdrant_api_key,
        })
    
    return config


def clear_provider_cache():
    """Clear the cached provider instance (useful for testing)"""
    get_vector_store_provider.cache_clear()
    logger.debug("Cleared vector store provider cache")
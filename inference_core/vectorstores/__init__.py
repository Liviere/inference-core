"""
Vector Store Module

Provides abstractions and implementations for vector storage and retrieval.
Supports pluggable backends for different vector databases.
"""

from .base import BaseVectorStoreProvider
from .factory import create_vector_store_provider, get_vector_store_provider

__all__ = [
    "BaseVectorStoreProvider",
    "create_vector_store_provider",
    "get_vector_store_provider",
]
"""
Batch provider registry for dynamic provider lookup.

Manages registration and retrieval of batch provider implementations
using provider string keys.
"""

from typing import Dict, List, Type
import logging

from .base_provider import BaseBatchProvider
from .exceptions import ProviderPermanentError

logger = logging.getLogger(__name__)


class ProviderNotRegisteredError(ProviderPermanentError):
    """Raised when attempting to access a provider that is not registered"""
    pass


class BatchProviderRegistry:
    """
    Registry for batch provider implementations.
    
    Provides a centralized mechanism to register provider classes
    and retrieve them by name for dynamic instantiation.
    """
    
    def __init__(self):
        """Initialize an empty provider registry"""
        self._providers: Dict[str, Type[BaseBatchProvider]] = {}
    
    def register(self, provider_class: Type[BaseBatchProvider]) -> None:
        """
        Register a provider class in the registry.
        
        Args:
            provider_class: A class that inherits from BaseBatchProvider
            
        Raises:
            ValueError: If the provider class is invalid or missing PROVIDER_NAME
        """
        if not issubclass(provider_class, BaseBatchProvider):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseBatchProvider"
            )
        
        if not hasattr(provider_class, 'PROVIDER_NAME') or not provider_class.PROVIDER_NAME:
            raise ValueError(
                f"Provider class {provider_class.__name__} must define a PROVIDER_NAME constant"
            )
        
        provider_name = provider_class.PROVIDER_NAME
        
        if provider_name in self._providers:
            logger.warning(
                f"Overriding existing provider registration for '{provider_name}' "
                f"(was: {self._providers[provider_name].__name__}, "
                f"now: {provider_class.__name__})"
            )
        
        self._providers[provider_name] = provider_class
        logger.info(f"Registered batch provider: {provider_name} ({provider_class.__name__})")
    
    def get(self, provider_name: str) -> Type[BaseBatchProvider]:
        """
        Get a provider class by name.
        
        Args:
            provider_name: Name of the provider to retrieve
            
        Returns:
            The provider class for the given name
            
        Raises:
            ProviderNotRegisteredError: If the provider is not registered
        """
        if provider_name not in self._providers:
            available = list(self._providers.keys())
            raise ProviderNotRegisteredError(
                f"Provider '{provider_name}' is not registered. "
                f"Available providers: {available}",
                provider_name=provider_name
            )
        
        return self._providers[provider_name]
    
    def list(self) -> List[str]:
        """
        Get a list of all registered provider names.
        
        Returns:
            List of registered provider names
        """
        return list(self._providers.keys())
    
    def is_registered(self, provider_name: str) -> bool:
        """
        Check if a provider is registered.
        
        Args:
            provider_name: Name of the provider to check
            
        Returns:
            True if the provider is registered, False otherwise
        """
        return provider_name in self._providers
    
    def unregister(self, provider_name: str) -> bool:
        """
        Unregister a provider.
        
        Args:
            provider_name: Name of the provider to unregister
            
        Returns:
            True if the provider was unregistered, False if it wasn't registered
        """
        if provider_name in self._providers:
            del self._providers[provider_name]
            logger.info(f"Unregistered batch provider: {provider_name}")
            return True
        return False
    
    def clear(self) -> None:
        """Clear all registered providers"""
        self._providers.clear()
        logger.info("Cleared all registered batch providers")


# Global registry instance
batch_provider_registry = BatchProviderRegistry()
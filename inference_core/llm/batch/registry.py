"""
Batch Provider Registry

Simple registry for managing batch provider implementations.
Provides registration, lookup, and listing functionality.
"""

from typing import Dict, List, Type

from .exceptions import ProviderNotFoundError, ProviderRegistrationError
from .providers.base import BaseBatchProvider


class BatchProviderRegistry:
    """
    Registry for batch provider implementations.

    Provides a simple dictionary-based registry for provider lookup
    using provider string keys.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._providers: Dict[str, Type[BaseBatchProvider]] = {}

    def register(self, provider_class: Type[BaseBatchProvider]) -> None:
        """
        Register a provider implementation.

        Args:
            provider_class: Provider class that inherits from BaseBatchProvider

        Raises:
            ProviderRegistrationError: If provider class is invalid or name is missing
        """
        # Validate that it's a proper subclass
        if not issubclass(provider_class, BaseBatchProvider):
            raise ProviderRegistrationError(
                f"Provider class {provider_class.__name__} must inherit from BaseBatchProvider"
            )

        # Validate that PROVIDER_NAME is set
        if not provider_class.PROVIDER_NAME:
            raise ProviderRegistrationError(
                f"Provider class {provider_class.__name__} must set PROVIDER_NAME class attribute"
            )

        provider_name = provider_class.PROVIDER_NAME

        # Check for name conflicts
        if provider_name in self._providers:
            existing_class = self._providers[provider_name]
            raise ProviderRegistrationError(
                f"Provider '{provider_name}' is already registered with class {existing_class.__name__}"
            )

        self._providers[provider_name] = provider_class

    def get(self, provider_name: str) -> Type[BaseBatchProvider]:
        """
        Get a registered provider class by name.

        Args:
            provider_name: Name of the provider to retrieve

        Returns:
            Provider class for the given name

        Raises:
            ProviderNotFoundError: If provider is not registered
        """
        if provider_name not in self._providers:
            raise ProviderNotFoundError(provider_name)

        return self._providers[provider_name]

    def list(self) -> List[str]:
        """
        List all registered provider names.

        Returns:
            List of provider names currently registered
        """
        return list(self._providers.keys())

    def is_registered(self, provider_name: str) -> bool:
        """
        Check if a provider is registered.

        Args:
            provider_name: Name of the provider to check

        Returns:
            True if provider is registered, False otherwise
        """
        return provider_name in self._providers

    def create_provider(
        self, provider_name: str, config: dict = None
    ) -> BaseBatchProvider:
        """
        Create an instance of a registered provider.

        Args:
            provider_name: Name of the provider to create
            config: Configuration to pass to provider constructor

        Returns:
            Configured provider instance

        Raises:
            ProviderNotFoundError: If provider is not registered
        """
        provider_class = self.get(provider_name)
        return provider_class(config)

    def unregister(self, provider_name: str) -> bool:
        """
        Unregister a provider.

        Args:
            provider_name: Name of the provider to unregister

        Returns:
            True if provider was unregistered, False if it wasn't registered
        """
        if provider_name in self._providers:
            del self._providers[provider_name]
            return True
        return False

    def clear(self) -> None:
        """Clear all registered providers."""
        self._providers.clear()


# Global registry instance
registry = None


def get_global_registry() -> BatchProviderRegistry:
    """Get the global batch provider registry instance."""
    global registry
    if registry is None:
        registry = BatchProviderRegistry()
    return registry

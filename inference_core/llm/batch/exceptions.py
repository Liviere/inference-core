"""
Exception Classes for Batch Processing

Defines error types for provider interactions and batch operations.
Distinguishes between transient and permanent errors for retry logic.
"""

from typing import Optional


class BatchProviderError(Exception):
    """Base exception for batch provider errors."""
    
    def __init__(self, message: str, provider: Optional[str] = None, details: Optional[dict] = None):
        self.message = message
        self.provider = provider
        self.details = details or {}
        super().__init__(message)


class ProviderTransientError(BatchProviderError):
    """
    Transient error that may succeed on retry.
    
    Examples: rate limits, temporary service unavailability, network issues.
    Should trigger retry logic with exponential backoff.
    """
    
    def __init__(self, message: str, provider: Optional[str] = None, retry_after: Optional[int] = None, details: Optional[dict] = None):
        super().__init__(message, provider, details)
        self.retry_after = retry_after  # Suggested retry delay in seconds


class ProviderPermanentError(BatchProviderError):
    """
    Permanent error that will not succeed on retry.
    
    Examples: invalid API key, unsupported model, malformed request.
    Should not trigger retry logic.
    """
    
    def __init__(self, message: str, provider: Optional[str] = None, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, provider, details)
        self.error_code = error_code


class ProviderNotFoundError(BatchProviderError):
    """Raised when attempting to access an unregistered provider."""
    
    def __init__(self, provider_name: str):
        message = f"Provider '{provider_name}' is not registered"
        super().__init__(message, provider_name)
        self.provider_name = provider_name


class ProviderRegistrationError(BatchProviderError):
    """Raised when provider registration fails."""
    
    def __init__(self, message: str, provider_name: Optional[str] = None):
        super().__init__(message, provider_name)
        self.provider_name = provider_name
"""
Batch provider exception classes.

Defines specific exception types for batch provider operations
to enable proper error handling and retry logic.
"""


class ProviderError(Exception):
    """Base exception for provider-related errors"""
    
    def __init__(self, message: str, provider_name: str = None, provider_batch_id: str = None):
        super().__init__(message)
        self.provider_name = provider_name
        self.provider_batch_id = provider_batch_id


class ProviderTransientError(ProviderError):
    """
    Transient error that may be resolved by retrying the operation.
    
    Examples:
    - Rate limiting
    - Temporary service unavailability
    - Network timeouts
    - Temporary authentication issues
    """
    pass


class ProviderPermanentError(ProviderError):
    """
    Permanent error that will not be resolved by retrying.
    
    Examples:
    - Invalid API key
    - Unsupported model
    - Malformed request data
    - Quota exceeded
    """
    pass
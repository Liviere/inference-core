"""
Abstract Base Provider for Batch Processing

Defines the interface that all batch providers must implement.
Provides the strategy pattern for pluggable provider implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..dto import PreparedSubmission, ProviderSubmitResult, ProviderStatus, ProviderResultRow
from ..exceptions import ProviderTransientError, ProviderPermanentError


class BaseBatchProvider(ABC):
    """
    Abstract base class for batch providers.
    
    All provider implementations must inherit from this class and implement
    the required abstract methods for batch processing lifecycle.
    """
    
    # Must be set by each provider implementation
    PROVIDER_NAME: str = ""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the provider with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def supports_model(self, model: str, mode: str) -> bool:
        """
        Check if the provider supports a specific model and mode.
        
        Args:
            model: Model name to check
            mode: Processing mode (chat, embedding, completion, custom)
            
        Returns:
            True if the model and mode are supported, False otherwise
        """
        pass
    
    @abstractmethod
    def prepare_payloads(self, batch_items: List[dict], model: str, mode: str, config: Optional[dict] = None) -> PreparedSubmission:
        """
        Prepare batch items for submission to the provider.
        
        Converts internal batch item format to provider-specific format
        and creates a prepared submission ready for the submit() method.
        
        Args:
            batch_items: List of batch items with input_payload data
            model: Model name to use for processing
            mode: Processing mode (chat, embedding, completion, custom)
            config: Additional configuration for the batch
            
        Returns:
            PreparedSubmission with provider-formatted data
            
        Raises:
            ProviderPermanentError: If items cannot be prepared (e.g., invalid format)
        """
        pass
    
    @abstractmethod
    def submit(self, prepared_submission: PreparedSubmission) -> ProviderSubmitResult:
        """
        Submit a prepared batch to the provider.
        
        Args:
            prepared_submission: Prepared batch submission
            
        Returns:
            ProviderSubmitResult with provider batch ID and status
            
        Raises:
            ProviderTransientError: For retryable errors (rate limits, temporary issues)
            ProviderPermanentError: For permanent errors (auth, unsupported model)
        """
        pass
    
    @abstractmethod
    def poll_status(self, provider_batch_id: str) -> ProviderStatus:
        """
        Poll the status of a submitted batch.
        
        Args:
            provider_batch_id: Provider's batch identifier
            
        Returns:
            ProviderStatus with current status and progress information
            
        Raises:
            ProviderTransientError: For retryable errors (network issues, temporary unavailability)
            ProviderPermanentError: For permanent errors (batch not found, invalid ID)
        """
        pass
    
    @abstractmethod
    def fetch_results(self, provider_batch_id: str) -> List[ProviderResultRow]:
        """
        Fetch results from a completed batch.
        
        Args:
            provider_batch_id: Provider's batch identifier
            
        Returns:
            List of ProviderResultRow with outputs and metadata
            
        Raises:
            ProviderTransientError: For retryable errors (download issues, temporary unavailability)
            ProviderPermanentError: For permanent errors (batch not found, results expired)
        """
        pass
    
    @abstractmethod
    def cancel(self, provider_batch_id: str) -> bool:
        """
        Cancel a submitted batch if possible.
        
        Args:
            provider_batch_id: Provider's batch identifier
            
        Returns:
            True if cancellation was successful or already complete, False if cancellation failed
            
        Raises:
            ProviderTransientError: For retryable errors (network issues)
            ProviderPermanentError: For permanent errors (batch not found, cannot be cancelled)
        """
        pass
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return self.PROVIDER_NAME
    
    def validate_config(self, config: dict) -> bool:
        """
        Validate provider-specific configuration.
        
        Override in subclasses to implement provider-specific validation.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        return True
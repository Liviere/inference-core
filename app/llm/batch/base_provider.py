"""
Base batch provider interface.

Defines the abstract interface that all batch providers must implement
to ensure consistent behavior across different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from uuid import UUID

from .dto import PreparedSubmission, ProviderResultRow, ProviderStatus, ProviderSubmitResult
from .enums import BatchMode


class BaseBatchProvider(ABC):
    """
    Abstract base class for batch processing providers.
    
    All batch providers must inherit from this class and implement
    the required abstract methods to provide consistent batch processing
    capabilities across different LLM providers.
    """
    
    # Each provider implementation must define this constant
    PROVIDER_NAME: str = None
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the provider with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def supports_model(self, model: str) -> bool:
        """
        Check if the provider supports a specific model for batch processing.
        
        TODO: Issue #002 - Replace hard-coded model checks with YAML config lookup
        This method should read supported models from llm_config.yaml rather than
        using hard-coded lists in provider implementations.
        
        Args:
            model: Model name to check
            
        Returns:
            True if the model is supported for batch processing, False otherwise
        """
        pass
    
    @abstractmethod
    def prepare_payloads(
        self, 
        batch_id: UUID, 
        model: str, 
        mode: BatchMode, 
        requests: List[Dict[str, Any]],
        config: Dict[str, Any] = None
    ) -> PreparedSubmission:
        """
        Prepare request payloads for batch submission to the provider.
        
        TODO: Issue #002 - Support chunking for large batches
        This method may need to return multiple PreparedSubmission objects
        or the system may need a higher-level splitter to handle large batches
        that exceed provider limits.
        
        Args:
            batch_id: Internal batch job identifier
            model: Model name to use
            mode: Processing mode (BatchMode enum)
            requests: List of individual requests to process
            config: Additional configuration for the batch
            
        Returns:
            PreparedSubmission object ready for submission
        """
        pass
    
    @abstractmethod
    async def submit(self, prepared_submission: PreparedSubmission) -> ProviderSubmitResult:
        """
        Submit a prepared batch to the provider.
        
        Args:
            prepared_submission: PreparedSubmission object with all necessary data
            
        Returns:
            ProviderSubmitResult with provider batch ID and initial status
            
        Raises:
            ProviderTransientError: For temporary issues that can be retried
            ProviderPermanentError: For permanent issues that should not be retried
        """
        pass
    
    @abstractmethod
    async def poll_status(self, provider_batch_id: str) -> ProviderStatus:
        """
        Poll the status of a submitted batch job.
        
        Args:
            provider_batch_id: Provider's unique identifier for the batch
            
        Returns:
            ProviderStatus with current status and progress information
            
        Raises:
            ProviderTransientError: For temporary polling issues
            ProviderPermanentError: For permanent issues (e.g., batch not found)
        """
        pass
    
    @abstractmethod
    async def fetch_results(self, provider_batch_id: str) -> List[ProviderResultRow]:
        """
        Fetch the results of a completed batch job.
        
        Args:
            provider_batch_id: Provider's unique identifier for the batch
            
        Returns:
            List of ProviderResultRow objects with individual results
            
        Raises:
            ProviderTransientError: For temporary result fetching issues
            ProviderPermanentError: For permanent issues (e.g., results not available)
        """
        pass
    
    @abstractmethod
    async def cancel(self, provider_batch_id: str) -> bool:
        """
        Cancel a submitted batch job if possible.
        
        Args:
            provider_batch_id: Provider's unique identifier for the batch
            
        Returns:
            True if cancellation was successful or batch was already cancelled/completed,
            False if cancellation is not supported or failed
            
        Raises:
            ProviderTransientError: For temporary cancellation issues
            ProviderPermanentError: For permanent issues
        """
        pass
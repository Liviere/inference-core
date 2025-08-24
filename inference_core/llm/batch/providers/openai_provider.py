"""
OpenAI Batch Provider

Implementation of BaseBatchProvider for OpenAI's /v1/batches API.
Handles JSONL construction, file upload, batch submission, polling, and result retrieval.
"""

import json
import logging
from datetime import datetime, timedelta
from io import StringIO
from typing import List, Optional, Dict, Any
from uuid import uuid4

from openai import OpenAI
from openai.types import Batch, FileObject

from ..dto import PreparedSubmission, ProviderSubmitResult, ProviderStatus, ProviderResultRow
from ..exceptions import ProviderTransientError, ProviderPermanentError
from .base import BaseBatchProvider

logger = logging.getLogger(__name__)


class OpenAIBatchProvider(BaseBatchProvider):
    """
    OpenAI Batch Provider implementation.
    
    Provides batch processing capabilities using OpenAI's native /v1/batches API.
    Supports chat completion models with automatic JSONL formatting and file management.
    """
    
    PROVIDER_NAME = "openai"
    
    # OpenAI batch status mappings to internal status
    STATUS_MAPPING = {
        "validating": "submitted",
        "failed": "failed", 
        "in_progress": "in_progress",
        "finalizing": "in_progress",
        "completed": "completed",
        "expired": "failed",
        "cancelled": "cancelled"
    }
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize OpenAI batch provider.
        
        Args:
            config: Configuration dictionary containing API key and other settings
        """
        super().__init__(config)
        self.config = config or {}
        
        # Initialize OpenAI client
        api_key = self.config.get("api_key")
        if not api_key:
            raise ProviderPermanentError(
                "OpenAI API key is required",
                self.PROVIDER_NAME
            )
        
        self.client = OpenAI(api_key=api_key)
        logger.debug(f"Initialized OpenAI batch provider with config keys: {list(self.config.keys())}")
    
    def supports_model(self, model: str, mode: str) -> bool:
        """
        Check if the provider supports the given model and mode.
        
        Args:
            model: Model name to check
            mode: Processing mode (chat, completion, etc.)
            
        Returns:
            True if supported, False otherwise
        """
        # OpenAI batch API supports chat completion models
        supported_modes = ["chat"]
        
        # Common OpenAI model patterns
        openai_model_patterns = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4o",
            "gpt-5"
        ]
        
        if mode not in supported_modes:
            return False
        
        # Check if model matches OpenAI patterns
        return any(pattern in model.lower() for pattern in openai_model_patterns)
    
    def prepare_payloads(self, batch_items: List[dict], model: str, mode: str, config: Optional[dict] = None) -> PreparedSubmission:
        """
        Prepare batch items for OpenAI submission.
        
        Converts internal batch items to OpenAI JSONL format with custom_id mapping.
        
        Args:
            batch_items: List of batch items with input_payload data
            model: Model name to use for processing
            mode: Processing mode (chat, completion, etc.)
            config: Additional configuration for the batch
            
        Returns:
            PreparedSubmission with OpenAI-formatted JSONL data
            
        Raises:
            ProviderPermanentError: If items cannot be prepared
        """
        if not batch_items:
            raise ProviderPermanentError("Batch items cannot be empty", self.PROVIDER_NAME)
        
        if not self.supports_model(model, mode):
            raise ProviderPermanentError(
                f"Model {model} with mode {mode} is not supported by OpenAI provider",
                self.PROVIDER_NAME
            )
        
        formatted_items = []
        
        try:
            for item in batch_items:
                # Extract required fields
                batch_item_id = item.get("id")
                input_payload = item.get("input_payload", {})
                
                if not batch_item_id:
                    raise ProviderPermanentError("Batch item missing required 'id' field", self.PROVIDER_NAME)
                
                if not input_payload:
                    raise ProviderPermanentError("Batch item missing 'input_payload'", self.PROVIDER_NAME)
                
                # Parse input payload if it's a string
                if isinstance(input_payload, str):
                    try:
                        input_payload = json.loads(input_payload)
                    except json.JSONDecodeError as e:
                        raise ProviderPermanentError(f"Invalid JSON in input_payload: {e}", self.PROVIDER_NAME)
                
                # Construct OpenAI batch request format
                openai_request = {
                    "custom_id": str(batch_item_id),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        **input_payload
                    }
                }
                
                # Ensure messages are present for chat mode
                if mode == "chat" and "messages" not in openai_request["body"]:
                    raise ProviderPermanentError(
                        f"Chat mode requires 'messages' in input_payload for item {batch_item_id}",
                        self.PROVIDER_NAME
                    )
                
                formatted_items.append(openai_request)
                
        except ProviderPermanentError:
            raise
        except Exception as e:
            raise ProviderPermanentError(f"Failed to prepare payloads: {str(e)}", self.PROVIDER_NAME)
        
        logger.debug(f"Prepared {len(formatted_items)} items for OpenAI batch submission")
        
        return PreparedSubmission(
            batch_job_id=config.get("batch_job_id") if config and config.get("batch_job_id") else uuid4(),
            provider=self.PROVIDER_NAME,
            model=model,
            mode=mode,
            items=formatted_items,
            config=config
        )
    
    def submit(self, prepared_submission: PreparedSubmission) -> ProviderSubmitResult:
        """
        Submit a prepared batch to OpenAI.
        
        Args:
            prepared_submission: Prepared batch submission
            
        Returns:
            ProviderSubmitResult with OpenAI batch ID and status
            
        Raises:
            ProviderTransientError: For retryable errors (rate limits, temporary issues)
            ProviderPermanentError: For permanent errors (auth, unsupported model)
        """
        try:
            # Convert items to JSONL format
            jsonl_content = self._create_jsonl_content(prepared_submission.items)
            
            # Upload JSONL file to OpenAI
            file_obj = self._upload_batch_file(jsonl_content)
            
            # Create batch request
            batch_request = {
                "input_file_id": file_obj.id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h"
            }
            
            # Add optional metadata
            if prepared_submission.config:
                metadata = {
                    "batch_job_id": str(prepared_submission.batch_job_id),
                    "model": prepared_submission.model,
                    "mode": prepared_submission.mode
                }
                batch_request["metadata"] = metadata
            
            # Submit batch to OpenAI
            batch = self.client.batches.create(**batch_request)
            
            logger.info(f"Successfully submitted batch {batch.id} to OpenAI with {len(prepared_submission.items)} items")
            
            return ProviderSubmitResult(
                provider_batch_id=batch.id,
                status=batch.status,
                submitted_at=datetime.fromtimestamp(batch.created_at),
                estimated_completion=self._calculate_estimated_completion(batch),
                submission_metadata={
                    "input_file_id": file_obj.id,
                    "endpoint": batch.endpoint,
                    "completion_window": batch.completion_window
                },
                item_count=len(prepared_submission.items)
            )
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for transient errors (rate limits, temporary issues)
            if any(keyword in error_msg for keyword in ["rate limit", "timeout", "temporarily", "503", "502", "429"]):
                retry_after = self._extract_retry_after(str(e))
                raise ProviderTransientError(
                    f"OpenAI batch submission failed (transient): {str(e)}",
                    self.PROVIDER_NAME,
                    retry_after=retry_after
                )
            
            # Check for permanent errors
            if any(keyword in error_msg for keyword in ["unauthorized", "invalid", "not found", "401", "403", "404"]):
                raise ProviderPermanentError(
                    f"OpenAI batch submission failed (permanent): {str(e)}",
                    self.PROVIDER_NAME
                )
            
            # Default to permanent error for unknown errors
            raise ProviderPermanentError(
                f"OpenAI batch submission failed: {str(e)}",
                self.PROVIDER_NAME
            )
    
    def poll_status(self, provider_batch_id: str) -> ProviderStatus:
        """
        Poll the status of an OpenAI batch.
        
        Args:
            provider_batch_id: OpenAI batch identifier
            
        Returns:
            ProviderStatus with current status and progress information
            
        Raises:
            ProviderTransientError: For retryable errors (network issues, temporary unavailability)
            ProviderPermanentError: For permanent errors (batch not found, invalid ID)
        """
        try:
            batch = self.client.batches.retrieve(provider_batch_id)
            
            # Map OpenAI status to internal status
            normalized_status = self.STATUS_MAPPING.get(batch.status, batch.status)
            
            # Build progress information
            progress_info = {
                "total_requests": batch.request_counts.total if batch.request_counts else 0,
                "completed_requests": batch.request_counts.completed if batch.request_counts else 0,
                "failed_requests": batch.request_counts.failed if batch.request_counts else 0
            }
            
            # Add timing information
            if batch.completed_at:
                progress_info["completed_at"] = datetime.fromtimestamp(batch.completed_at)
            if batch.expires_at:
                progress_info["expires_at"] = datetime.fromtimestamp(batch.expires_at)
            
            # Add file information
            if batch.output_file_id:
                progress_info["output_file_id"] = batch.output_file_id
            if batch.error_file_id:
                progress_info["error_file_id"] = batch.error_file_id
            
            estimated_completion = None
            if batch.status == "in_progress" and batch.expires_at:
                estimated_completion = datetime.fromtimestamp(batch.expires_at)
            
            logger.debug(f"Polled batch {provider_batch_id}: status={batch.status}, normalized={normalized_status}")
            
            return ProviderStatus(
                provider_batch_id=provider_batch_id,
                status=batch.status,
                normalized_status=normalized_status,
                progress_info=progress_info,
                estimated_completion=estimated_completion
            )
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for transient errors
            if any(keyword in error_msg for keyword in ["timeout", "network", "503", "502", "temporarily"]):
                raise ProviderTransientError(
                    f"OpenAI batch status polling failed (transient): {str(e)}",
                    self.PROVIDER_NAME
                )
            
            # Check for permanent errors
            if any(keyword in error_msg for keyword in ["not found", "invalid", "401", "403", "404"]):
                raise ProviderPermanentError(
                    f"OpenAI batch {provider_batch_id} not found or invalid",
                    self.PROVIDER_NAME
                )
            
            # Default to transient error for status polling
            raise ProviderTransientError(
                f"OpenAI batch status polling failed: {str(e)}",
                self.PROVIDER_NAME
            )
    
    def fetch_results(self, provider_batch_id: str) -> List[ProviderResultRow]:
        """
        Fetch results from a completed OpenAI batch.
        
        Args:
            provider_batch_id: OpenAI batch identifier
            
        Returns:
            List of ProviderResultRow with results for each item
            
        Raises:
            ProviderTransientError: For retryable errors
            ProviderPermanentError: For permanent errors (batch not found, not completed)
        """
        try:
            # Get batch status first
            batch = self.client.batches.retrieve(provider_batch_id)
            
            if batch.status != "completed":
                raise ProviderPermanentError(
                    f"Cannot fetch results for batch {provider_batch_id} with status {batch.status}",
                    self.PROVIDER_NAME
                )
            
            results = []
            
            # Process output file if available
            if batch.output_file_id:
                output_content = self.client.files.content(batch.output_file_id).read()
                results.extend(self._parse_output_file(output_content))
            
            # Process error file if available
            if batch.error_file_id:
                error_content = self.client.files.content(batch.error_file_id).read()
                error_results = self._parse_error_file(error_content)
                results.extend(error_results)
            
            logger.info(f"Fetched {len(results)} results from OpenAI batch {provider_batch_id}")
            return results
            
        except ProviderPermanentError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for transient errors
            if any(keyword in error_msg for keyword in ["timeout", "network", "503", "502", "temporarily"]):
                raise ProviderTransientError(
                    f"OpenAI batch result fetching failed (transient): {str(e)}",
                    self.PROVIDER_NAME
                )
            
            # Default to permanent error for result fetching
            raise ProviderPermanentError(
                f"OpenAI batch result fetching failed: {str(e)}",
                self.PROVIDER_NAME
            )
    
    def cancel(self, provider_batch_id: str) -> bool:
        """
        Cancel an OpenAI batch.
        
        Args:
            provider_batch_id: OpenAI batch identifier
            
        Returns:
            True if cancellation was successful
            
        Raises:
            ProviderPermanentError: If batch cannot be cancelled or not found
        """
        try:
            batch = self.client.batches.cancel(provider_batch_id)
            
            if batch.status == "cancelled":
                logger.info(f"Successfully cancelled OpenAI batch {provider_batch_id}")
                return True
            else:
                logger.warning(f"OpenAI batch {provider_batch_id} cancellation returned status: {batch.status}")
                return False
                
        except Exception as e:
            error_msg = str(e).lower()
            
            if any(keyword in error_msg for keyword in ["not found", "invalid", "401", "403", "404"]):
                raise ProviderPermanentError(
                    f"OpenAI batch {provider_batch_id} not found or cannot be cancelled",
                    self.PROVIDER_NAME
                )
            
            raise ProviderPermanentError(
                f"OpenAI batch cancellation failed: {str(e)}",
                self.PROVIDER_NAME
            )
    
    def _create_jsonl_content(self, items: List[Dict[str, Any]]) -> str:
        """
        Convert items to JSONL format for OpenAI batch processing.
        
        Args:
            items: List of formatted OpenAI request items
            
        Returns:
            JSONL content as string
        """
        jsonl_lines = []
        for item in items:
            jsonl_lines.append(json.dumps(item, separators=(',', ':')))
        
        return '\n'.join(jsonl_lines)
    
    def _upload_batch_file(self, jsonl_content: str) -> FileObject:
        """
        Upload JSONL content to OpenAI as a batch input file.
        
        Args:
            jsonl_content: JSONL formatted content
            
        Returns:
            OpenAI FileObject
        """
        # Create file-like object from string content
        file_obj = StringIO(jsonl_content)
        file_obj.name = f"batch_input_{datetime.now().isoformat()}.jsonl"
        
        try:
            uploaded_file = self.client.files.create(
                file=(file_obj.name, file_obj.getvalue()),
                purpose="batch"
            )
            
            logger.debug(f"Uploaded batch file {uploaded_file.id} with {len(jsonl_content.splitlines())} lines")
            return uploaded_file
            
        except Exception as e:
            raise ProviderPermanentError(f"Failed to upload batch file: {str(e)}", self.PROVIDER_NAME)
    
    def _parse_output_file(self, content: bytes) -> List[ProviderResultRow]:
        """
        Parse OpenAI batch output file content.
        
        Args:
            content: Raw file content from OpenAI
            
        Returns:
            List of successful ProviderResultRow objects
        """
        results = []
        content_str = content.decode('utf-8')
        
        for line in content_str.strip().split('\n'):
            if not line.strip():
                continue
                
            try:
                response_data = json.loads(line)
                custom_id = response_data.get("custom_id")
                response_body = response_data.get("response", {}).get("body", {})
                
                # Extract content from OpenAI response
                output_text = None
                output_data = response_body
                
                if "choices" in response_body and response_body["choices"]:
                    choice = response_body["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        output_text = choice["message"]["content"]
                
                results.append(ProviderResultRow(
                    custom_id=custom_id,
                    output_text=output_text,
                    output_data=output_data,
                    raw_metadata=response_data,
                    error_message=None,
                    is_success=True
                ))
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse output line: {line[:100]}... Error: {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to process output line: {line[:100]}... Error: {e}")
                continue
        
        return results
    
    def _parse_error_file(self, content: bytes) -> List[ProviderResultRow]:
        """
        Parse OpenAI batch error file content.
        
        Args:
            content: Raw error file content from OpenAI
            
        Returns:
            List of failed ProviderResultRow objects
        """
        results = []
        content_str = content.decode('utf-8')
        
        for line in content_str.strip().split('\n'):
            if not line.strip():
                continue
                
            try:
                error_data = json.loads(line)
                custom_id = error_data.get("custom_id")
                error_info = error_data.get("error", {})
                
                error_message = error_info.get("message", "Unknown error")
                error_code = error_info.get("code")
                
                if error_code:
                    error_message = f"[{error_code}] {error_message}"
                
                results.append(ProviderResultRow(
                    custom_id=custom_id,
                    output_text=None,
                    output_data=None,
                    raw_metadata=error_data,
                    error_message=error_message,
                    is_success=False
                ))
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse error line: {line[:100]}... Error: {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to process error line: {line[:100]}... Error: {e}")
                continue
        
        return results
    
    def _calculate_estimated_completion(self, batch: Batch) -> Optional[datetime]:
        """
        Calculate estimated completion time for a batch.
        
        Args:
            batch: OpenAI Batch object
            
        Returns:
            Estimated completion datetime or None
        """
        if batch.completion_window == "24h":
            return datetime.fromtimestamp(batch.created_at) + timedelta(hours=24)
        return None
    
    def _extract_retry_after(self, error_message: str) -> Optional[int]:
        """
        Extract retry-after value from error message.
        
        Args:
            error_message: Error message from OpenAI
            
        Returns:
            Retry after seconds or None
        """
        # Look for common retry-after patterns
        import re
        
        # Pattern: "retry after 60 seconds"
        match = re.search(r'retry after (\d+) seconds?', error_message.lower())
        if match:
            return int(match.group(1))
        
        # Pattern: "try again in 30s"
        match = re.search(r'try again in (\d+)s', error_message.lower())
        if match:
            return int(match.group(1))
        
        # Default retry time for rate limits
        if "rate limit" in error_message.lower():
            return 60
        
        return None
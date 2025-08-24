"""
Anthropic Claude Batch Provider

Implementation of BaseBatchProvider for Anthropic Claude's message batch API.
Handles message batch construction, submission, polling, and result retrieval.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import uuid4

from anthropic import Anthropic

from ..dto import (
    PreparedSubmission,
    ProviderResultRow,
    ProviderStatus,
    ProviderSubmitResult,
)
from ..exceptions import ProviderPermanentError, ProviderTransientError
from .base import BaseBatchProvider

logger = logging.getLogger(__name__)


class ClaudeBatchProvider(BaseBatchProvider):
    """
    Anthropic Claude Batch Provider implementation.

    Provides batch processing capabilities using Anthropic Claude's message batch API.
    Supports Claude models with automatic message batch formatting.
    """

    PROVIDER_NAME = "claude"

    # Claude batch status mappings to internal status
    STATUS_MAPPING = {
        "in_progress": "in_progress",
        "canceling": "cancelled",
        "ended": "completed",
        "errored": "failed",
        "expired": "failed",
    }

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize Claude batch provider.

        Args:
            config: Configuration dictionary containing API key and other settings
        """
        super().__init__(config)
        self.config = config or {}

        # Initialize Anthropic client
        api_key = self.config.get("api_key")
        if not api_key:
            raise ProviderPermanentError(
                "Anthropic API key is required", self.PROVIDER_NAME
            )

        try:
            self.client = Anthropic(api_key=api_key)
            logger.debug(
                f"Initialized Claude batch provider with config keys: {list(self.config.keys())}"
            )
        except Exception as e:
            raise ProviderPermanentError(
                f"Failed to initialize Claude client: {str(e)}", self.PROVIDER_NAME
            )

    def supports_model(self, model: str, mode: str) -> bool:
        """
        Check if the provider supports the given model and mode.

        Args:
            model: Model name to check
            mode: Processing mode (chat, completion, etc.)

        Returns:
            True if supported, False otherwise
        """
        # Claude batch API supports chat mode primarily
        supported_modes = ["chat"]

        # Common Claude model patterns
        claude_model_patterns = [
            "claude-3",
            "claude-3.5",
            "claude-4",
            "claude-sonnet",
            "claude-opus",
            "claude-haiku",
        ]

        if mode not in supported_modes:
            return False

        # Check if model matches Claude patterns
        model_lower = model.lower()
        return any(pattern in model_lower for pattern in claude_model_patterns)

    def prepare_payloads(
        self,
        batch_items: List[dict],
        model: str,
        mode: str,
        config: Optional[dict] = None,
    ) -> PreparedSubmission:
        """
        Prepare batch items for Claude submission.

        Converts internal batch items to Claude message batch format.

        Args:
            batch_items: List of batch items with input_payload data
            model: Model name to use for processing
            mode: Processing mode (chat, completion, etc.)
            config: Additional configuration for the batch

        Returns:
            PreparedSubmission with Claude-formatted data

        Raises:
            ProviderPermanentError: If items cannot be prepared
        """
        if not batch_items:
            raise ProviderPermanentError(
                "Batch items cannot be empty", self.PROVIDER_NAME
            )

        if not self.supports_model(model, mode):
            raise ProviderPermanentError(
                f"Model {model} with mode {mode} is not supported by Claude provider",
                self.PROVIDER_NAME,
            )

        formatted_items = []

        try:
            for item in batch_items:
                # Extract required fields
                batch_item_id = item.get("id")
                input_payload = item.get("input_payload", {})

                if not batch_item_id:
                    raise ProviderPermanentError(
                        "Batch item missing required 'id' field", self.PROVIDER_NAME
                    )

                if not input_payload:
                    raise ProviderPermanentError(
                        "Batch item missing 'input_payload'", self.PROVIDER_NAME
                    )

                # Parse input payload if it's a string
                if isinstance(input_payload, str):
                    try:
                        input_payload = json.loads(input_payload)
                    except json.JSONDecodeError as e:
                        raise ProviderPermanentError(
                            f"Invalid JSON in input_payload: {e}", self.PROVIDER_NAME
                        )

                # Convert to Claude batch format
                claude_request = self._convert_to_claude_format(
                    batch_item_id, input_payload, model, mode
                )
                formatted_items.append(claude_request)

        except ProviderPermanentError:
            raise
        except Exception as e:
            raise ProviderPermanentError(
                f"Failed to prepare payloads: {str(e)}", self.PROVIDER_NAME
            )

        logger.debug(
            f"Prepared {len(formatted_items)} items for Claude batch submission"
        )

        return PreparedSubmission(
            batch_job_id=(
                config.get("batch_job_id")
                if config and config.get("batch_job_id")
                else uuid4()
            ),
            provider=self.PROVIDER_NAME,
            model=model,
            mode=mode,
            items=formatted_items,
            config=config,
        )

    def _convert_to_claude_format(
        self, custom_id: str, input_payload: dict, model: str, mode: str
    ) -> dict:
        """
        Convert input payload to Claude message batch request format.

        Args:
            custom_id: Custom identifier for the request
            input_payload: Input data for the request
            model: Model name
            mode: Processing mode

        Returns:
            Claude-formatted request
        """
        # Create Claude batch request format based on documentation
        claude_request = {
            "custom_id": custom_id,
            "params": {
                "model": model,
                "max_tokens": input_payload.get("max_tokens", 1024),
                "messages": [],
            },
        }

        # Handle different input formats
        if "messages" in input_payload:
            # Use messages directly if provided
            messages = input_payload["messages"]
            for message in messages:
                claude_message = {
                    "role": message.get("role", "user"),
                    "content": message.get("content", ""),
                }
                claude_request["params"]["messages"].append(claude_message)

        elif "content" in input_payload:
            # Direct content as user message
            content = input_payload["content"]
            claude_request["params"]["messages"].append(
                {"role": "user", "content": content}
            )

        elif "text" in input_payload:
            # Text field as user message
            text = input_payload["text"]
            claude_request["params"]["messages"].append(
                {"role": "user", "content": text}
            )

        else:
            raise ProviderPermanentError(
                f"Unsupported input format for Claude. Expected 'messages', 'content', or 'text' field",
                self.PROVIDER_NAME,
            )

        # Add optional parameters
        if "temperature" in input_payload:
            claude_request["params"]["temperature"] = input_payload["temperature"]

        if "top_p" in input_payload:
            claude_request["params"]["top_p"] = input_payload["top_p"]

        if "top_k" in input_payload:
            claude_request["params"]["top_k"] = input_payload["top_k"]

        if "system" in input_payload:
            claude_request["params"]["system"] = input_payload["system"]

        # Override max_tokens if provided in input
        if "max_tokens" in input_payload:
            claude_request["params"]["max_tokens"] = input_payload["max_tokens"]

        return claude_request

    def submit(self, prepared_submission: PreparedSubmission) -> ProviderSubmitResult:
        """
        Submit a prepared batch to Claude.

        Args:
            prepared_submission: Prepared batch submission

        Returns:
            ProviderSubmitResult with Claude batch ID and status

        Raises:
            ProviderTransientError: For retryable errors (rate limits, temporary issues)
            ProviderPermanentError: For permanent errors (auth, unsupported model)
        """
        try:
            # Create message batch using Claude API
            batch = self.client.messages.batches.create(
                requests=prepared_submission.items
            )

            logger.info(
                f"Successfully submitted batch {batch.id} to Claude with {len(prepared_submission.items)} items"
            )

            # Build submission metadata
            submission_metadata = {
                "model": prepared_submission.model,
                "mode": prepared_submission.mode,
                "request_counts": {"total": len(prepared_submission.items)},
            }

            return ProviderSubmitResult(
                provider_batch_id=batch.id,
                status=batch.processing_status,
                submitted_at=batch.created_at,
                estimated_completion=self._calculate_estimated_completion(batch),
                submission_metadata=submission_metadata,
                item_count=len(prepared_submission.items),
            )

        except Exception as e:
            error_msg = str(e).lower()

            # Check for transient errors (rate limits, temporary issues)
            if any(
                keyword in error_msg
                for keyword in [
                    "rate limit",
                    "timeout",
                    "temporarily",
                    "503",
                    "502",
                    "429",
                ]
            ):
                retry_after = self._extract_retry_after(str(e))
                raise ProviderTransientError(
                    f"Claude batch submission failed (transient): {str(e)}",
                    self.PROVIDER_NAME,
                    retry_after=retry_after,
                )

            # Check for permanent errors
            if any(
                keyword in error_msg
                for keyword in [
                    "unauthorized",
                    "invalid",
                    "not found",
                    "401",
                    "403",
                    "404",
                ]
            ):
                raise ProviderPermanentError(
                    f"Claude batch submission failed (permanent): {str(e)}",
                    self.PROVIDER_NAME,
                )

            # Default to permanent error for unknown errors
            raise ProviderPermanentError(
                f"Claude batch submission failed: {str(e)}", self.PROVIDER_NAME
            )

    def poll_status(self, provider_batch_id: str) -> ProviderStatus:
        """
        Poll the status of a Claude message batch.

        Args:
            provider_batch_id: Claude batch identifier

        Returns:
            ProviderStatus with current status and progress information

        Raises:
            ProviderTransientError: For retryable errors (network issues, temporary unavailability)
            ProviderPermanentError: For permanent errors (batch not found, invalid ID)
        """
        try:
            # Get batch info by listing and filtering (Claude doesn't have a direct get method in some versions)
            # Try to get batch details - the API may vary by version
            batch = None
            try:
                # Some versions may have a get method
                if hasattr(self.client.messages.batches, "retrieve"):
                    batch = self.client.messages.batches.retrieve(provider_batch_id)
                else:
                    # Fallback to listing and finding the batch
                    batches_page = self.client.messages.batches.list(limit=100)
                    for b in batches_page.data:
                        if b.id == provider_batch_id:
                            batch = b
                            break
            except AttributeError:
                # Handle API variations - list and find
                batches_page = self.client.messages.batches.list(limit=100)
                for b in batches_page.data:
                    if b.id == provider_batch_id:
                        batch = b
                        break

            if not batch:
                raise ProviderPermanentError(
                    f"Claude batch {provider_batch_id} not found", self.PROVIDER_NAME
                )

            # Map Claude status to internal status
            normalized_status = self.STATUS_MAPPING.get(
                batch.processing_status, batch.processing_status
            )

            # Build progress information
            progress_info = {
                "processing_status": batch.processing_status,
                "created_at": (
                    batch.created_at.isoformat() if batch.created_at else None
                ),
                "expires_at": (
                    batch.expires_at.isoformat() if batch.expires_at else None
                ),
            }

            # Add request counts if available
            if hasattr(batch, "request_counts"):
                progress_info["request_counts"] = {
                    "processing": getattr(batch.request_counts, "processing", 0),
                    "succeeded": getattr(batch.request_counts, "succeeded", 0),
                    "errored": getattr(batch.request_counts, "errored", 0),
                    "canceled": getattr(batch.request_counts, "canceled", 0),
                    "expired": getattr(batch.request_counts, "expired", 0),
                }

            estimated_completion = None
            if batch.processing_status == "in_progress":
                # Estimate completion based on expires_at if available
                if batch.expires_at:
                    estimated_completion = batch.expires_at
                elif batch.created_at:
                    # Default to 24h from creation
                    estimated_completion = batch.created_at + timedelta(hours=24)

            logger.debug(
                f"Polled batch {provider_batch_id}: status={batch.processing_status}, normalized={normalized_status}"
            )

            return ProviderStatus(
                provider_batch_id=provider_batch_id,
                status=batch.processing_status,
                normalized_status=normalized_status,
                progress_info=progress_info,
                estimated_completion=estimated_completion,
            )

        except Exception as e:
            error_msg = str(e).lower()

            # Check for transient errors
            if any(
                keyword in error_msg
                for keyword in ["timeout", "network", "503", "502", "temporarily"]
            ):
                raise ProviderTransientError(
                    f"Claude batch status polling failed (transient): {str(e)}",
                    self.PROVIDER_NAME,
                )

            # Check for permanent errors
            if any(
                keyword in error_msg
                for keyword in ["not found", "invalid", "401", "403", "404"]
            ):
                raise ProviderPermanentError(
                    f"Claude batch {provider_batch_id} not found or invalid",
                    self.PROVIDER_NAME,
                )

            # Default to transient error for status polling
            raise ProviderTransientError(
                f"Claude batch status polling failed: {str(e)}", self.PROVIDER_NAME
            )

    def fetch_results(self, provider_batch_id: str) -> List[ProviderResultRow]:
        """
        Fetch results from a completed Claude message batch.

        Args:
            provider_batch_id: Claude batch identifier

        Returns:
            List of ProviderResultRow with results for each item

        Raises:
            ProviderTransientError: For retryable errors
            ProviderPermanentError: For permanent errors (batch not found, not completed)
        """
        try:
            # First check if batch is completed
            status = self.poll_status(provider_batch_id)

            if status.normalized_status != "completed":
                raise ProviderPermanentError(
                    f"Cannot fetch results for batch {provider_batch_id} with status {status.status}",
                    self.PROVIDER_NAME,
                )

            results = []

            # Helper declared once (was previously mis-indented inside dict)
            def _json_safe(obj, depth=0, max_depth=6):
                if depth > max_depth:
                    return str(obj)
                if obj is None or isinstance(obj, (str, int, float, bool)):
                    return obj
                if isinstance(obj, dict):
                    return {str(k): _json_safe(v, depth + 1) for k, v in obj.items()}
                if isinstance(obj, (list, tuple, set)):
                    return [_json_safe(v, depth + 1) for v in obj]
                if hasattr(obj, "model_dump"):
                    try:
                        return _json_safe(obj.model_dump(), depth + 1)
                    except Exception:
                        return str(obj)
                if hasattr(obj, "__dict__"):
                    try:
                        raw = {
                            k: v
                            for k, v in obj.__dict__.items()
                            if not k.startswith("_") and not callable(v)
                        }
                        return _json_safe(raw, depth + 1)
                    except Exception:
                        return str(obj)
                return str(obj)

            result_stream = self.client.messages.batches.results(provider_batch_id)
            for entry in result_stream:
                try:
                    custom_id = entry.custom_id
                    output_text = None
                    output_data = None
                    error_message = None
                    is_success = False

                    result_type = getattr(entry.result, "type", None)
                    if result_type == "succeeded":
                        is_success = True
                        message = entry.result.message
                        # Extract first text block
                        if getattr(message, "content", None):
                            for block in message.content:
                                txt = getattr(block, "text", None)
                                if txt:
                                    output_text = txt
                                    break
                        output_data = {
                            "id": getattr(message, "id", None),
                            "type": getattr(message, "type", None),
                            "role": getattr(message, "role", None),
                            "model": getattr(message, "model", None),
                            "content": _json_safe(getattr(message, "content", None)),
                            "stop_reason": getattr(message, "stop_reason", None),
                            "stop_sequence": getattr(message, "stop_sequence", None),
                            "usage": _json_safe(getattr(message, "usage", None)),
                        }
                    elif result_type == "errored":
                        err = getattr(entry.result, "error", None)
                        if err:
                            error_message = f"[{getattr(err, 'type', 'error')}] {getattr(err, 'message', str(err))}"
                        else:
                            error_message = "Unknown error"
                    else:
                        error_message = f"Unknown result type: {result_type}"

                    results.append(
                        ProviderResultRow(
                            custom_id=custom_id,
                            output_text=output_text,
                            output_data=(
                                _json_safe(output_data) if output_data else None
                            ),
                            raw_metadata=_json_safe(entry.__dict__),
                            error_message=error_message,
                            is_success=is_success,
                        )
                    )
                except Exception as e:  # pragma: no cover (defensive)
                    logger.error(f"Failed to parse Claude result entry: {e}")
                    results.append(
                        ProviderResultRow(
                            custom_id=getattr(entry, "custom_id", "unknown"),
                            output_text=None,
                            output_data=None,
                            raw_metadata=getattr(entry, "__dict__", {}),
                            error_message=f"Failed to parse result: {str(e)}",
                            is_success=False,
                        )
                    )

            logger.info(
                f"Fetched {len(results)} results from Claude batch {provider_batch_id}"
            )
            return results

        except ProviderPermanentError:
            raise
        except Exception as e:
            error_msg = str(e).lower()

            # Check for transient errors
            if any(
                keyword in error_msg
                for keyword in ["timeout", "network", "503", "502", "temporarily"]
            ):
                raise ProviderTransientError(
                    f"Claude batch result fetching failed (transient): {str(e)}",
                    self.PROVIDER_NAME,
                )

            # Default to permanent error for result fetching
            raise ProviderPermanentError(
                f"Claude batch result fetching failed: {str(e)}", self.PROVIDER_NAME
            )

    def cancel(self, provider_batch_id: str) -> bool:
        """
        Cancel a Claude message batch.

        Args:
            provider_batch_id: Claude batch identifier

        Returns:
            True if cancellation was successful

        Raises:
            ProviderPermanentError: If batch cannot be cancelled or not found
        """
        try:
            # Claude may have a cancel method on the batches API
            if hasattr(self.client.messages.batches, "cancel"):
                result = self.client.messages.batches.cancel(provider_batch_id)
                logger.info(f"Successfully cancelled Claude batch {provider_batch_id}")
                return True
            else:
                # If no cancel method available, return False
                logger.warning(
                    f"Claude batch cancellation not supported by current API version"
                )
                return False

        except Exception as e:
            error_msg = str(e).lower()

            if any(
                keyword in error_msg
                for keyword in ["not found", "invalid", "401", "403", "404"]
            ):
                raise ProviderPermanentError(
                    f"Claude batch {provider_batch_id} not found or cannot be cancelled",
                    self.PROVIDER_NAME,
                )

            raise ProviderPermanentError(
                f"Claude batch cancellation failed: {str(e)}", self.PROVIDER_NAME
            )

    def _calculate_estimated_completion(self, batch) -> Optional[datetime]:
        """
        Calculate estimated completion time for a batch.

        Args:
            batch: Claude batch object

        Returns:
            Estimated completion datetime or None
        """
        # Use expires_at if available, otherwise estimate 24 hours
        if hasattr(batch, "expires_at") and batch.expires_at:
            return batch.expires_at
        elif hasattr(batch, "created_at") and batch.created_at:
            return batch.created_at + timedelta(hours=24)
        return None

    def _extract_retry_after(self, error_message: str) -> Optional[int]:
        """
        Extract retry-after value from error message.

        Args:
            error_message: Error message from Claude

        Returns:
            Retry after seconds or None
        """
        # Look for common retry-after patterns
        import re

        # Pattern: "retry after 60 seconds"
        match = re.search(r"retry after (\d+) seconds?", error_message.lower())
        if match:
            return int(match.group(1))

        # Pattern: "try again in 30s"
        match = re.search(r"try again in (\d+)s", error_message.lower())
        if match:
            return int(match.group(1))

        # Default retry time for rate limits
        if "rate limit" in error_message.lower():
            return 60

        return None

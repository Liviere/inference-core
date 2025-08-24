"""
Google Gemini Batch Provider

Implementation of BaseBatchProvider for Google Gemini's native batch mode API.
Handles inlined request construction, job submission, polling, and result retrieval.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import uuid4

from google import genai

from inference_core.database.sql.models.batch import BatchJobStatus

from ..dto import (
    PreparedSubmission,
    ProviderResultRow,
    ProviderStatus,
    ProviderSubmitResult,
)
from ..exceptions import ProviderPermanentError, ProviderTransientError
from .base import BaseBatchProvider

logger = logging.getLogger(__name__)


class GeminiBatchProvider(BaseBatchProvider):
    """
    Google Gemini Batch Provider implementation.

    Provides batch processing capabilities using Google Gemini's native batch mode API.
    Supports chat models with automatic inlined request formatting.
    """

    PROVIDER_NAME = "gemini"

    # Gemini batch job status mappings to internal status
    STATUS_MAPPING = {
        "BATCH_STATE_UNSPECIFIED": BatchJobStatus.SUBMITTED,
        "BATCH_STATE_PENDING": BatchJobStatus.SUBMITTED,
        "BATCH_STATE_RUNNING": BatchJobStatus.IN_PROGRESS,
        "BATCH_STATE_SUCCEEDED": BatchJobStatus.COMPLETED,
        "BATCH_STATE_FAILED": BatchJobStatus.FAILED,
        "BATCH_STATE_CANCELLED": BatchJobStatus.CANCELLED,
        "JOB_STATE_UNSPECIFIED": BatchJobStatus.SUBMITTED,
        "JOB_STATE_QUEUED": BatchJobStatus.SUBMITTED,
        "JOB_STATE_PENDING": BatchJobStatus.SUBMITTED,
        "JOB_STATE_RUNNING": BatchJobStatus.IN_PROGRESS,
        "JOB_STATE_SUCCEEDED": BatchJobStatus.COMPLETED,
        "JOB_STATE_FAILED": BatchJobStatus.FAILED,
        "JOB_STATE_CANCELLED": BatchJobStatus.CANCELLED,
        "JOB_STATE_CANCELLING": BatchJobStatus.CANCELLED,
        "JOB_STATE_PAUSED": BatchJobStatus.FAILED,
        "JOB_STATE_UPDATING": BatchJobStatus.IN_PROGRESS,
        "JOB_STATE_PARTIALLY_SUCCEEDED": BatchJobStatus.COMPLETED,
        "JOB_STATE_EXPIRED": BatchJobStatus.EXPIRED,
    }

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize Gemini batch provider.

        Args:
            config: Configuration dictionary containing API key and other settings
        """
        super().__init__(config)
        self.config = config or {}

        # Initialize Google GenAI client
        api_key = self.config.get("api_key")
        if not api_key:
            raise ProviderPermanentError(
                "Google GenAI API key is required", self.PROVIDER_NAME
            )

        try:
            self.client = genai.Client(api_key=api_key)
            logger.debug(
                f"Initialized Gemini batch provider with config keys: {list(self.config.keys())}"
            )
        except Exception as e:
            raise ProviderPermanentError(
                f"Failed to initialize Gemini client: {str(e)}", self.PROVIDER_NAME
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
        # Source: Gemini Batch Mode docs (2025-08-04) – examples show
        # usage of both "gemini-2.5-flash" and prefix variant "models/gemini-2.5-flash".
        # We mirror OpenAI provider style: static pattern list substring match.
        # NOTE: If future versions appear (e.g. 3.x), extend list or refactor to regex.

        supported_modes = ["chat"]  # Gemini batch supports chat-style requests
        if mode not in supported_modes:
            return False

        model_lower = model.lower().strip()
        # Accept and normalize optional "models/" prefix returned/used in some SDK examples
        if model_lower.startswith("models/"):
            model_lower = model_lower[len("models/") :]

        gemini_model_patterns = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]

        return any(p in model_lower for p in gemini_model_patterns)

    def prepare_payloads(
        self,
        batch_items: List[dict],
        model: str,
        mode: str,
        config: Optional[dict] = None,
    ) -> PreparedSubmission:
        """
        Prepare batch items for Gemini submission.

        Converts internal batch items to Gemini inlined request format.

        Args:
            batch_items: List of batch items with input_payload data
            model: Model name to use for processing
            mode: Processing mode (chat, completion, etc.)
            config: Additional configuration for the batch

        Returns:
            PreparedSubmission with Gemini-formatted data

        Raises:
            ProviderPermanentError: If items cannot be prepared
        """
        if not batch_items:
            raise ProviderPermanentError(
                "Batch items cannot be empty", self.PROVIDER_NAME
            )

        if not self.supports_model(model, mode):
            raise ProviderPermanentError(
                f"Model {model} with mode {mode} is not supported by Gemini provider",
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

                # Convert to Gemini batch format
                gemini_request = self._convert_to_gemini_format(
                    batch_item_id, input_payload, model, mode
                )
                formatted_items.append(gemini_request)

        except ProviderPermanentError:
            raise
        except Exception as e:
            raise ProviderPermanentError(
                f"Failed to prepare payloads: {str(e)}", self.PROVIDER_NAME
            )

        logger.debug(
            f"Prepared {len(formatted_items)} items for Gemini batch submission"
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

    def _convert_to_gemini_format(
        self, custom_id: str, input_payload: dict, model: str, mode: str
    ) -> dict:
        """
        Convert input payload to Gemini batch request format.

        Args:
            custom_id: Custom identifier for the request
            input_payload: Input data for the request
            model: Model name
            mode: Processing mode

        Returns:
            Gemini-formatted request
        """
        # Create Gemini batch request format based on documentation
        gemini_request = {"contents": [], "config": {"response_modalities": ["text"]}}

        # Handle different input formats
        if "messages" in input_payload:
            # Convert messages to Gemini contents format
            for message in input_payload["messages"]:
                role = message.get("role", "user")
                content = message.get("content", "")

                # Map roles to Gemini format
                gemini_role = "user" if role in ["user", "human"] else "model"

                gemini_content = {"role": gemini_role, "parts": [{"text": content}]}
                gemini_request["contents"].append(gemini_content)

        elif "content" in input_payload:
            # Direct content
            content = input_payload["content"]
            gemini_request["contents"].append(
                {"role": "user", "parts": [{"text": content}]}
            )

        elif "text" in input_payload:
            # Text field
            text = input_payload["text"]
            gemini_request["contents"].append(
                {"role": "user", "parts": [{"text": text}]}
            )

        else:
            raise ProviderPermanentError(
                f"Unsupported input format for Gemini. Expected 'messages', 'content', or 'text' field",
                self.PROVIDER_NAME,
            )

        # Add generation config if provided
        if "generation_config" in input_payload:
            gemini_request["config"].update(input_payload["generation_config"])

        # Store custom_id for result mapping (Gemini uses array index internally)
        gemini_request["_custom_id"] = custom_id

        return gemini_request

    def submit(self, prepared_submission: PreparedSubmission) -> ProviderSubmitResult:
        """
        Submit a prepared batch to Gemini.

        Args:
            prepared_submission: Prepared batch submission

        Returns:
            ProviderSubmitResult with Gemini batch ID and status

        Raises:
            ProviderTransientError: For retryable errors (rate limits, temporary issues)
            ProviderPermanentError: For permanent errors (auth, unsupported model)
        """
        try:
            # Remove custom_id from items for submission (keep for tracking)
            submission_items = []
            custom_id_mapping = {}

            for idx, item in enumerate(prepared_submission.items):
                custom_id = item.pop("_custom_id", f"item_{idx}")
                custom_id_mapping[idx] = custom_id
                submission_items.append(item)

            # Create batch job using inlined requests
            batch_job = self.client.batches.create(
                model=prepared_submission.model,
                src=submission_items,  # Inlined requests
            )

            logger.info(
                f"Successfully submitted batch {batch_job.name} to Gemini with {len(submission_items)} items"
            )

            # Store custom_id mapping in submission metadata
            submission_metadata = {
                "custom_id_mapping": custom_id_mapping,
                "model": prepared_submission.model,
                "mode": prepared_submission.mode,
            }

            return ProviderSubmitResult(
                provider_batch_id=batch_job.name,
                status=batch_job.state,
                submitted_at=batch_job.create_time,
                estimated_completion=self._calculate_estimated_completion(batch_job),
                submission_metadata=submission_metadata,
                item_count=len(submission_items),
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
                    f"Gemini batch submission failed (transient): {str(e)}",
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
                    f"Gemini batch submission failed (permanent): {str(e)}",
                    self.PROVIDER_NAME,
                )

            # Default to permanent error for unknown errors
            raise ProviderPermanentError(
                f"Gemini batch submission failed: {str(e)}", self.PROVIDER_NAME
            )

    def poll_status(self, provider_batch_id: str) -> ProviderStatus:
        """
        Poll the status of a Gemini batch job.

        Args:
            provider_batch_id: Gemini batch job identifier

        Returns:
            ProviderStatus with current status and progress information

        Raises:
            ProviderTransientError: For retryable errors (network issues, temporary unavailability)
            ProviderPermanentError: For permanent errors (batch not found, invalid ID)
        """
        try:
            batch_job = self.client.batches.get(name=provider_batch_id)

            # Map Gemini status to internal status
            normalized_status = self.STATUS_MAPPING.get(
                batch_job.state, batch_job.state
            )

            # Build progress information
            progress_info = {
                "state": batch_job.state,
                "create_time": (
                    batch_job.create_time.isoformat() if batch_job.create_time else None
                ),
                "update_time": (
                    batch_job.update_time.isoformat() if batch_job.update_time else None
                ),
            }

            # Add timing information
            if batch_job.start_time:
                progress_info["start_time"] = batch_job.start_time.isoformat()
            if batch_job.end_time:
                progress_info["end_time"] = batch_job.end_time.isoformat()

            # Add error information if present
            if hasattr(batch_job, "error") and batch_job.error:
                progress_info["error"] = str(batch_job.error)

            estimated_completion = None
            if batch_job.state in ["JOB_STATE_RUNNING", "JOB_STATE_PENDING"]:
                # Estimate completion based on creation time (Gemini batches typically complete within 24h)
                if batch_job.create_time:
                    estimated_completion = batch_job.create_time + timedelta(hours=24)

            logger.debug(
                f"Polled batch {provider_batch_id}: state={batch_job.state}, normalized={normalized_status}"
            )

            return ProviderStatus(
                provider_batch_id=provider_batch_id,
                status=batch_job.state,
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
                    f"Gemini batch status polling failed (transient): {str(e)}",
                    self.PROVIDER_NAME,
                )

            # Check for permanent errors
            if any(
                keyword in error_msg
                for keyword in ["not found", "invalid", "401", "403", "404"]
            ):
                raise ProviderPermanentError(
                    f"Gemini batch {provider_batch_id} not found or invalid",
                    self.PROVIDER_NAME,
                )

            # Default to transient error for status polling
            raise ProviderTransientError(
                f"Gemini batch status polling failed: {str(e)}", self.PROVIDER_NAME
            )

    def fetch_results(
        self, provider_batch_id: str, custom_id_mapping: Optional[dict] = None
    ) -> List[ProviderResultRow]:
        """
        Fetch results from a completed Gemini batch job.

        Args:
            provider_batch_id: Gemini batch job identifier

        Returns:
            List of ProviderResultRow with results for each item

        Raises:
            ProviderTransientError: For retryable errors
            ProviderPermanentError: For permanent errors (batch not found, not completed)
        """
        try:
            # Get batch status first
            batch_job = self.client.batches.get(name=provider_batch_id)

            if batch_job.state != "JOB_STATE_SUCCEEDED":
                raise ProviderPermanentError(
                    f"Cannot fetch results for batch {provider_batch_id} with state {batch_job.state}",
                    self.PROVIDER_NAME,
                )

            results = []

            # For Gemini, results are stored with inlined responses in the destination
            if hasattr(batch_job, "dest") and batch_job.dest:
                # Handle inlined responses
                if (
                    hasattr(batch_job.dest, "inlined_responses")
                    and batch_job.dest.inlined_responses
                ):
                    parsed = self._parse_inlined_responses(
                        batch_job.dest.inlined_responses
                    )
                    # If mapping provided (index -> original id) remap here instead of higher layer
                    if custom_id_mapping:
                        remapped = []
                        for idx, row in enumerate(parsed):
                            if row.custom_id.startswith("item_"):
                                try:
                                    i = int(row.custom_id.split("_", 1)[1])
                                    original = custom_id_mapping.get(i)
                                    if original:
                                        row.custom_id = str(original)
                                except Exception:
                                    pass
                            remapped.append(row)
                        results.extend(remapped)
                    else:
                        results.extend(parsed)
                else:
                    # Could be file-based results, but for now we focus on inlined
                    logger.warning(
                        f"Batch {provider_batch_id} has non-inlined results which are not yet supported"
                    )

            logger.info(
                f"Fetched {len(results)} results from Gemini batch {provider_batch_id}"
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
                    f"Gemini batch result fetching failed (transient): {str(e)}",
                    self.PROVIDER_NAME,
                )

            # Default to permanent error for result fetching
            raise ProviderPermanentError(
                f"Gemini batch result fetching failed: {str(e)}", self.PROVIDER_NAME
            )

    def _parse_inlined_responses(self, inlined_responses) -> List[ProviderResultRow]:
        """
        Parse Gemini inlined responses into ProviderResultRow format.

        Args:
            inlined_responses: List of response objects from Gemini

        Returns:
            List of ProviderResultRow objects
        """
        results = []

        for idx, item_data in enumerate(inlined_responses):
            try:
                custom_id = f"item_{idx}"  # Default fallback
                response = None
                output_text = None
                error_message = None
                is_success = True

                # Extract content from Gemini response
                response = item_data.response
                try:
                    output_text = response.text
                except AttributeError:
                    # Fallback for raw response structure
                    try:
                        if response.candidates:
                            candidate = response.candidates[0]
                            content = candidate.content
                            parts = content.parts

                            # Extract text from first part
                            first_part = parts[0]
                            output_text = first_part.text
                    except (IndexError, KeyError, AttributeError):
                        output_text = None

                # Check for errors
                if "error" in response:
                    error_message = str(response["error"])
                    is_success = False

                results.append(
                    ProviderResultRow(
                        custom_id=custom_id,
                        output_text=output_text,
                        output_data=response.model_dump() if response else None,
                        raw_metadata=item_data.model_dump() if item_data else None,
                        error_message=error_message,
                        is_success=is_success,
                    )
                )

            except Exception as e:
                logger.error(f"Failed to parse Gemini response {idx}: {e}")
                # Create error result for unparseable response
                results.append(
                    ProviderResultRow(
                        custom_id=f"item_{idx}",
                        output_text=None,
                        output_data=None,
                        raw_metadata=response,
                        error_message=f"Failed to parse response: {str(e)}",
                        is_success=False,
                    )
                )

        return results

    def cancel(self, provider_batch_id: str) -> bool:
        """
        Cancel a Gemini batch job.

        Args:
            provider_batch_id: Gemini batch job identifier

        Returns:
            True if cancellation was successful

        Raises:
            ProviderPermanentError: If batch cannot be cancelled or not found
        """
        try:
            # Gemini uses the cancel method on the batches API
            result = self.client.batches.cancel(name=provider_batch_id)

            # Check if cancellation was successful
            if hasattr(result, "state") and result.state in [
                "JOB_STATE_CANCELLED",
                "JOB_STATE_CANCELLING",
            ]:
                logger.info(f"Successfully cancelled Gemini batch {provider_batch_id}")
                return True
            else:
                logger.warning(
                    f"Gemini batch {provider_batch_id} cancellation returned unexpected state"
                )
                return False

        except Exception as e:
            error_msg = str(e).lower()

            if any(
                keyword in error_msg
                for keyword in ["not found", "invalid", "401", "403", "404"]
            ):
                raise ProviderPermanentError(
                    f"Gemini batch {provider_batch_id} not found or cannot be cancelled",
                    self.PROVIDER_NAME,
                )

            raise ProviderPermanentError(
                f"Gemini batch cancellation failed: {str(e)}", self.PROVIDER_NAME
            )

    def _calculate_estimated_completion(self, batch_job) -> Optional[datetime]:
        """
        Calculate estimated completion time for a batch job.

        Args:
            batch_job: Gemini BatchJob object

        Returns:
            Estimated completion datetime or None
        """
        # Gemini batches typically complete within 24 hours
        if batch_job.create_time:
            return batch_job.create_time + timedelta(hours=24)
        return None

    def _extract_retry_after(self, error_message: str) -> Optional[int]:
        """
        Extract retry-after value from error message.

        Args:
            error_message: Error message from Gemini

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

"""
Integration tests for OpenAI Batch Provider

Tests end-to-end batch processing flow with mocked OpenAI API responses.
Covers the acceptance criteria: ≥3 prompts, status transitions, partial failures.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest
from openai.types import Batch, BatchRequestCounts, FileObject

from inference_core.llm.batch.providers.openai_provider import OpenAIBatchProvider
from inference_core.llm.batch.registry import get_global_registry


@pytest.mark.integration
class TestOpenAIBatchProviderIntegration:
    """Integration tests for OpenAI Batch Provider end-to-end flow."""

    def setup_method(self):
        """Setup test environment."""
        self.config = {"api_key": "test-api-key"}
        self.provider = OpenAIBatchProvider(self.config)

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_end_to_end_successful_batch_flow(self, mock_openai_class):
        """
        Test complete end-to-end batch processing flow with 3+ prompts.

        This test verifies the acceptance criteria:
        - End-to-end flow succeeds for ≥3 prompts
        - All status transitions work correctly
        """
        # Setup mocked OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        # Prepare test data with 4 prompts (exceeds requirement of ≥3)
        batch_items = [
            {
                "id": "request-1",
                "input_payload": {
                    "messages": [
                        {"role": "user", "content": "What is the capital of France?"}
                    ],
                    "max_tokens": 50,
                },
            },
            {
                "id": "request-2",
                "input_payload": {
                    "messages": [
                        {"role": "user", "content": "Explain quantum computing"}
                    ],
                    "max_tokens": 100,
                },
            },
            {
                "id": "request-3",
                "input_payload": {
                    "messages": [
                        {"role": "user", "content": "Write a haiku about programming"}
                    ],
                    "max_tokens": 75,
                },
            },
            {
                "id": "request-4",
                "input_payload": {
                    "messages": [
                        {"role": "user", "content": "List the primary colors"}
                    ],
                    "max_tokens": 30,
                },
            },
        ]

        # Step 1: Prepare payloads
        batch_job_id = uuid4()
        prepared = provider.prepare_payloads(
            batch_items, "gpt-4", "chat", {"batch_job_id": batch_job_id}
        )

        assert len(prepared.items) == 4
        assert prepared.provider == "openai"
        assert prepared.model == "gpt-4"
        assert prepared.mode == "chat"

        # Verify JSONL formatting
        for i, item in enumerate(prepared.items):
            assert item["custom_id"] == f"request-{i+1}"
            assert item["method"] == "POST"
            assert item["url"] == "/v1/chat/completions"
            assert item["body"]["model"] == "gpt-4"
            assert "messages" in item["body"]

        # Step 2: Submit batch
        # Mock file upload
        mock_file = Mock(spec=FileObject)
        mock_file.id = "file-abc123"
        mock_client.files.create.return_value = mock_file

        # Mock batch creation
        mock_batch = Mock(spec=Batch)
        mock_batch.id = "batch-def456"
        mock_batch.status = "validating"
        mock_batch.created_at = int(datetime.now().timestamp())
        mock_batch.endpoint = "/v1/chat/completions"
        mock_batch.completion_window = "24h"
        mock_client.batches.create.return_value = mock_batch

        submit_result = provider.submit(prepared)

        assert submit_result.provider_batch_id == "batch-def456"
        assert submit_result.status == "validating"
        assert submit_result.item_count == 4
        assert submit_result.submission_metadata["input_file_id"] == "file-abc123"

        # Verify OpenAI API calls
        mock_client.files.create.assert_called_once()
        file_call_args = mock_client.files.create.call_args[1]
        assert file_call_args["purpose"] == "batch"

        mock_client.batches.create.assert_called_once()
        batch_call_args = mock_client.batches.create.call_args[1]
        assert batch_call_args["input_file_id"] == "file-abc123"
        assert batch_call_args["endpoint"] == "/v1/chat/completions"

        # Step 3: Poll status transitions (validating → in_progress → completed)
        batch_id = submit_result.provider_batch_id

        # Status: validating
        mock_batch_validating = Mock(spec=Batch)
        mock_batch_validating.id = batch_id
        mock_batch_validating.status = "validating"
        mock_batch_validating.request_counts = Mock(spec=BatchRequestCounts)
        mock_batch_validating.request_counts.total = 4
        mock_batch_validating.request_counts.completed = 0
        mock_batch_validating.request_counts.failed = 0
        mock_batch_validating.completed_at = None
        mock_batch_validating.expires_at = int(
            (datetime.now() + timedelta(hours=23)).timestamp()
        )
        mock_batch_validating.output_file_id = None
        mock_batch_validating.error_file_id = None

        mock_client.batches.retrieve.return_value = mock_batch_validating
        status1 = provider.poll_status(batch_id)

        assert status1.status == "validating"
        assert status1.normalized_status == "submitted"
        assert status1.progress_info["total_requests"] == 4
        assert status1.progress_info["completed_requests"] == 0

        # Status: in_progress
        mock_batch_progress = Mock(spec=Batch)
        mock_batch_progress.id = batch_id
        mock_batch_progress.status = "in_progress"
        mock_batch_progress.request_counts = Mock(spec=BatchRequestCounts)
        mock_batch_progress.request_counts.total = 4
        mock_batch_progress.request_counts.completed = 2
        mock_batch_progress.request_counts.failed = 0
        mock_batch_progress.completed_at = None
        mock_batch_progress.expires_at = int(
            (datetime.now() + timedelta(hours=22)).timestamp()
        )
        mock_batch_progress.output_file_id = None
        mock_batch_progress.error_file_id = None

        mock_client.batches.retrieve.return_value = mock_batch_progress
        status2 = provider.poll_status(batch_id)

        assert status2.status == "in_progress"
        assert status2.normalized_status == "in_progress"
        assert status2.progress_info["completed_requests"] == 2

        # Status: completed
        mock_batch_completed = Mock(spec=Batch)
        mock_batch_completed.id = batch_id
        mock_batch_completed.status = "completed"
        mock_batch_completed.request_counts = Mock(spec=BatchRequestCounts)
        mock_batch_completed.request_counts.total = 4
        mock_batch_completed.request_counts.completed = 4
        mock_batch_completed.request_counts.failed = 0
        mock_batch_completed.completed_at = int(datetime.now().timestamp())
        mock_batch_completed.expires_at = None
        mock_batch_completed.output_file_id = "file-output-789"
        mock_batch_completed.error_file_id = None

        mock_client.batches.retrieve.return_value = mock_batch_completed
        status3 = provider.poll_status(batch_id)

        assert status3.status == "completed"
        assert status3.normalized_status == "completed"
        assert status3.progress_info["completed_requests"] == 4
        assert status3.progress_info["output_file_id"] == "file-output-789"

        # Step 4: Fetch results
        # Mock successful results for all 4 requests
        output_content = """{"custom_id": "request-1", "response": {"body": {"choices": [{"message": {"content": "The capital of France is Paris."}}]}}}
{"custom_id": "request-2", "response": {"body": {"choices": [{"message": {"content": "Quantum computing uses quantum mechanics principles..."}}]}}}
{"custom_id": "request-3", "response": {"body": {"choices": [{"message": {"content": "Code flows like water\\nLogic branches through the night\\nBugs hide in shadows"}}]}}}
{"custom_id": "request-4", "response": {"body": {"choices": [{"message": {"content": "The primary colors are red, blue, and yellow."}}]}}}"""

        mock_output_response = Mock()
        mock_output_response.read.return_value = output_content.encode("utf-8")
        mock_client.files.content.return_value = mock_output_response

        results = provider.fetch_results(batch_id)

        # Verify results
        assert len(results) == 4

        # All results should be successful
        successful_results = [r for r in results if r.is_success]
        assert len(successful_results) == 4

        # Verify specific responses
        result_by_id = {r.custom_id: r for r in results}

        assert (
            result_by_id["request-1"].output_text == "The capital of France is Paris."
        )
        assert result_by_id["request-2"].output_text.startswith(
            "Quantum computing uses"
        )
        assert "Code flows like water" in result_by_id["request-3"].output_text
        assert "red, blue, and yellow" in result_by_id["request-4"].output_text

        # Verify API calls for result fetching
        mock_client.files.content.assert_called_once_with("file-output-789")

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_partial_failure_scenario(self, mock_openai_class):
        """
        Test batch processing with partial failures.

        Verifies acceptance criteria:
        - Partial failure scenario recorded (one item fails, others succeed)
        """
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        # Prepare test data with 3 prompts
        batch_items = [
            {
                "id": "success-1",
                "input_payload": {
                    "messages": [{"role": "user", "content": "Hello world"}]
                },
            },
            {
                "id": "failure-1",
                "input_payload": {
                    "messages": [{"role": "user", "content": "This will fail"}]
                },
            },
            {
                "id": "success-2",
                "input_payload": {
                    "messages": [{"role": "user", "content": "Another success"}]
                },
            },
        ]

        prepared = provider.prepare_payloads(batch_items, "gpt-4", "chat")

        # Mock submission
        mock_file = Mock(spec=FileObject)
        mock_file.id = "file-test"
        mock_client.files.create.return_value = mock_file

        mock_batch = Mock(spec=Batch)
        mock_batch.id = "batch-partial"
        mock_batch.status = "validating"
        mock_batch.created_at = int(datetime.now().timestamp())
        mock_batch.endpoint = "/v1/chat/completions"
        mock_batch.completion_window = "24h"
        mock_client.batches.create.return_value = mock_batch

        submit_result = provider.submit(prepared)

        # Mock completed status with partial failures
        mock_batch_completed = Mock(spec=Batch)
        mock_batch_completed.id = "batch-partial"
        mock_batch_completed.status = "completed"
        mock_batch_completed.request_counts = Mock(spec=BatchRequestCounts)
        mock_batch_completed.request_counts.total = 3
        mock_batch_completed.request_counts.completed = 2
        mock_batch_completed.request_counts.failed = 1
        mock_batch_completed.completed_at = int(datetime.now().timestamp())
        mock_batch_completed.expires_at = None
        mock_batch_completed.output_file_id = "file-output-partial"
        mock_batch_completed.error_file_id = "file-error-partial"

        mock_client.batches.retrieve.return_value = mock_batch_completed

        # Mock result files with mixed success/failure
        output_content = """{"custom_id": "success-1", "response": {"body": {"choices": [{"message": {"content": "Hello! How can I help you?"}}]}}}
{"custom_id": "success-2", "response": {"body": {"choices": [{"message": {"content": "I'm here to assist you."}}]}}}"""

        error_content = """{"custom_id": "failure-1", "error": {"message": "Request failed due to policy violation", "code": "content_policy_violation"}}"""

        mock_output_response = Mock()
        mock_output_response.read.return_value = output_content.encode("utf-8")
        mock_error_response = Mock()
        mock_error_response.read.return_value = error_content.encode("utf-8")

        mock_client.files.content.side_effect = [
            mock_output_response,
            mock_error_response,
        ]

        # Fetch and verify results
        results = provider.fetch_results("batch-partial")

        assert len(results) == 3

        # Check successful results
        successful_results = [r for r in results if r.is_success]
        assert len(successful_results) == 2

        success_ids = {r.custom_id for r in successful_results}
        assert "success-1" in success_ids
        assert "success-2" in success_ids

        # Check failed results
        failed_results = [r for r in results if not r.is_success]
        assert len(failed_results) == 1

        failed_result = failed_results[0]
        assert failed_result.custom_id == "failure-1"
        assert (
            failed_result.error_message
            == "[content_policy_violation] Request failed due to policy violation"
        )
        assert failed_result.output_text is None
        assert failed_result.is_success is False

        # Verify status reflects partial completion
        status = provider.poll_status("batch-partial")
        assert status.progress_info["total_requests"] == 3
        assert status.progress_info["completed_requests"] == 2
        assert status.progress_info["failed_requests"] == 1

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_error_handling_and_status_transitions(self, mock_openai_class):
        """
        Test error handling and different status transitions.

        Verifies acceptance criteria:
        - Handles provider status transitions (queued→in_progress→completed/failed)
        """
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        batch_items = [
            {
                "id": "test-1",
                "input_payload": {"messages": [{"role": "user", "content": "Test"}]},
            }
        ]

        prepared = provider.prepare_payloads(batch_items, "gpt-4", "chat")

        # Mock submission
        mock_file = Mock(spec=FileObject)
        mock_file.id = "file-error-test"
        mock_client.files.create.return_value = mock_file

        mock_batch = Mock(spec=Batch)
        mock_batch.id = "batch-error-test"
        mock_batch.status = "validating"
        mock_batch.created_at = int(datetime.now().timestamp())
        mock_batch.endpoint = "/v1/chat/completions"
        mock_batch.completion_window = "24h"
        mock_client.batches.create.return_value = mock_batch

        submit_result = provider.submit(prepared)
        batch_id = submit_result.provider_batch_id

        # Test different status transitions
        status_sequence = [
            ("validating", "submitted"),
            ("in_progress", "in_progress"),
            ("finalizing", "in_progress"),
            ("failed", "failed"),
            ("expired", "failed"),
            ("cancelled", "cancelled"),
        ]

        for provider_status, expected_normalized in status_sequence:
            mock_batch_status = Mock(spec=Batch)
            mock_batch_status.id = batch_id
            mock_batch_status.status = provider_status
            mock_batch_status.request_counts = Mock(spec=BatchRequestCounts)
            mock_batch_status.request_counts.total = 1
            mock_batch_status.request_counts.completed = (
                0 if provider_status != "completed" else 1
            )
            mock_batch_status.request_counts.failed = (
                1 if provider_status in ["failed", "expired"] else 0
            )
            mock_batch_status.completed_at = (
                int(datetime.now().timestamp())
                if provider_status in ["completed", "failed", "expired", "cancelled"]
                else None
            )
            mock_batch_status.expires_at = None
            mock_batch_status.output_file_id = None
            mock_batch_status.error_file_id = None

            mock_client.batches.retrieve.return_value = mock_batch_status

            status = provider.poll_status(batch_id)
            assert status.status == provider_status
            assert status.normalized_status == expected_normalized

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_batch_cancellation_flow(self, mock_openai_class):
        """Test batch cancellation functionality."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        # Mock successful cancellation
        mock_cancelled_batch = Mock(spec=Batch)
        mock_cancelled_batch.status = "cancelled"
        mock_client.batches.cancel.return_value = mock_cancelled_batch

        result = provider.cancel("batch-to-cancel")

        assert result is True
        mock_client.batches.cancel.assert_called_once_with("batch-to-cancel")

    def test_provider_available_in_registry(self):
        """Test that OpenAI provider is properly registered and accessible."""
        # Verify provider is registered
        registry = get_global_registry()
        assert registry.is_registered("openai") is True

        # Verify we can create an instance
        config = {"api_key": "test-key"}
        provider = registry.create_provider("openai", config)

        assert isinstance(provider, OpenAIBatchProvider)
        assert provider.get_provider_name() == "openai"

        # Verify provider supports expected models and modes
        assert provider.supports_model("gpt-4", "chat") is True
        assert provider.supports_model("gpt-3.5-turbo", "chat") is True
        assert provider.supports_model("claude-3", "chat") is False

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_comprehensive_jsonl_formatting(self, mock_openai_class):
        """Test comprehensive JSONL formatting with various input types."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        # Test complex batch items with various payload types
        batch_items = [
            {
                "id": "complex-1",
                "input_payload": {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is AI?"},
                    ],
                    "max_tokens": 150,
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
            },
            {
                "id": "complex-2",
                "input_payload": json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Explain machine learning"}
                        ],
                        "max_tokens": 200,
                        "presence_penalty": 0.1,
                    }
                ),
            },
        ]

        prepared = provider.prepare_payloads(batch_items, "gpt-4", "chat")

        # Verify JSONL structure
        jsonl_content = provider._create_jsonl_content(prepared.items)
        lines = jsonl_content.split("\n")

        assert len(lines) == 2

        # Parse and verify first line
        line1_data = json.loads(lines[0])
        assert line1_data["custom_id"] == "complex-1"
        assert line1_data["method"] == "POST"
        assert line1_data["url"] == "/v1/chat/completions"
        assert line1_data["body"]["model"] == "gpt-4"
        assert len(line1_data["body"]["messages"]) == 2
        assert line1_data["body"]["max_tokens"] == 150
        assert line1_data["body"]["temperature"] == 0.7
        assert line1_data["body"]["top_p"] == 0.9

        # Parse and verify second line
        line2_data = json.loads(lines[1])
        assert line2_data["custom_id"] == "complex-2"
        assert line2_data["body"]["max_tokens"] == 200
        assert line2_data["body"]["presence_penalty"] == 0.1

        # Verify no extra whitespace or formatting issues
        assert not jsonl_content.startswith("\n")
        assert not jsonl_content.endswith("\n\n")
        assert "\n\n" not in jsonl_content

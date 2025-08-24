"""
Unit tests for OpenAI Batch Provider

Tests the OpenAI provider implementation with mocked OpenAI API responses.
Covers all provider methods and error scenarios.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest
from openai.types import Batch, BatchRequestCounts, FileObject

from inference_core.llm.batch import (
    ProviderPermanentError,
    ProviderTransientError,
    registry,
)
from inference_core.llm.batch.providers.openai_provider import OpenAIBatchProvider


class TestOpenAIBatchProvider:
    """Test OpenAI Batch Provider implementation."""

    def setup_method(self):
        """Setup test environment."""
        self.config = {"api_key": "test-api-key"}
        self.provider = OpenAIBatchProvider(self.config)

    def test_provider_name(self):
        """Test provider name is correct."""
        assert self.provider.PROVIDER_NAME == "openai"
        assert self.provider.get_provider_name() == "openai"

    def test_initialization_with_config(self):
        """Test provider initialization with config."""
        assert self.provider.config == self.config
        assert hasattr(self.provider, "client")

    def test_initialization_without_api_key(self):
        """Test provider initialization fails without API key."""
        with pytest.raises(ProviderPermanentError, match="OpenAI API key is required"):
            OpenAIBatchProvider({})

    def test_supports_model_valid_combinations(self):
        """Test supports_model for valid model/mode combinations."""
        # Valid combinations
        assert self.provider.supports_model("gpt-3.5-turbo", "chat") is True
        assert self.provider.supports_model("gpt-4", "chat") is True
        assert self.provider.supports_model("gpt-4o", "chat") is True
        assert self.provider.supports_model("gpt-5-mini", "chat") is True

    def test_supports_model_invalid_combinations(self):
        """Test supports_model for invalid model/mode combinations."""
        # Invalid modes
        assert self.provider.supports_model("gpt-4", "embedding") is False
        assert self.provider.supports_model("gpt-4", "completion") is False

        # Invalid models
        assert self.provider.supports_model("text-embedding-ada-002", "chat") is False
        assert self.provider.supports_model("claude-3", "chat") is False

    def test_prepare_payloads_success(self):
        """Test successful payload preparation."""
        batch_items = [
            {
                "id": "item-1",
                "input_payload": {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 100,
                },
            },
            {
                "id": "item-2",
                "input_payload": {
                    "messages": [{"role": "user", "content": "World"}],
                    "temperature": 0.7,
                },
            },
        ]

        result = self.provider.prepare_payloads(batch_items, "gpt-4", "chat")

        assert result.provider == "openai"
        assert result.model == "gpt-4"
        assert result.mode == "chat"
        assert len(result.items) == 2

        # Check first item format
        item1 = result.items[0]
        assert item1["custom_id"] == "item-1"
        assert item1["method"] == "POST"
        assert item1["url"] == "/v1/chat/completions"
        assert item1["body"]["model"] == "gpt-4"
        assert item1["body"]["messages"] == [{"role": "user", "content": "Hello"}]
        assert item1["body"]["max_tokens"] == 100

        # Check second item format
        item2 = result.items[1]
        assert item2["custom_id"] == "item-2"
        assert item2["body"]["temperature"] == 0.7

    def test_prepare_payloads_with_json_string_input(self):
        """Test payload preparation with JSON string input_payload."""
        batch_items = [
            {
                "id": "item-1",
                "input_payload": json.dumps(
                    {"messages": [{"role": "user", "content": "Test"}]}
                ),
            }
        ]

        result = self.provider.prepare_payloads(batch_items, "gpt-4", "chat")

        assert len(result.items) == 1
        item = result.items[0]
        assert item["body"]["messages"] == [{"role": "user", "content": "Test"}]

    def test_prepare_payloads_empty_items(self):
        """Test prepare_payloads fails with empty items."""
        with pytest.raises(ProviderPermanentError, match="Batch items cannot be empty"):
            self.provider.prepare_payloads([], "gpt-4", "chat")

    def test_prepare_payloads_unsupported_model(self):
        """Test prepare_payloads fails with unsupported model."""
        batch_items = [
            {
                "id": "item-1",
                "input_payload": {"messages": [{"role": "user", "content": "Test"}]},
            }
        ]

        with pytest.raises(
            ProviderPermanentError, match="is not supported by OpenAI provider"
        ):
            self.provider.prepare_payloads(batch_items, "claude-3", "chat")

    def test_prepare_payloads_missing_id(self):
        """Test prepare_payloads fails with missing item ID."""
        batch_items = [
            {"input_payload": {"messages": [{"role": "user", "content": "Test"}]}}
        ]

        with pytest.raises(ProviderPermanentError, match="missing required 'id' field"):
            self.provider.prepare_payloads(batch_items, "gpt-4", "chat")

    def test_prepare_payloads_missing_input_payload(self):
        """Test prepare_payloads fails with missing input_payload."""
        batch_items = [{"id": "item-1"}]

        with pytest.raises(ProviderPermanentError, match="missing 'input_payload'"):
            self.provider.prepare_payloads(batch_items, "gpt-4", "chat")

    def test_prepare_payloads_missing_messages_for_chat(self):
        """Test prepare_payloads fails with missing messages for chat mode."""
        batch_items = [{"id": "item-1", "input_payload": {"temperature": 0.7}}]

        with pytest.raises(
            ProviderPermanentError, match="Chat mode requires 'messages'"
        ):
            self.provider.prepare_payloads(batch_items, "gpt-4", "chat")

    def test_prepare_payloads_invalid_json_string(self):
        """Test prepare_payloads fails with invalid JSON string."""
        batch_items = [{"id": "item-1", "input_payload": "{invalid json}"}]

        with pytest.raises(
            ProviderPermanentError, match="Invalid JSON in input_payload"
        ):
            self.provider.prepare_payloads(batch_items, "gpt-4", "chat")

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_submit_success(self, mock_openai_class):
        """Test successful batch submission."""
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock file upload
        mock_file = Mock(spec=FileObject)
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file

        # Mock batch creation
        mock_batch = Mock(spec=Batch)
        mock_batch.id = "batch-456"
        mock_batch.status = "validating"
        mock_batch.created_at = int(datetime.now().timestamp())
        mock_batch.endpoint = "/v1/chat/completions"
        mock_batch.completion_window = "24h"
        mock_client.batches.create.return_value = mock_batch

        # Create provider and prepared submission
        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        batch_items = [
            {
                "id": "item-1",
                "input_payload": {"messages": [{"role": "user", "content": "Test"}]},
            }
        ]
        prepared = provider.prepare_payloads(
            batch_items, "gpt-4", "chat", {"batch_job_id": uuid4()}
        )

        # Submit
        result = provider.submit(prepared)

        # Verify result
        assert result.provider_batch_id == "batch-456"
        assert result.status == "validating"
        assert isinstance(result.submitted_at, datetime)
        assert result.item_count == 1
        assert result.submission_metadata["input_file_id"] == "file-123"

        # Verify OpenAI API calls
        mock_client.files.create.assert_called_once()
        mock_client.batches.create.assert_called_once()

        batch_args = mock_client.batches.create.call_args[1]
        assert batch_args["input_file_id"] == "file-123"
        assert batch_args["endpoint"] == "/v1/chat/completions"
        assert batch_args["completion_window"] == "24h"

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_submit_file_upload_error(self, mock_openai_class):
        """Test submit fails when file upload fails."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.files.create.side_effect = Exception("Upload failed")

        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        batch_items = [
            {
                "id": "item-1",
                "input_payload": {"messages": [{"role": "user", "content": "Test"}]},
            }
        ]
        prepared = provider.prepare_payloads(batch_items, "gpt-4", "chat")

        with pytest.raises(
            ProviderPermanentError, match="OpenAI batch submission failed"
        ):
            provider.submit(prepared)

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_submit_rate_limit_error(self, mock_openai_class):
        """Test submit raises transient error for rate limits."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.files.create.side_effect = Exception("Rate limit exceeded")

        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        batch_items = [
            {
                "id": "item-1",
                "input_payload": {"messages": [{"role": "user", "content": "Test"}]},
            }
        ]
        prepared = provider.prepare_payloads(batch_items, "gpt-4", "chat")

        with pytest.raises(ProviderTransientError) as exc_info:
            provider.submit(prepared)

        assert "transient" in str(exc_info.value)

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_submit_auth_error(self, mock_openai_class):
        """Test submit raises permanent error for auth issues."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.files.create.side_effect = Exception("Unauthorized access")

        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        batch_items = [
            {
                "id": "item-1",
                "input_payload": {"messages": [{"role": "user", "content": "Test"}]},
            }
        ]
        prepared = provider.prepare_payloads(batch_items, "gpt-4", "chat")

        with pytest.raises(ProviderPermanentError) as exc_info:
            provider.submit(prepared)

        assert "permanent" in str(exc_info.value)

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_poll_status_success(self, mock_openai_class):
        """Test successful status polling."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock batch status response
        mock_batch = Mock(spec=Batch)
        mock_batch.id = "batch-456"
        mock_batch.status = "in_progress"
        mock_batch.request_counts = Mock(spec=BatchRequestCounts)
        mock_batch.request_counts.total = 10
        mock_batch.request_counts.completed = 3
        mock_batch.request_counts.failed = 1
        mock_batch.completed_at = None
        mock_batch.expires_at = int((datetime.now() + timedelta(hours=23)).timestamp())
        mock_batch.output_file_id = None
        mock_batch.error_file_id = None

        mock_client.batches.retrieve.return_value = mock_batch

        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        # Poll status
        result = provider.poll_status("batch-456")

        # Verify result
        assert result.provider_batch_id == "batch-456"
        assert result.status == "in_progress"
        assert result.normalized_status == "in_progress"
        assert result.progress_info["total_requests"] == 10
        assert result.progress_info["completed_requests"] == 3
        assert result.progress_info["failed_requests"] == 1
        assert "expires_at" in result.progress_info

        # Verify API call
        mock_client.batches.retrieve.assert_called_once_with("batch-456")

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_poll_status_completed(self, mock_openai_class):
        """Test polling status for completed batch."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_batch = Mock(spec=Batch)
        mock_batch.id = "batch-456"
        mock_batch.status = "completed"
        mock_batch.request_counts = Mock(spec=BatchRequestCounts)
        mock_batch.request_counts.total = 5
        mock_batch.request_counts.completed = 4
        mock_batch.request_counts.failed = 1
        mock_batch.completed_at = int(datetime.now().timestamp())
        mock_batch.expires_at = None
        mock_batch.output_file_id = "file-output-123"
        mock_batch.error_file_id = "file-error-123"

        mock_client.batches.retrieve.return_value = mock_batch

        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        result = provider.poll_status("batch-456")

        assert result.status == "completed"
        assert result.normalized_status == "completed"
        assert "completed_at" in result.progress_info
        assert result.progress_info["output_file_id"] == "file-output-123"
        assert result.progress_info["error_file_id"] == "file-error-123"

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_poll_status_not_found(self, mock_openai_class):
        """Test polling status for non-existent batch."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.batches.retrieve.side_effect = Exception("Batch not found")

        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        with pytest.raises(ProviderPermanentError, match="not found or invalid"):
            provider.poll_status("nonexistent-batch")

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_poll_status_network_error(self, mock_openai_class):
        """Test polling status with network error."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.batches.retrieve.side_effect = Exception("Network timeout")

        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        with pytest.raises(ProviderTransientError, match="transient"):
            provider.poll_status("batch-456")

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_fetch_results_success(self, mock_openai_class):
        """Test successful result fetching."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock batch status
        mock_batch = Mock(spec=Batch)
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file-output-123"
        mock_batch.error_file_id = "file-error-123"
        mock_client.batches.retrieve.return_value = mock_batch

        # Mock output file content
        output_content = """{"custom_id": "item-1", "response": {"body": {"choices": [{"message": {"content": "Success response"}}]}}}
{"custom_id": "item-2", "response": {"body": {"choices": [{"message": {"content": "Another response"}}]}}}"""

        # Mock error file content
        error_content = """{"custom_id": "item-3", "error": {"message": "Failed to process", "code": "invalid_request"}}"""

        # Mock file content retrieval
        mock_output_response = Mock()
        mock_output_response.read.return_value = output_content.encode("utf-8")
        mock_error_response = Mock()
        mock_error_response.read.return_value = error_content.encode("utf-8")

        mock_client.files.content.side_effect = [
            mock_output_response,
            mock_error_response,
        ]

        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        # Fetch results
        results = provider.fetch_results("batch-456")

        # Verify results
        assert len(results) == 3

        # Check successful results
        success_results = [r for r in results if r.is_success]
        assert len(success_results) == 2
        assert success_results[0].custom_id == "item-1"
        assert success_results[0].output_text == "Success response"
        assert success_results[1].custom_id == "item-2"
        assert success_results[1].output_text == "Another response"

        # Check failed result
        failed_results = [r for r in results if not r.is_success]
        assert len(failed_results) == 1
        assert failed_results[0].custom_id == "item-3"
        assert failed_results[0].error_message == "[invalid_request] Failed to process"
        assert failed_results[0].output_text is None

        # Verify API calls
        mock_client.batches.retrieve.assert_called_once_with("batch-456")
        assert mock_client.files.content.call_count == 2

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_fetch_results_not_completed(self, mock_openai_class):
        """Test fetching results for non-completed batch."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_batch = Mock(spec=Batch)
        mock_batch.status = "in_progress"
        mock_client.batches.retrieve.return_value = mock_batch

        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        with pytest.raises(
            ProviderPermanentError,
            match="Cannot fetch results.*with status in_progress",
        ):
            provider.fetch_results("batch-456")

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_cancel_success(self, mock_openai_class):
        """Test successful batch cancellation."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_batch = Mock(spec=Batch)
        mock_batch.status = "cancelled"
        mock_client.batches.cancel.return_value = mock_batch

        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        result = provider.cancel("batch-456")

        assert result is True
        mock_client.batches.cancel.assert_called_once_with("batch-456")

    @patch("inference_core.llm.batch.providers.openai_provider.OpenAI")
    def test_cancel_not_found(self, mock_openai_class):
        """Test cancelling non-existent batch."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.batches.cancel.side_effect = Exception("Batch not found")

        provider = OpenAIBatchProvider(self.config)
        provider.client = mock_client

        with pytest.raises(
            ProviderPermanentError, match="not found or cannot be cancelled"
        ):
            provider.cancel("nonexistent-batch")

    def test_status_mapping(self):
        """Test OpenAI status mapping to internal status."""
        assert self.provider.STATUS_MAPPING["validating"] == "submitted"
        assert self.provider.STATUS_MAPPING["in_progress"] == "in_progress"
        assert self.provider.STATUS_MAPPING["finalizing"] == "in_progress"
        assert self.provider.STATUS_MAPPING["completed"] == "completed"
        assert self.provider.STATUS_MAPPING["failed"] == "failed"
        assert self.provider.STATUS_MAPPING["expired"] == "failed"
        assert self.provider.STATUS_MAPPING["cancelled"] == "cancelled"

    def test_create_jsonl_content(self):
        """Test JSONL content creation."""
        items = [
            {"custom_id": "1", "method": "POST", "body": {"test": "data1"}},
            {"custom_id": "2", "method": "POST", "body": {"test": "data2"}},
        ]

        result = self.provider._create_jsonl_content(items)

        lines = result.split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["custom_id"] == "1"
        assert json.loads(lines[1])["custom_id"] == "2"

    def test_extract_retry_after(self):
        """Test retry-after extraction from error messages."""
        assert self.provider._extract_retry_after("retry after 60 seconds") == 60
        assert self.provider._extract_retry_after("try again in 30s") == 30
        assert self.provider._extract_retry_after("rate limit exceeded") == 60
        assert self.provider._extract_retry_after("unknown error") is None


class TestOpenAIProviderRegistration:
    """Test OpenAI provider registration."""

    def test_provider_registered_in_global_registry(self):
        """Test that OpenAI provider is registered globally."""
        # Ensure the provider is registered (it should be auto-registered on import)
        from inference_core.llm.batch.providers.openai_provider import (
            OpenAIBatchProvider,
        )

        if not registry.is_registered("openai"):
            registry.register(OpenAIBatchProvider)

        assert registry.is_registered("openai") is True
        provider_class = registry.get("openai")
        assert provider_class == OpenAIBatchProvider

    def test_create_provider_instance_from_registry(self):
        """Test creating provider instance from registry."""
        # Ensure the provider is registered (it should be auto-registered on import)
        from inference_core.llm.batch.providers.openai_provider import (
            OpenAIBatchProvider,
        )

        if not registry.is_registered("openai"):
            registry.register(OpenAIBatchProvider)

        config = {"api_key": "test-key"}
        provider = registry.create_provider("openai", config)

        assert isinstance(provider, OpenAIBatchProvider)
        assert provider.get_provider_name() == "openai"
        assert provider.config == config

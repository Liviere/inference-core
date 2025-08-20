"""
Unit tests for Gemini and Claude batch providers

Tests the provider implementations to ensure they follow the interface
contract and handle provider-specific logic correctly.
"""

from datetime import datetime
from typing import List, Optional
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from app.llm.batch import (
    BaseBatchProvider,
    PreparedSubmission,
    ProviderPermanentError,
    ProviderResultRow,
    ProviderStatus,
    ProviderSubmitResult,
    ProviderTransientError,
)
from app.llm.batch.providers.claude_provider import ClaudeBatchProvider
from app.llm.batch.providers.gemini_provider import GeminiBatchProvider


class TestGeminiBatchProvider:
    """Test the GeminiBatchProvider implementation."""

    def setup_method(self):
        """Setup test environment."""
        self.config = {"api_key": "test_gemini_key"}

        # Mock the Google GenAI client
        with patch(
            "app.llm.batch.providers.gemini_provider.genai.Client"
        ) as mock_client_class:
            self.mock_client = Mock()
            mock_client_class.return_value = self.mock_client
            self.provider = GeminiBatchProvider(self.config)

    def test_provider_name(self):
        """Test provider name is set correctly."""
        assert self.provider.get_provider_name() == "gemini"
        assert self.provider.PROVIDER_NAME == "gemini"

    def test_initialization_without_api_key(self):
        """Test provider initialization fails without API key."""
        with pytest.raises(
            ProviderPermanentError, match="Google GenAI API key is required"
        ):
            GeminiBatchProvider({})

    def test_supports_model_valid_combinations(self):
        """Test supports_model returns True for valid combinations."""
        # Valid Gemini models
        assert self.provider.supports_model("gemini-2.5-flash", "chat") is True
        assert self.provider.supports_model("gemini-2.5-pro", "chat") is True
        assert (
            self.provider.supports_model("models/gemini-2.5-flash", "chat") is True
        )  # prefix variant
        assert self.provider.supports_model("gemini-2.0-flash", "chat") is True
        # Legacy models intentionally not supported in batch mode anymore:
        assert self.provider.supports_model("gemini-1.5-pro", "chat") is False
        assert self.provider.supports_model("gemini-1.5-flash", "chat") is False
        assert self.provider.supports_model("gemini-pro", "chat") is False

    def test_supports_model_invalid_combinations(self):
        """Test supports_model returns False for invalid combinations."""
        # Unsupported models
        assert self.provider.supports_model("gpt-4", "chat") is False
        assert self.provider.supports_model("claude-3", "chat") is False

        # Unsupported modes

        assert self.provider.supports_model("gemini-2.5-flash", "embedding") is False
        assert self.provider.supports_model("gemini-2.5-flash", "completion") is False

    def test_prepare_payloads_with_messages(self):
        """Test payload preparation with messages format."""
        batch_items = [
            {
                "id": "item_1",
                "input_payload": {"messages": [{"role": "user", "content": "Hello"}]},
            }
        ]

        result = self.provider.prepare_payloads(batch_items, "gemini-2.0-flash", "chat")

        assert isinstance(result, PreparedSubmission)
        assert result.provider == "gemini"
        assert result.model == "gemini-2.0-flash"
        assert result.mode == "chat"
        assert len(result.items) == 1

        item = result.items[0]
        assert "contents" in item
        assert len(item["contents"]) == 1
        assert item["contents"][0]["role"] == "user"
        assert item["contents"][0]["parts"][0]["text"] == "Hello"

    def test_prepare_payloads_with_content(self):
        """Test payload preparation with direct content."""
        batch_items = [{"id": "item_1", "input_payload": {"content": "Hello Gemini"}}]

        result = self.provider.prepare_payloads(batch_items, "gemini-2.0-flash", "chat")

        item = result.items[0]
        assert "contents" in item
        assert len(item["contents"]) == 1
        assert item["contents"][0]["role"] == "user"
        assert item["contents"][0]["parts"][0]["text"] == "Hello Gemini"

    def test_prepare_payloads_empty_items_error(self):
        """Test prepare_payloads raises error for empty items."""
        with pytest.raises(ProviderPermanentError, match="Batch items cannot be empty"):
            self.provider.prepare_payloads([], "gemini-2.0-flash", "chat")

    def test_prepare_payloads_unsupported_model_error(self):
        """Test prepare_payloads raises error for unsupported model."""
        batch_items = [{"id": "item_1", "input_payload": {"content": "test"}}]

        with pytest.raises(
            ProviderPermanentError, match="not supported by Gemini provider"
        ):
            self.provider.prepare_payloads(batch_items, "gpt-4", "chat")

    def test_submit_success(self):
        """Test successful batch submission."""
        # Mock batch job response
        mock_batch_job = Mock()
        mock_batch_job.name = "batch_123"
        mock_batch_job.state = "JOB_STATE_QUEUED"
        mock_batch_job.create_time = datetime.now()

        self.mock_client.batches.create.return_value = mock_batch_job

        # Prepare submission
        prepared = PreparedSubmission(
            batch_job_id=uuid4(),
            provider="gemini",
            model="gemini-2.0-flash",
            mode="chat",
            items=[
                {
                    "_custom_id": "item_1",
                    "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
                }
            ],
        )

        result = self.provider.submit(prepared)

        assert isinstance(result, ProviderSubmitResult)
        assert result.provider_batch_id == "batch_123"
        assert result.status == "JOB_STATE_QUEUED"
        assert result.item_count == 1

        # Verify client was called correctly
        self.mock_client.batches.create.assert_called_once()
        args, kwargs = self.mock_client.batches.create.call_args
        assert kwargs["model"] == "gemini-2.0-flash"

    def test_poll_status_success(self):
        """Test successful status polling."""
        # Mock batch job response
        mock_batch_job = Mock()
        mock_batch_job.state = "JOB_STATE_RUNNING"
        mock_batch_job.create_time = datetime.now()
        mock_batch_job.update_time = datetime.now()
        mock_batch_job.start_time = None
        mock_batch_job.end_time = None

        self.mock_client.batches.get.return_value = mock_batch_job

        result = self.provider.poll_status("batch_123")

        assert isinstance(result, ProviderStatus)
        assert result.provider_batch_id == "batch_123"
        assert result.status == "JOB_STATE_RUNNING"
        assert result.normalized_status == "in_progress"

        self.mock_client.batches.get.assert_called_once_with(name="batch_123")

    def test_status_mapping(self):
        """Test that Gemini statuses are mapped correctly."""
        test_cases = [
            ("JOB_STATE_QUEUED", "submitted"),
            ("JOB_STATE_RUNNING", "in_progress"),
            ("JOB_STATE_SUCCEEDED", "completed"),
            ("JOB_STATE_FAILED", "failed"),
            ("JOB_STATE_CANCELLED", "cancelled"),
        ]

        for gemini_status, expected_internal in test_cases:
            assert self.provider.STATUS_MAPPING[gemini_status] == expected_internal

    def test_cancel_success(self):
        """Test successful batch cancellation."""
        mock_result = Mock()
        mock_result.state = "JOB_STATE_CANCELLED"

        self.mock_client.batches.cancel.return_value = mock_result

        result = self.provider.cancel("batch_123")

        assert result is True
        self.mock_client.batches.cancel.assert_called_once_with(name="batch_123")


class TestClaudeBatchProvider:
    """Test the ClaudeBatchProvider implementation."""

    def setup_method(self):
        """Setup test environment."""
        self.config = {"api_key": "test_claude_key"}

        # Mock the Anthropic client
        with patch(
            "app.llm.batch.providers.claude_provider.Anthropic"
        ) as mock_client_class:
            self.mock_client = Mock()
            mock_client_class.return_value = self.mock_client
            self.provider = ClaudeBatchProvider(self.config)

    def test_provider_name(self):
        """Test provider name is set correctly."""
        assert self.provider.get_provider_name() == "claude"
        assert self.provider.PROVIDER_NAME == "claude"

    def test_initialization_without_api_key(self):
        """Test provider initialization fails without API key."""
        with pytest.raises(
            ProviderPermanentError, match="Anthropic API key is required"
        ):
            ClaudeBatchProvider({})

    def test_supports_model_valid_combinations(self):
        """Test supports_model returns True for valid combinations."""
        # Valid Claude models
        assert self.provider.supports_model("claude-3.5-sonnet", "chat") is True
        assert self.provider.supports_model("claude-3-opus", "chat") is True
        assert self.provider.supports_model("claude-3-haiku", "chat") is True
        assert self.provider.supports_model("claude-sonnet-4", "chat") is True

    def test_supports_model_invalid_combinations(self):
        """Test supports_model returns False for invalid combinations."""
        # Unsupported models
        assert self.provider.supports_model("gpt-4", "chat") is False
        assert self.provider.supports_model("gemini-pro", "chat") is False

        # Unsupported modes
        assert self.provider.supports_model("claude-3.5-sonnet", "embedding") is False
        assert self.provider.supports_model("claude-3.5-sonnet", "completion") is False

    def test_prepare_payloads_with_messages(self):
        """Test payload preparation with messages format."""
        batch_items = [
            {
                "id": "item_1",
                "input_payload": {
                    "messages": [{"role": "user", "content": "Hello Claude"}],
                    "max_tokens": 2048,
                },
            }
        ]

        result = self.provider.prepare_payloads(
            batch_items, "claude-3.5-sonnet", "chat"
        )

        assert isinstance(result, PreparedSubmission)
        assert result.provider == "claude"
        assert result.model == "claude-3.5-sonnet"
        assert result.mode == "chat"
        assert len(result.items) == 1

        item = result.items[0]
        assert item["custom_id"] == "item_1"
        assert item["params"]["model"] == "claude-3.5-sonnet"
        assert item["params"]["max_tokens"] == 2048
        assert len(item["params"]["messages"]) == 1
        assert item["params"]["messages"][0]["role"] == "user"
        assert item["params"]["messages"][0]["content"] == "Hello Claude"

    def test_prepare_payloads_with_content(self):
        """Test payload preparation with direct content."""
        batch_items = [{"id": "item_1", "input_payload": {"content": "Hello Claude"}}]

        result = self.provider.prepare_payloads(
            batch_items, "claude-3.5-sonnet", "chat"
        )

        item = result.items[0]
        assert item["custom_id"] == "item_1"
        assert len(item["params"]["messages"]) == 1
        assert item["params"]["messages"][0]["role"] == "user"
        assert item["params"]["messages"][0]["content"] == "Hello Claude"

    def test_prepare_payloads_with_optional_parameters(self):
        """Test payload preparation includes optional parameters."""
        batch_items = [
            {
                "id": "item_1",
                "input_payload": {
                    "content": "Hello Claude",
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 512,
                    "system": "You are a helpful assistant",
                },
            }
        ]

        result = self.provider.prepare_payloads(
            batch_items, "claude-3.5-sonnet", "chat"
        )

        item = result.items[0]
        params = item["params"]
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9
        assert params["max_tokens"] == 512
        assert params["system"] == "You are a helpful assistant"

    def test_submit_success(self):
        """Test successful batch submission."""
        # Mock batch response
        mock_batch = Mock()
        mock_batch.id = "batch_abc123"
        mock_batch.processing_status = "in_progress"
        mock_batch.created_at = datetime.now()
        mock_batch.expires_at = None

        self.mock_client.messages.batches.create.return_value = mock_batch

        # Prepare submission
        prepared = PreparedSubmission(
            batch_job_id=uuid4(),
            provider="claude",
            model="claude-3.5-sonnet",
            mode="chat",
            items=[
                {
                    "custom_id": "item_1",
                    "params": {
                        "model": "claude-3.5-sonnet",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                }
            ],
        )

        result = self.provider.submit(prepared)

        assert isinstance(result, ProviderSubmitResult)
        assert result.provider_batch_id == "batch_abc123"
        assert result.status == "in_progress"
        assert result.item_count == 1

        # Verify client was called correctly
        self.mock_client.messages.batches.create.assert_called_once()

    def test_poll_status_success(self):
        """Test successful status polling."""
        # Mock batch response
        mock_batch = Mock()
        mock_batch.id = "batch_abc123"
        mock_batch.processing_status = "ended"
        mock_batch.created_at = datetime.now()
        mock_batch.expires_at = datetime.now()
        mock_batch.request_counts = Mock()
        mock_batch.request_counts.processing = 0
        mock_batch.request_counts.succeeded = 5
        mock_batch.request_counts.errored = 0

        # Mock the list method to return our batch
        mock_page = Mock()
        mock_page.data = [mock_batch]
        self.mock_client.messages.batches.list.return_value = mock_page

        # Also mock retrieve in case it exists
        self.mock_client.messages.batches.retrieve = Mock(return_value=mock_batch)

        result = self.provider.poll_status("batch_abc123")

        assert isinstance(result, ProviderStatus)
        assert result.provider_batch_id == "batch_abc123"
        assert result.status == "ended"
        assert result.normalized_status == "completed"
        assert result.progress_info["request_counts"]["succeeded"] == 5

    def test_status_mapping(self):
        """Test that Claude statuses are mapped correctly."""
        test_cases = [
            ("in_progress", "in_progress"),
            ("ended", "completed"),
            ("errored", "failed"),
            ("expired", "failed"),
            ("canceling", "cancelled"),
        ]

        for claude_status, expected_internal in test_cases:
            assert self.provider.STATUS_MAPPING[claude_status] == expected_internal

    def test_fetch_results_success(self):
        """Test successful result fetching."""
        # Mock successful result
        mock_entry = Mock()
        mock_entry.custom_id = "item_1"
        mock_entry.result = Mock()
        mock_entry.result.type = "succeeded"
        mock_entry.result.message = Mock()
        mock_entry.result.message.id = "msg_123"
        mock_entry.result.message.type = "message"
        mock_entry.result.message.role = "assistant"
        mock_entry.result.message.model = "claude-3.5-sonnet"
        mock_entry.result.message.stop_reason = "end_turn"
        mock_entry.result.message.stop_sequence = None
        mock_entry.result.message.usage = None

        # Mock content with text
        mock_content = Mock()
        mock_content.text = "Hello! How can I help you?"
        mock_entry.result.message.content = [mock_content]

        # Mock status polling to return completed
        with patch.object(self.provider, "poll_status") as mock_poll:
            mock_status = Mock()
            mock_status.normalized_status = "completed"
            mock_poll.return_value = mock_status

            # Mock results stream
            self.mock_client.messages.batches.results.return_value = [mock_entry]

            results = self.provider.fetch_results("batch_abc123")

            assert len(results) == 1
            result = results[0]
            assert isinstance(result, ProviderResultRow)
            assert result.custom_id == "item_1"
            assert result.output_text == "Hello! How can I help you?"
            assert result.is_success is True
            assert result.error_message is None

    def test_fetch_results_sanitizes_complex_objects(self):
        """Ensure fetch_results produces JSON-serializable output_data for complex SDK objects."""
        # Arrange: mock status polling to 'completed'
        with patch.object(self.provider, "poll_status") as mock_poll:
            mock_poll.return_value = ProviderStatus(
                provider_batch_id="batch_123",
                status="ended",
                normalized_status="completed",
                progress_info={},
                estimated_completion=None,
            )

            # Build complex content block (simulate SDK object)
            content_block = Mock()
            content_block.text = "Hello sanitized"
            content_block.type = "text"
            content_block.extra_field = object()  # Non-serializable

            usage_obj = Mock()
            usage_obj.input_tokens = 10
            usage_obj.output_tokens = 20
            usage_obj.nested = Mock()
            usage_obj.nested.foo = "bar"

            message = Mock()
            message.id = "msg_1"
            message.type = "message"
            message.role = "assistant"
            message.model = "claude-3.5-sonnet"
            message.content = [content_block]
            message.stop_reason = "end_turn"
            message.stop_sequence = None
            message.usage = usage_obj

            entry = Mock()
            entry.custom_id = "item_1"
            entry.result = Mock()
            entry.result.type = "succeeded"
            entry.result.message = message

            # Mock results stream
            self.mock_client.messages.batches.results.return_value = [entry]

            rows = self.provider.fetch_results("batch_123")
            assert len(rows) == 1
            row = rows[0]
            assert row.is_success is True
            assert row.output_text == "Hello sanitized"
            import json as _json

            # Should be JSON serializable
            _json.dumps(row.output_data)


class TestProviderIntegration:
    """Test integration between new providers and the registry."""

    def test_providers_can_be_registered(self):
        """Test that new providers can be registered in the registry."""
        from app.llm.batch.registry import BatchProviderRegistry

        # Create a fresh registry for testing
        test_registry = BatchProviderRegistry()

        # Register providers manually
        test_registry.register(GeminiBatchProvider)
        test_registry.register(ClaudeBatchProvider)

        # Check that providers are registered
        registered_providers = test_registry.list()
        assert "gemini" in registered_providers
        assert "claude" in registered_providers

    def test_providers_can_be_created_from_registry(self):
        """Test that providers can be created from the registry."""
        from app.llm.batch.registry import BatchProviderRegistry

        # Create a fresh registry for testing
        test_registry = BatchProviderRegistry()

        # Register providers
        test_registry.register(GeminiBatchProvider)
        test_registry.register(ClaudeBatchProvider)

        # Test Gemini provider creation
        with patch("app.llm.batch.providers.gemini_provider.genai.Client"):
            gemini_provider = test_registry.create_provider(
                "gemini", {"api_key": "test"}
            )
            assert isinstance(gemini_provider, GeminiBatchProvider)

        # Test Claude provider creation
        with patch("app.llm.batch.providers.claude_provider.Anthropic"):
            claude_provider = test_registry.create_provider(
                "claude", {"api_key": "test"}
            )
            assert isinstance(claude_provider, ClaudeBatchProvider)

    def test_provider_error_handling_patterns(self):
        """Test that providers follow consistent error handling patterns."""
        # Test providers raise appropriate errors for missing API keys
        with pytest.raises(ProviderPermanentError):
            GeminiBatchProvider({})

        with pytest.raises(ProviderPermanentError):
            ClaudeBatchProvider({})

    def test_provider_model_support_coverage(self):
        """Test that providers support expected model patterns."""
        # Mock clients to avoid initialization errors
        with patch("app.llm.batch.providers.gemini_provider.genai.Client"):
            gemini = GeminiBatchProvider({"api_key": "test"})

        with patch("app.llm.batch.providers.claude_provider.Anthropic"):
            claude = ClaudeBatchProvider({"api_key": "test"})

        # Test Gemini models
        assert gemini.supports_model("gemini-2.0-flash", "chat")
        assert not gemini.supports_model("claude-3.5-sonnet", "chat")

        # Test Claude models
        assert claude.supports_model("claude-3.5-sonnet", "chat")
        assert not claude.supports_model("gemini-2.0-flash", "chat")

        # Both should reject unsupported modes
        assert not gemini.supports_model("gemini-2.0-flash", "embedding")
        assert not claude.supports_model("claude-3.5-sonnet", "embedding")

"""
Integration tests for Gemini and Claude batch providers

Tests the providers with mocked HTTP requests to simulate real API interactions.
Covers success and failure scenarios as specified in acceptance criteria.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import json

from app.llm.batch.providers.gemini_provider import GeminiBatchProvider
from app.llm.batch.providers.claude_provider import ClaudeBatchProvider
from app.llm.batch import (
    ProviderTransientError,
    ProviderPermanentError,
    ProviderResultRow
)


class TestGeminiProviderIntegration:
    """Integration tests for Gemini provider with mocked HTTP requests."""

    def setup_method(self):
        """Setup test environment with mocked client."""
        with patch('app.llm.batch.providers.gemini_provider.genai.Client') as mock_client_class:
            self.mock_client = Mock()
            mock_client_class.return_value = self.mock_client
            self.provider = GeminiBatchProvider({"api_key": "test_key"})

    def test_complete_batch_workflow_success(self):
        """Test complete batch workflow from submission to results."""
        # Test data
        batch_items = [
            {
                "id": "item_1",
                "input_payload": {
                    "messages": [{"role": "user", "content": "Hello Gemini"}]
                }
            },
            {
                "id": "item_2", 
                "input_payload": {
                    "content": "How are you?"
                }
            }
        ]

        # 1. Prepare payloads
        prepared = self.provider.prepare_payloads(batch_items, "gemini-2.0-flash", "chat")
        assert len(prepared.items) == 2
        assert prepared.provider == "gemini"

        # 2. Mock batch submission
        mock_batch = Mock()
        mock_batch.name = "batch_gemini_123"
        mock_batch.state = "JOB_STATE_QUEUED"
        mock_batch.create_time = datetime.now()
        
        self.mock_client.batches.create.return_value = mock_batch
        
        # Submit batch
        submit_result = self.provider.submit(prepared)
        assert submit_result.provider_batch_id == "batch_gemini_123"
        assert submit_result.status == "JOB_STATE_QUEUED"
        assert submit_result.item_count == 2

        # 3. Mock status polling (in progress)
        mock_batch_running = Mock()
        mock_batch_running.state = "JOB_STATE_RUNNING"
        mock_batch_running.create_time = datetime.now()
        mock_batch_running.update_time = datetime.now()
        mock_batch_running.start_time = datetime.now()
        mock_batch_running.end_time = None
        
        self.mock_client.batches.get.return_value = mock_batch_running
        
        status = self.provider.poll_status("batch_gemini_123")
        assert status.normalized_status == "in_progress"

        # 4. Mock completion and results
        mock_batch_completed = Mock()
        mock_batch_completed.state = "JOB_STATE_SUCCEEDED"
        mock_batch_completed.create_time = datetime.now()
        mock_batch_completed.update_time = datetime.now()
        mock_batch_completed.start_time = datetime.now()
        mock_batch_completed.end_time = datetime.now()
        mock_batch_completed.dest = Mock()
        mock_batch_completed.dest.inlined_responses = [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Hello! Nice to meet you."}]
                        }
                    }
                ]
            },
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "I'm doing well, thank you!"}]
                        }
                    }
                ]
            }
        ]
        
        self.mock_client.batches.get.return_value = mock_batch_completed
        
        # Check final status
        final_status = self.provider.poll_status("batch_gemini_123")
        assert final_status.normalized_status == "completed"
        
        # Fetch results
        results = self.provider.fetch_results("batch_gemini_123")
        assert len(results) == 2
        assert all(isinstance(r, ProviderResultRow) for r in results)
        assert all(r.is_success for r in results)
        assert results[0].output_text == "Hello! Nice to meet you."
        assert results[1].output_text == "I'm doing well, thank you!"

    def test_batch_failure_scenario(self):
        """Test batch failure handling."""
        batch_items = [{"id": "item_1", "input_payload": {"content": "Test"}}]
        prepared = self.provider.prepare_payloads(batch_items, "gemini-2.0-flash", "chat")

        # Mock failed batch
        mock_batch = Mock()
        mock_batch.name = "batch_gemini_failed"
        mock_batch.state = "JOB_STATE_FAILED"
        mock_batch.create_time = datetime.now()
        mock_batch.error = "Rate limit exceeded"
        
        self.mock_client.batches.create.return_value = mock_batch
        self.mock_client.batches.get.return_value = mock_batch
        
        # Submit and check status
        submit_result = self.provider.submit(prepared)
        status = self.provider.poll_status("batch_gemini_failed")
        
        assert status.normalized_status == "failed"
        assert "error" in status.progress_info

    def test_transient_error_handling(self):
        """Test handling of transient errors (rate limits)."""
        batch_items = [{"id": "item_1", "input_payload": {"content": "Test"}}]
        prepared = self.provider.prepare_payloads(batch_items, "gemini-2.0-flash", "chat")

        # Mock rate limit error
        self.mock_client.batches.create.side_effect = Exception("Rate limit exceeded - retry after 60 seconds")
        
        with pytest.raises(ProviderTransientError) as exc_info:
            self.provider.submit(prepared)
        
        assert "transient" in str(exc_info.value)
        assert exc_info.value.retry_after == 60

    def test_permanent_error_handling(self):
        """Test handling of permanent errors (auth, invalid model)."""
        batch_items = [{"id": "item_1", "input_payload": {"content": "Test"}}]
        prepared = self.provider.prepare_payloads(batch_items, "gemini-2.0-flash", "chat")

        # Mock auth error
        self.mock_client.batches.create.side_effect = Exception("Unauthorized - invalid API key")
        
        with pytest.raises(ProviderPermanentError) as exc_info:
            self.provider.submit(prepared)
        
        assert "permanent" in str(exc_info.value)


class TestClaudeProviderIntegration:
    """Integration tests for Claude provider with mocked HTTP requests."""

    def setup_method(self):
        """Setup test environment with mocked client."""
        with patch('app.llm.batch.providers.claude_provider.Anthropic') as mock_client_class:
            self.mock_client = Mock()
            mock_client_class.return_value = self.mock_client
            self.provider = ClaudeBatchProvider({"api_key": "test_key"})

    def test_complete_batch_workflow_success(self):
        """Test complete batch workflow from submission to results."""
        # Test data
        batch_items = [
            {
                "id": "item_1",
                "input_payload": {
                    "messages": [{"role": "user", "content": "Hello Claude"}],
                    "max_tokens": 1024
                }
            },
            {
                "id": "item_2",
                "input_payload": {
                    "content": "How are you today?",
                    "temperature": 0.7
                }
            }
        ]

        # 1. Prepare payloads
        prepared = self.provider.prepare_payloads(batch_items, "claude-3.5-sonnet", "chat")
        assert len(prepared.items) == 2
        assert prepared.provider == "claude"
        
        # Check Claude-specific formatting
        assert prepared.items[0]["custom_id"] == "item_1"
        assert prepared.items[0]["params"]["model"] == "claude-3.5-sonnet"
        assert prepared.items[0]["params"]["max_tokens"] == 1024
        assert prepared.items[1]["params"]["temperature"] == 0.7

        # 2. Mock batch submission
        mock_batch = Mock()
        mock_batch.id = "batch_claude_123"
        mock_batch.processing_status = "in_progress"
        mock_batch.created_at = datetime.now()
        mock_batch.expires_at = None
        
        self.mock_client.messages.batches.create.return_value = mock_batch
        
        # Submit batch
        submit_result = self.provider.submit(prepared)
        assert submit_result.provider_batch_id == "batch_claude_123"
        assert submit_result.status == "in_progress"
        assert submit_result.item_count == 2

        # 3. Mock status polling (completed)
        mock_batch_completed = Mock()
        mock_batch_completed.id = "batch_claude_123"
        mock_batch_completed.processing_status = "ended"
        mock_batch_completed.created_at = datetime.now()
        mock_batch_completed.expires_at = datetime.now()
        mock_batch_completed.request_counts = Mock()
        mock_batch_completed.request_counts.processing = 0
        mock_batch_completed.request_counts.succeeded = 2
        mock_batch_completed.request_counts.errored = 0
        mock_batch_completed.request_counts.canceled = 0
        mock_batch_completed.request_counts.expired = 0
        
        # Mock list method for polling
        mock_page = Mock()
        mock_page.data = [mock_batch_completed]
        self.mock_client.messages.batches.list.return_value = mock_page
        
        # Also mock retrieve in case it exists
        self.mock_client.messages.batches.retrieve = Mock(return_value=mock_batch_completed)
        
        status = self.provider.poll_status("batch_claude_123")
        assert status.normalized_status == "completed"
        assert status.progress_info["request_counts"]["succeeded"] == 2

        # 4. Mock results fetching
        mock_entry1 = Mock()
        mock_entry1.custom_id = "item_1"
        mock_entry1.result = Mock()
        mock_entry1.result.type = "succeeded"
        mock_entry1.result.message = Mock()
        mock_entry1.result.message.id = "msg_1"
        mock_entry1.result.message.type = "message"
        mock_entry1.result.message.role = "assistant"
        mock_entry1.result.message.model = "claude-3.5-sonnet"
        mock_entry1.result.message.stop_reason = "end_turn"
        mock_entry1.result.message.stop_sequence = None
        mock_entry1.result.message.usage = None
        
        mock_content1 = Mock()
        mock_content1.text = "Hello! It's nice to meet you."
        mock_entry1.result.message.content = [mock_content1]
        
        mock_entry2 = Mock()
        mock_entry2.custom_id = "item_2"
        mock_entry2.result = Mock()
        mock_entry2.result.type = "succeeded"
        mock_entry2.result.message = Mock()
        mock_entry2.result.message.id = "msg_2"
        mock_entry2.result.message.type = "message"
        mock_entry2.result.message.role = "assistant"
        mock_entry2.result.message.model = "claude-3.5-sonnet"
        mock_entry2.result.message.stop_reason = "end_turn"
        mock_entry2.result.message.stop_sequence = None
        mock_entry2.result.message.usage = None
        
        mock_content2 = Mock()
        mock_content2.text = "I'm doing very well today, thank you for asking!"
        mock_entry2.result.message.content = [mock_content2]
        
        # Mock poll_status to return completed for fetch_results
        with patch.object(self.provider, 'poll_status') as mock_poll:
            mock_status = Mock()
            mock_status.normalized_status = "completed"
            mock_poll.return_value = mock_status
            
            self.mock_client.messages.batches.results.return_value = [mock_entry1, mock_entry2]
            
            results = self.provider.fetch_results("batch_claude_123")
            assert len(results) == 2
            assert all(isinstance(r, ProviderResultRow) for r in results)
            assert all(r.is_success for r in results)
            assert results[0].custom_id == "item_1"
            assert results[0].output_text == "Hello! It's nice to meet you."
            assert results[1].custom_id == "item_2"
            assert results[1].output_text == "I'm doing very well today, thank you for asking!"

    def test_partial_failure_scenario(self):
        """Test batch with some successful and some failed items."""
        # Mock status polling to return completed
        with patch.object(self.provider, 'poll_status') as mock_poll:
            mock_status = Mock()
            mock_status.normalized_status = "completed"
            mock_poll.return_value = mock_status
            
            # Mock mixed results (success and error)
            mock_success_entry = Mock()
            mock_success_entry.custom_id = "item_1"
            mock_success_entry.result = Mock()
            mock_success_entry.result.type = "succeeded"
            mock_success_entry.result.message = Mock()
            mock_success_entry.result.message.content = [Mock(text="Success response")]
            mock_success_entry.result.message.id = "msg_1"
            mock_success_entry.result.message.type = "message"
            mock_success_entry.result.message.role = "assistant"
            mock_success_entry.result.message.model = "claude-3.5-sonnet"
            mock_success_entry.result.message.stop_reason = "end_turn"
            mock_success_entry.result.message.stop_sequence = None
            mock_success_entry.result.message.usage = None
            
            mock_error_entry = Mock()
            mock_error_entry.custom_id = "item_2"
            mock_error_entry.result = Mock()
            mock_error_entry.result.type = "errored"
            mock_error_entry.result.error = Mock()
            mock_error_entry.result.error.type = "invalid_request_error"
            mock_error_entry.result.error.message = "Token limit exceeded"
            
            self.mock_client.messages.batches.results.return_value = [mock_success_entry, mock_error_entry]
            
            results = self.provider.fetch_results("batch_claude_123")
            assert len(results) == 2
            
            # Check success result
            success_result = next(r for r in results if r.custom_id == "item_1")
            assert success_result.is_success is True
            assert success_result.output_text == "Success response"
            assert success_result.error_message is None
            
            # Check error result
            error_result = next(r for r in results if r.custom_id == "item_2")
            assert error_result.is_success is False
            assert error_result.output_text is None
            assert "invalid_request_error" in error_result.error_message

    def test_transient_error_handling(self):
        """Test handling of transient errors."""
        batch_items = [{"id": "item_1", "input_payload": {"content": "Test"}}]
        prepared = self.provider.prepare_payloads(batch_items, "claude-3.5-sonnet", "chat")

        # Mock rate limit error
        self.mock_client.messages.batches.create.side_effect = Exception("Rate limit exceeded")
        
        with pytest.raises(ProviderTransientError) as exc_info:
            self.provider.submit(prepared)
        
        assert "transient" in str(exc_info.value)

    def test_permanent_error_handling(self):
        """Test handling of permanent errors."""
        batch_items = [{"id": "item_1", "input_payload": {"content": "Test"}}]
        prepared = self.provider.prepare_payloads(batch_items, "claude-3.5-sonnet", "chat")

        # Mock auth error
        self.mock_client.messages.batches.create.side_effect = Exception("Unauthorized access")
        
        with pytest.raises(ProviderPermanentError) as exc_info:
            self.provider.submit(prepared)
        
        assert "permanent" in str(exc_info.value)


class TestProviderConfigToggling:
    """Test config toggling to disable providers."""

    def test_disabled_provider_prevents_submission(self):
        """Test that disabled providers prevent submission."""
        # This would typically be tested through configuration
        # For now, we test that missing API keys prevent initialization
        
        with pytest.raises(ProviderPermanentError, match="API key is required"):
            GeminiBatchProvider({})
        
        with pytest.raises(ProviderPermanentError, match="API key is required"):
            ClaudeBatchProvider({})

    def test_provider_availability_based_on_dependencies(self):
        """Test that providers are only available when dependencies are installed."""
        # Test with fresh registry to avoid conflicts with cleared global registry
        from app.llm.batch.registry import BatchProviderRegistry
        from app.llm.batch.providers.gemini_provider import GeminiBatchProvider
        from app.llm.batch.providers.claude_provider import ClaudeBatchProvider
        from app.llm.batch.providers.openai_provider import OpenAIBatchProvider
        
        test_registry = BatchProviderRegistry()
        test_registry.register(GeminiBatchProvider)
        test_registry.register(ClaudeBatchProvider)
        test_registry.register(OpenAIBatchProvider)
        
        providers = test_registry.list()
        assert "gemini" in providers
        assert "claude" in providers
        assert "openai" in providers


class TestCrossProviderCompatibility:
    """Test compatibility between providers."""

    def test_providers_reject_incompatible_models(self):
        """Test that providers properly reject models from other providers."""
        with patch('app.llm.batch.providers.gemini_provider.genai.Client'):
            gemini = GeminiBatchProvider({"api_key": "test"})
        
        with patch('app.llm.batch.providers.claude_provider.Anthropic'):
            claude = ClaudeBatchProvider({"api_key": "test"})
        
        # Gemini should reject Claude models
        assert not gemini.supports_model("claude-3.5-sonnet", "chat")
        assert not gemini.supports_model("claude-opus", "chat")
        
        # Claude should reject Gemini models
        assert not claude.supports_model("gemini-2.0-flash", "chat")
        assert not claude.supports_model("gemini-pro", "chat")
        
        # Both should reject OpenAI models
        assert not gemini.supports_model("gpt-4", "chat")
        assert not claude.supports_model("gpt-4", "chat")

    def test_consistent_error_handling_patterns(self):
        """Test that all providers use consistent error handling patterns."""
        with patch('app.llm.batch.providers.gemini_provider.genai.Client'):
            gemini = GeminiBatchProvider({"api_key": "test"})
        
        with patch('app.llm.batch.providers.claude_provider.Anthropic'):
            claude = ClaudeBatchProvider({"api_key": "test"})
        
        # Both should raise ProviderPermanentError for empty batch items
        with pytest.raises(ProviderPermanentError, match="Batch items cannot be empty"):
            gemini.prepare_payloads([], "gemini-2.0-flash", "chat")
        
        with pytest.raises(ProviderPermanentError, match="Batch items cannot be empty"):
            claude.prepare_payloads([], "claude-3.5-sonnet", "chat")
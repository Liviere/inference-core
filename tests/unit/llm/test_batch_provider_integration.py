"""
Integration test demonstrating the provider registry with example providers.
"""

import pytest
from uuid import uuid4

from app.llm.batch import BatchMode, batch_provider_registry
from app.llm.batch.providers.openai_provider import OpenAIBatchProvider


class TestProviderRegistryIntegration:
    """Test provider registry with example providers"""
    
    def setup_method(self):
        """Clean registry before each test"""
        batch_provider_registry.clear()
    
    def teardown_method(self):
        """Clean registry after each test"""
        batch_provider_registry.clear()
    
    def test_register_openai_provider(self):
        """Test registering the OpenAI example provider"""
        batch_provider_registry.register(OpenAIBatchProvider)
        
        assert batch_provider_registry.is_registered("openai")
        assert "openai" in batch_provider_registry.list()
        
        provider_class = batch_provider_registry.get("openai")
        assert provider_class == OpenAIBatchProvider
    
    def test_instantiate_provider_from_registry(self):
        """Test creating provider instance from registry"""
        batch_provider_registry.register(OpenAIBatchProvider)
        
        provider_class = batch_provider_registry.get("openai")
        provider = provider_class(config={"api_key": "test-key"})
        
        assert isinstance(provider, OpenAIBatchProvider)
        assert provider.PROVIDER_NAME == "openai"
        assert provider.config["api_key"] == "test-key"
    
    def test_openai_provider_supports_models(self):
        """Test OpenAI provider model support"""
        provider = OpenAIBatchProvider()
        
        # Should support these models
        assert provider.supports_model("gpt-4o-mini") is True
        assert provider.supports_model("gpt-4") is True
        assert provider.supports_model("gpt-3.5-turbo") is True
        
        # Should not support these
        assert provider.supports_model("claude-3-haiku") is False
        assert provider.supports_model("gemini-pro") is False
    
    def test_openai_provider_prepare_chat_payloads(self):
        """Test OpenAI provider payload preparation for chat"""
        provider = OpenAIBatchProvider()
        batch_id = uuid4()
        requests = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "temperature": 0.7
            },
            {
                "messages": [
                    {"role": "user", "content": "World"}
                ],
                "max_tokens": 100
            }
        ]
        
        result = provider.prepare_payloads(
            batch_id=batch_id,
            model="gpt-4o-mini",
            mode=BatchMode.CHAT,
            requests=requests
        )
        
        assert result.batch_id == batch_id
        assert result.provider_name == "openai"
        assert result.model == "gpt-4o-mini"
        assert result.mode == "chat"
        assert len(result.payloads) == 2
        
        # Check first payload structure
        payload1 = result.payloads[0]
        assert payload1["custom_id"] == "request_0"
        assert payload1["method"] == "POST"
        assert payload1["url"] == "/v1/chat/completions"
        assert payload1["body"]["model"] == "gpt-4o-mini"
        assert payload1["body"]["messages"] == [{"role": "user", "content": "Hello"}]
        assert payload1["body"]["temperature"] == 0.7
        
        # Check second payload structure
        payload2 = result.payloads[1]
        assert payload2["custom_id"] == "request_1"
        assert payload2["body"]["max_tokens"] == 100
    
    def test_openai_provider_prepare_completion_payloads(self):
        """Test OpenAI provider payload preparation for completion"""
        provider = OpenAIBatchProvider()
        batch_id = uuid4()
        requests = [
            {
                "prompt": "Complete this: Hello",
                "max_tokens": 50
            }
        ]
        
        result = provider.prepare_payloads(
            batch_id=batch_id,
            model="gpt-3.5-turbo",
            mode=BatchMode.COMPLETION,
            requests=requests
        )
        
        assert result.mode == BatchMode.COMPLETION
        payload = result.payloads[0]
        assert payload["url"] == "/v1/completions"
        assert payload["body"]["prompt"] == "Complete this: Hello"
        assert payload["body"]["max_tokens"] == 50
    
    def test_openai_provider_unsupported_mode(self):
        """Test OpenAI provider with unsupported mode"""
        provider = OpenAIBatchProvider()
        
        with pytest.raises(ValueError, match="Unsupported mode for OpenAI: BatchMode.EMBEDDING"):
            provider.prepare_payloads(
                batch_id=uuid4(),
                model="gpt-4",
                mode=BatchMode.EMBEDDING,
                requests=[{"input": "test"}]
            )
    
    @pytest.mark.asyncio
    async def test_openai_provider_workflow(self):
        """Test complete OpenAI provider workflow"""
        provider = OpenAIBatchProvider()
        batch_id = uuid4()
        
        # Prepare payloads
        prepared = provider.prepare_payloads(
            batch_id=batch_id,
            model="gpt-4o-mini",
            mode=BatchMode.CHAT,
            requests=[{"messages": [{"role": "user", "content": "Test"}]}]
        )
        
        # Submit batch
        submit_result = await provider.submit(prepared)
        assert submit_result.provider_batch_id.startswith("batch_openai_")
        assert submit_result.status == "validating"
        
        # Poll status
        status = await provider.poll_status(submit_result.provider_batch_id)
        assert status.provider_batch_id == submit_result.provider_batch_id
        assert status.status == "completed"
        
        # Fetch results
        results = await provider.fetch_results(submit_result.provider_batch_id)
        assert len(results) == 1
        assert results[0].request_id == "request_0"
        assert results[0].status == "success"
        
        # Cancel (should work even if completed)
        cancel_success = await provider.cancel(submit_result.provider_batch_id)
        assert cancel_success is True
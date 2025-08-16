"""
Test OpenAI Batch Provider with configuration

Tests that the OpenAI provider works with the actual configuration setup.
"""

import pytest
from unittest.mock import patch
import os

from app.llm.batch import registry


class TestOpenAIProviderConfiguration:
    """Test OpenAI provider with real configuration setup."""
    
    def test_provider_can_be_created_with_env_config(self):
        """Test creating OpenAI provider with environment configuration."""
        # Mock environment variable
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-from-env"}):
            config = {"api_key": os.environ.get("OPENAI_API_KEY")}
            
            # Create provider instance
            provider = registry.create_provider("openai", config)
            
            assert provider is not None
            assert provider.get_provider_name() == "openai"
            assert provider.config["api_key"] == "test-key-from-env"
    
    def test_provider_supports_configured_models(self):
        """Test that provider supports models from configuration."""
        config = {"api_key": "test-key"}
        provider = registry.create_provider("openai", config)
        
        # Test models from llm_config.yaml batch section
        assert provider.supports_model("gpt-5-mini", "chat") is True
        assert provider.supports_model("gpt-4", "chat") is True
        assert provider.supports_model("gpt-3.5-turbo", "chat") is True
        
        # Test unsupported combinations
        assert provider.supports_model("gpt-4", "embedding") is False
        assert provider.supports_model("claude-3", "chat") is False
    
    def test_provider_in_global_registry(self):
        """Test that OpenAI provider is available in global registry."""
        assert registry.is_registered("openai") is True
        
        available_providers = registry.list()
        assert "openai" in available_providers
        
        # Should be able to get the provider class
        provider_class = registry.get("openai")
        assert provider_class.__name__ == "OpenAIBatchProvider"
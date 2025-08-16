"""
Unit tests for batch configuration in LLM module

Tests validation, parsing, and helper methods for batch processing configuration.
"""

import pytest
from unittest.mock import patch, mock_open
import yaml
from pydantic import ValidationError

from app.llm.config import (
    BatchConfig,
    BatchRetryConfig,
    BatchModelConfig,
    BatchProviderConfig,
    BatchDefaultsConfig,
    LLMConfig,
)


class TestBatchRetryConfig:
    """Test BatchRetryConfig validation"""

    def test_valid_retry_config(self):
        """Test valid retry configuration"""
        config = BatchRetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0
        )
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0

    def test_default_retry_config(self):
        """Test default retry configuration values"""
        config = BatchRetryConfig()
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 60.0

    def test_invalid_max_attempts_bounds(self):
        """Test validation of max_attempts bounds"""
        with pytest.raises(ValidationError):
            BatchRetryConfig(max_attempts=0)
        
        with pytest.raises(ValidationError):
            BatchRetryConfig(max_attempts=25)

    def test_invalid_delay_bounds(self):
        """Test validation of delay bounds"""
        with pytest.raises(ValidationError):
            BatchRetryConfig(base_delay=0.05)
        
        with pytest.raises(ValidationError):
            BatchRetryConfig(max_delay=700.0)

    def test_max_delay_less_than_base_delay(self):
        """Test validation that max_delay >= base_delay"""
        with pytest.raises(ValidationError, match="max_delay must be greater than or equal to base_delay"):
            BatchRetryConfig(base_delay=30.0, max_delay=10.0)


class TestBatchModelConfig:
    """Test BatchModelConfig validation"""

    def test_valid_model_config(self):
        """Test valid model configuration"""
        config = BatchModelConfig(
            name="gpt-4o-mini",
            mode="chat",
            max_prompts_per_batch=50,
            pricing_tier="batch"
        )
        assert config.name == "gpt-4o-mini"
        assert config.mode == "chat"
        assert config.max_prompts_per_batch == 50
        assert config.pricing_tier == "batch"

    def test_default_model_config(self):
        """Test default model configuration values"""
        config = BatchModelConfig(name="test-model", mode="chat")
        assert config.max_prompts_per_batch == 100
        assert config.pricing_tier is None

    def test_invalid_mode(self):
        """Test validation of processing mode"""
        with pytest.raises(ValidationError, match="mode must be one of: chat, embedding, completion, custom"):
            BatchModelConfig(name="test-model", mode="invalid_mode")

    def test_valid_modes(self):
        """Test all valid processing modes"""
        valid_modes = ["chat", "embedding", "completion", "custom"]
        for mode in valid_modes:
            config = BatchModelConfig(name="test-model", mode=mode)
            assert config.mode == mode

    def test_invalid_batch_size_bounds(self):
        """Test validation of max_prompts_per_batch bounds"""
        with pytest.raises(ValidationError):
            BatchModelConfig(name="test-model", mode="chat", max_prompts_per_batch=0)
        
        with pytest.raises(ValidationError):
            BatchModelConfig(name="test-model", mode="chat", max_prompts_per_batch=1500)


class TestBatchProviderConfig:
    """Test BatchProviderConfig validation"""

    def test_valid_provider_config(self):
        """Test valid provider configuration"""
        models = [
            BatchModelConfig(name="model1", mode="chat"),
            BatchModelConfig(name="model2", mode="embedding")
        ]
        config = BatchProviderConfig(enabled=True, models=models)
        assert config.enabled is True
        assert len(config.models) == 2

    def test_default_provider_config(self):
        """Test default provider configuration values"""
        config = BatchProviderConfig()
        assert config.enabled is True
        assert config.models == []


class TestBatchConfig:
    """Test BatchConfig validation and helper methods"""

    def test_valid_batch_config(self):
        """Test valid batch configuration"""
        providers = {
            "openai": BatchProviderConfig(
                enabled=True,
                models=[BatchModelConfig(name="gpt-4o-mini", mode="chat")]
            )
        }
        config = BatchConfig(
            enabled=True,
            default_poll_interval_seconds=60,
            max_concurrent_provider_polls=10,
            providers=providers
        )
        assert config.enabled is True
        assert config.default_poll_interval_seconds == 60
        assert config.max_concurrent_provider_polls == 10

    def test_default_batch_config(self):
        """Test default batch configuration values"""
        config = BatchConfig()
        assert config.enabled is True
        assert config.default_poll_interval_seconds == 30
        assert config.max_concurrent_provider_polls == 5
        assert isinstance(config.defaults, BatchDefaultsConfig)
        assert config.providers == {}

    def test_get_provider_models(self):
        """Test get_provider_models helper method"""
        models = [
            BatchModelConfig(name="gpt-4o-mini", mode="chat"),
            BatchModelConfig(name="gpt-4", mode="chat")
        ]
        providers = {
            "openai": BatchProviderConfig(enabled=True, models=models),
            "claude": BatchProviderConfig(enabled=False, models=[])
        }
        config = BatchConfig(providers=providers)
        
        # Test enabled provider
        openai_models = config.get_provider_models("openai")
        assert len(openai_models) == 2
        assert openai_models[0].name == "gpt-4o-mini"
        assert openai_models[1].name == "gpt-4"
        
        # Test disabled provider
        claude_models = config.get_provider_models("claude")
        assert len(claude_models) == 0
        
        # Test non-existent provider
        unknown_models = config.get_provider_models("unknown")
        assert len(unknown_models) == 0

    def test_is_provider_enabled(self):
        """Test is_provider_enabled helper method"""
        providers = {
            "openai": BatchProviderConfig(enabled=True, models=[]),
            "claude": BatchProviderConfig(enabled=False, models=[])
        }
        config = BatchConfig(enabled=True, providers=providers)
        
        assert config.is_provider_enabled("openai") is True
        assert config.is_provider_enabled("claude") is False
        assert config.is_provider_enabled("unknown") is False
        
        # Test when batch processing is globally disabled
        config.enabled = False
        assert config.is_provider_enabled("openai") is False

    def test_get_model_config(self):
        """Test get_model_config helper method"""
        models = [
            BatchModelConfig(name="gpt-4o-mini", mode="chat", max_prompts_per_batch=20),
            BatchModelConfig(name="gpt-4", mode="chat", max_prompts_per_batch=50)
        ]
        providers = {
            "openai": BatchProviderConfig(enabled=True, models=models)
        }
        config = BatchConfig(providers=providers)
        
        # Test existing model
        model_config = config.get_model_config("openai", "gpt-4o-mini")
        assert model_config is not None
        assert model_config.name == "gpt-4o-mini"
        assert model_config.max_prompts_per_batch == 20
        
        # Test non-existent model
        model_config = config.get_model_config("openai", "non-existent")
        assert model_config is None
        
        # Test non-existent provider
        model_config = config.get_model_config("unknown", "gpt-4o-mini")
        assert model_config is None

    def test_invalid_poll_interval_bounds(self):
        """Test validation of poll interval bounds"""
        with pytest.raises(ValidationError):
            BatchConfig(default_poll_interval_seconds=3)
        
        with pytest.raises(ValidationError):
            BatchConfig(default_poll_interval_seconds=4000)

    def test_invalid_concurrent_polls_bounds(self):
        """Test validation of concurrent polls bounds"""
        with pytest.raises(ValidationError):
            BatchConfig(max_concurrent_provider_polls=0)
        
        with pytest.raises(ValidationError):
            BatchConfig(max_concurrent_provider_polls=25)


class TestLLMConfigBatchIntegration:
    """Test batch configuration integration with LLMConfig"""

    def test_load_config_with_batch_section(self):
        """Test loading configuration with complete batch section"""
        yaml_content = """
providers:
  openai:
    name: 'OpenAI'
    requires_api_key: true
    openai_compatible: true

models:
  gpt-4o-mini:
    provider: 'openai'
    max_tokens: 4096

batch:
  enabled: true
  default_poll_interval_seconds: 45
  max_concurrent_provider_polls: 8
  defaults:
    retry:
      max_attempts: 3
      base_delay: 1.5
      max_delay: 45
  providers:
    openai:
      enabled: true
      models:
        - name: gpt-4o-mini
          mode: chat
          max_prompts_per_batch: 25
          pricing_tier: batch
"""
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            config = LLMConfig()
            
            assert config.batch_config.enabled is True
            assert config.batch_config.default_poll_interval_seconds == 45
            assert config.batch_config.max_concurrent_provider_polls == 8
            assert config.batch_config.defaults.retry.max_attempts == 3
            assert config.batch_config.defaults.retry.base_delay == 1.5
            assert config.batch_config.defaults.retry.max_delay == 45
            
            openai_models = config.batch_config.get_provider_models("openai")
            assert len(openai_models) == 1
            assert openai_models[0].name == "gpt-4o-mini"
            assert openai_models[0].mode == "chat"
            assert openai_models[0].max_prompts_per_batch == 25
            assert openai_models[0].pricing_tier == "batch"

    def test_load_config_without_batch_section(self):
        """Test loading configuration without batch section (fallback to defaults)"""
        yaml_content = """
providers:
  openai:
    name: 'OpenAI'
    requires_api_key: true

models:
  gpt-4o-mini:
    provider: 'openai'
    max_tokens: 4096
"""
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            config = LLMConfig()
            
            # Should use default batch configuration
            assert config.batch_config.enabled is True
            assert config.batch_config.default_poll_interval_seconds == 30
            assert config.batch_config.max_concurrent_provider_polls == 5
            assert config.batch_config.providers == {}

    def test_load_config_with_invalid_provider_reference(self):
        """Test loading configuration with invalid provider reference in batch section"""
        yaml_content = """
providers:
  openai:
    name: 'OpenAI'
    requires_api_key: true

batch:
  enabled: true
  providers:
    nonexistent_provider:
      enabled: true
      models:
        - name: some-model
          mode: chat
"""
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("logging.warning") as mock_warning:
                config = LLMConfig()
                
                # Should warn about unknown provider but continue
                mock_warning.assert_called_with("Batch configuration references unknown provider: nonexistent_provider")
                assert "nonexistent_provider" not in config.batch_config.providers

    def test_load_config_with_invalid_model_config(self):
        """Test loading configuration with invalid model configuration"""
        yaml_content = """
providers:
  openai:
    name: 'OpenAI'
    requires_api_key: true

batch:
  enabled: true
  providers:
    openai:
      enabled: true
      models:
        - name: gpt-4o-mini
          mode: invalid_mode
          max_prompts_per_batch: 25
"""
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("logging.error") as mock_error:
                config = LLMConfig()
                
                # Should log error for invalid model config but continue
                mock_error.assert_called()
                # Provider should exist but have no models due to validation error
                assert "openai" in config.batch_config.providers
                assert len(config.batch_config.get_provider_models("openai")) == 0

    def test_load_config_with_malformed_batch_section(self):
        """Test loading configuration with malformed batch section"""
        yaml_content = """
providers:
  openai:
    name: 'OpenAI'
    requires_api_key: true

batch:
  enabled: true
  default_poll_interval_seconds: -10  # Invalid value
  providers:
    openai:
      enabled: true
"""
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("logging.error") as mock_error:
                config = LLMConfig()
                
                # Should log error and fall back to default configuration
                mock_error.assert_called()
                assert config.batch_config.enabled is True  # Default value
                assert config.batch_config.default_poll_interval_seconds == 30  # Default value

    def test_load_config_file_not_found(self):
        """Test fallback when config file is not found"""
        with patch("builtins.open", side_effect=FileNotFoundError):
            config = LLMConfig()
            
            # Should use default batch configuration
            assert config.batch_config.enabled is True
            assert config.batch_config.default_poll_interval_seconds == 30
            assert config.batch_config.providers == {}

    def test_batch_config_disabled_feature(self):
        """Test that batch feature can be cleanly disabled"""
        yaml_content = """
providers:
  openai:
    name: 'OpenAI'
    requires_api_key: true

batch:
  enabled: false
  providers:
    openai:
      enabled: true
      models:
        - name: gpt-4o-mini
          mode: chat
"""
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            config = LLMConfig()
            
            assert config.batch_config.enabled is False
            # Even with provider enabled, global disabled should take precedence
            assert config.batch_config.is_provider_enabled("openai") is False
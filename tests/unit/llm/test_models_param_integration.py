"""
Unit tests for app.llm.models module with parameter policy integration

Tests that LLMModelFactory correctly uses the centralized parameter normalization.
"""

from unittest.mock import MagicMock, Mock, call, patch

import pytest

from app.llm.config import ModelConfig, ModelProvider
from app.llm.models import LLMModelFactory


class TestLLMModelFactoryParameterNormalization:
    """Test LLMModelFactory integration with parameter policy"""

    def setup_method(self):
        """Setup test environment"""
        # Create a mock config
        self.mock_config = MagicMock()
        self.mock_config.enable_caching = False
        self.factory = LLMModelFactory(self.mock_config)

    @patch("app.llm.models.normalize_params")
    @patch("app.llm.models.ChatOpenAI")
    def test_openai_model_uses_normalized_params(
        self, mock_chat_openai, mock_normalize
    ):
        """Test OpenAI model creation uses normalized parameters"""
        # Setup
        config = ModelConfig(
            name="gpt-4",
            provider=ModelProvider.OPENAI,
            api_key="test-key",
            temperature=0.7,
            max_tokens=100,
        )

        # Mock normalize_params to return expected params
        normalized_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "request_timeout": 60,
        }
        mock_normalize.return_value = normalized_params

        # Mock successful model creation
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model

        # Execute
        result = self.factory._create_model_instance(config)

        # Verify normalize_params was called correctly
        expected_raw_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "request_timeout": 60,
        }
        mock_normalize.assert_called_once_with(
            ModelProvider.OPENAI, expected_raw_params, model_name=config.name
        )

        # Verify model was created with normalized params
        mock_chat_openai.assert_called_once()
        call_args = mock_chat_openai.call_args
        assert "model" in call_args.kwargs
        assert call_args.kwargs["model"] == "gpt-4"

        # Verify normalized params were passed
        for key, value in normalized_params.items():
            assert call_args.kwargs[key] == value

        assert result == mock_model

    @patch("app.llm.models.normalize_params")
    @patch("app.llm.models.ChatGoogleGenerativeAI")
    def test_gemini_model_uses_normalized_params(
        self, mock_chat_gemini, mock_normalize
    ):
        """Test Gemini model creation uses normalized parameters"""
        # Setup
        config = ModelConfig(
            name="gemini-pro",
            provider=ModelProvider.GEMINI,
            api_key="test-key",
            temperature=0.8,
            max_tokens=200,
        )

        # Mock normalize_params to return Gemini-specific normalized params
        normalized_params = {
            "temperature": 0.8,
            "max_output_tokens": 200,  # max_tokens renamed to max_output_tokens
            "top_p": 1.0,
            # frequency_penalty, presence_penalty, request_timeout should be dropped
        }
        mock_normalize.return_value = normalized_params

        # Mock successful model creation
        mock_model = MagicMock()
        mock_chat_gemini.return_value = mock_model

        # Execute
        result = self.factory._create_model_instance(config)

        # Verify normalize_params was called correctly
        expected_raw_params = {
            "temperature": 0.8,
            "max_tokens": 200,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "request_timeout": 60,
        }
        mock_normalize.assert_called_once_with(
            ModelProvider.GEMINI, expected_raw_params, model_name=config.name
        )

        # Verify model was created with normalized params
        mock_chat_gemini.assert_called_once()
        call_args = mock_chat_gemini.call_args
        assert call_args.kwargs["model"] == "gemini-pro"
        assert call_args.kwargs["api_key"] == "test-key"

        # Verify only normalized params were passed (no max_tokens, only max_output_tokens)
        for key, value in normalized_params.items():
            assert call_args.kwargs[key] == value

        # Verify forbidden params were not passed
        assert "frequency_penalty" not in call_args.kwargs
        assert "presence_penalty" not in call_args.kwargs
        assert "request_timeout" not in call_args.kwargs
        assert (
            "max_tokens" not in call_args.kwargs
        )  # Should be renamed to max_output_tokens

        assert result == mock_model

    @patch("app.llm.models.normalize_params")
    @patch("app.llm.models.ChatAnthropic")
    def test_claude_model_uses_normalized_params(
        self, mock_chat_anthropic, mock_normalize
    ):
        """Test Claude model creation uses normalized parameters"""
        # Setup
        config = ModelConfig(
            name="claude-3-haiku",
            provider=ModelProvider.CLAUDE,
            api_key="test-key",
            temperature=0.5,
            max_tokens=150,
        )

        # Mock normalize_params to return Claude-specific normalized params
        normalized_params = {
            "temperature": 0.5,
            "max_tokens": 150,
            "top_p": 1.0,
            "timeout": 60,  # request_timeout renamed to timeout
            # frequency_penalty, presence_penalty should be dropped
        }
        mock_normalize.return_value = normalized_params

        # Mock successful model creation
        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model

        # Execute
        result = self.factory._create_model_instance(config)

        # Verify normalize_params was called correctly
        expected_raw_params = {
            "temperature": 0.5,
            "max_tokens": 150,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "request_timeout": 60,
        }
        mock_normalize.assert_called_once_with(
            ModelProvider.CLAUDE, expected_raw_params, model_name=config.name
        )

        # Verify model was created with normalized params
        mock_chat_anthropic.assert_called_once()
        call_args = mock_chat_anthropic.call_args
        assert call_args.kwargs["model"] == "claude-3-haiku"
        assert call_args.kwargs["api_key"] == "test-key"

        # Verify normalized params were passed
        for key, value in normalized_params.items():
            assert call_args.kwargs[key] == value

        # Verify forbidden params were not passed
        assert "frequency_penalty" not in call_args.kwargs
        assert "presence_penalty" not in call_args.kwargs
        assert "request_timeout" not in call_args.kwargs  # Should be renamed to timeout

        assert result == mock_model

    @patch("app.llm.models.normalize_params")
    @patch("app.llm.models.logger")
    def test_parameter_normalization_failure(self, mock_logger, mock_normalize):
        """Test handling of parameter normalization failure"""
        # Setup
        config = ModelConfig(
            name="test-model", provider=ModelProvider.OPENAI, api_key="test-key"
        )

        # Mock normalize_params to raise ValueError
        mock_normalize.side_effect = ValueError("Unsupported provider: fake_provider")

        # Execute
        result = self.factory._create_model_instance(config)

        # Verify error handling
        assert result is None
        mock_logger.error.assert_called_once_with(
            "Parameter normalization failed: Unsupported provider: fake_provider"
        )

    @patch("app.llm.models.normalize_params")
    @patch("app.llm.models.ChatOpenAI")
    def test_kwargs_override_config_values(self, mock_chat_openai, mock_normalize):
        """Test that kwargs override config default values before normalization"""
        # Setup
        config = ModelConfig(
            name="gpt-4",
            provider=ModelProvider.OPENAI,
            api_key="test-key",
            temperature=0.7,  # Default value
            max_tokens=100,  # Default value
        )

        # Mock normalize_params
        mock_normalize.return_value = {
            "temperature": 0.9,  # Overridden value
            "max_tokens": 200,  # Overridden value
            "top_p": 0.8,  # Overridden value
            "frequency_penalty": 0.1,
            "presence_penalty": 0.0,
            "request_timeout": 30,
        }

        # Mock successful model creation
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model

        # Execute with kwargs overrides
        result = self.factory._create_model_instance(
            config,
            temperature=0.9,  # Override
            max_tokens=200,  # Override
            top_p=0.8,  # Override
            frequency_penalty=0.1,  # Override
            timeout=30,  # Override (maps to request_timeout)
        )

        # Verify that raw params include overrides before normalization
        expected_raw_params = {
            "temperature": 0.9,  # From kwargs
            "max_tokens": 200,  # From kwargs
            "top_p": 0.8,  # From kwargs
            "frequency_penalty": 0.1,  # From kwargs
            "presence_penalty": 0.0,  # From config (not overridden)
            "request_timeout": 30,  # From kwargs (timeout -> request_timeout)
        }
        mock_normalize.assert_called_once_with(
            ModelProvider.OPENAI, expected_raw_params, model_name=config.name
        )

        assert result == mock_model

    def test_unsupported_provider_error(self):
        """Invalid provider should raise ValidationError at ModelConfig construction now."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelConfig(
                name="unknown-model",
                provider="unknown_provider",  # Invalid provider
                api_key="test-key",
            )


class TestProviderSpecificParameterHandling:
    """Test that provider-specific parameter handling is correct"""

    def setup_method(self):
        """Setup test environment"""
        self.mock_config = MagicMock()
        self.mock_config.enable_caching = False
        self.factory = LLMModelFactory(self.mock_config)

    @patch("app.llm.models.ChatGoogleGenerativeAI")
    @patch("app.llm.models.normalize_params")
    def test_gemini_no_manual_parameter_mapping(self, mock_normalize, mock_chat_gemini):
        """Test that Gemini model creation doesn't do manual parameter mapping anymore"""
        # Setup
        config = ModelConfig(
            name="gemini-pro", provider=ModelProvider.GEMINI, api_key="test-key"
        )

        # The normalized params should already have max_output_tokens instead of max_tokens
        normalized_params = {"temperature": 0.7, "max_output_tokens": 100}
        mock_normalize.return_value = normalized_params

        mock_model = MagicMock()
        mock_chat_gemini.return_value = mock_model

        # Execute
        result = self.factory._create_gemini_model(config, normalized_params)

        # Verify ChatGoogleGenerativeAI was called with normalized params directly
        mock_chat_gemini.assert_called_once_with(
            model="gemini-pro", api_key="test-key", **normalized_params
        )

        assert result == mock_model

    @patch("app.llm.models.ChatAnthropic")
    @patch("app.llm.models.normalize_params")
    def test_claude_no_manual_parameter_filtering(
        self, mock_normalize, mock_chat_anthropic
    ):
        """Test that Claude model creation doesn't do manual parameter filtering anymore"""
        # Setup
        config = ModelConfig(
            name="claude-3-haiku", provider=ModelProvider.CLAUDE, api_key="test-key"
        )

        # The normalized params should already have forbidden params removed
        normalized_params = {"temperature": 0.7, "max_tokens": 100, "timeout": 30}
        mock_normalize.return_value = normalized_params

        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model

        # Execute
        result = self.factory._create_claude_model(config, normalized_params)

        # Verify ChatAnthropic was called with normalized params directly
        mock_chat_anthropic.assert_called_once_with(
            model="claude-3-haiku", api_key="test-key", **normalized_params
        )

        assert result == mock_model

        # Ensure no forbidden params are present
        call_kwargs = mock_chat_anthropic.call_args.kwargs
        assert "frequency_penalty" not in call_kwargs
        assert "presence_penalty" not in call_kwargs
        assert "request_timeout" not in call_kwargs

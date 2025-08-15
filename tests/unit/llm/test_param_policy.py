"""
Unit tests for app.llm.param_policy module

Tests parameter normalization and validation for all LLM providers.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.llm.config import ModelProvider
from app.llm.param_policy import (
    POLICIES,
    ProviderParamPolicy,
    get_provider_policy,
    get_supported_providers,
    normalize_params,
    validate_provider_params,
)


class TestProviderParamPolicy:
    """Test ProviderParamPolicy dataclass"""

    def test_provider_param_policy_creation(self):
        """Test ProviderParamPolicy creation"""
        policy = ProviderParamPolicy(
            allowed={"temperature", "max_tokens"},
            renamed={"old_param": "new_param"},
            dropped={"unsupported_param"},
        )

        assert policy.allowed == {"temperature", "max_tokens"}
        assert policy.renamed == {"old_param": "new_param"}
        assert policy.dropped == {"unsupported_param"}


class TestPoliciesDefinition:
    """Test that all provider policies are correctly defined"""

    def test_all_providers_have_policies(self):
        """Test that all ModelProvider enum values have policies defined"""
        expected_providers = {
            ModelProvider.OPENAI,
            ModelProvider.CUSTOM_OPENAI_COMPATIBLE,
            ModelProvider.GEMINI,
            ModelProvider.CLAUDE,
        }

        assert set(POLICIES.keys()) == expected_providers

    def test_openai_policy(self):
        """Test OpenAI provider policy"""
        policy = POLICIES[ModelProvider.OPENAI]

        expected_base = {
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "request_timeout",
        }
        # Dynamic patch may add 'logit_bias'; ensure base subset and if present it's acceptable
        assert expected_base.issubset(policy.allowed)
        assert policy.renamed == {}
        assert policy.dropped == set()

    def test_custom_openai_policy(self):
        """Test custom OpenAI-compatible provider policy"""
        policy = POLICIES[ModelProvider.CUSTOM_OPENAI_COMPATIBLE]

        expected_allowed = {
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "request_timeout",
        }

        assert policy.allowed == expected_allowed
        assert policy.renamed == {}
        assert policy.dropped == set()

    def test_gemini_policy(self):
        """Test Gemini provider policy"""
        policy = POLICIES[ModelProvider.GEMINI]

        expected_allowed = {"temperature", "max_output_tokens", "top_p"}
        expected_renamed = {"max_tokens": "max_output_tokens"}
        expected_dropped = {"frequency_penalty", "presence_penalty", "request_timeout"}

        assert policy.allowed == expected_allowed
        assert policy.renamed == expected_renamed
        assert policy.dropped == expected_dropped

    def test_claude_policy(self):
        """Test Claude provider policy"""

    policy = POLICIES[ModelProvider.CLAUDE]
    # Base allowed; dynamic patch may add 'metadata' and 'system'. Accept superset.
    expected_allowed = {"temperature", "max_tokens", "top_p", "timeout"}
    expected_renamed = {"request_timeout": "timeout"}
    expected_dropped = {"frequency_penalty", "presence_penalty"}

    # Ensure base required params present
    assert expected_allowed.issubset(policy.allowed)
    assert policy.renamed == expected_renamed
    assert policy.dropped == expected_dropped


class TestNormalizeParams:
    """Test normalize_params function"""

    @patch("app.llm.param_policy.logger")
    def test_openai_params_passthrough(self, mock_logger):
        """Test OpenAI parameters pass through unchanged"""
        raw_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "request_timeout": 30,
        }

        result = normalize_params(ModelProvider.OPENAI, raw_params)

        assert result == raw_params
        # No debug/warning logs should be called for passthrough
        mock_logger.debug.assert_not_called()
        mock_logger.warning.assert_not_called()

    @patch("app.llm.param_policy.logger")
    def test_gemini_params_normalized(self, mock_logger):
        """Test Gemini parameters are correctly normalized"""
        raw_params = {
            "temperature": 0.7,
            "max_tokens": 100,  # Should be renamed
            "top_p": 0.9,
            "frequency_penalty": 0.1,  # Should be dropped
            "presence_penalty": 0.2,  # Should be dropped
            "request_timeout": 30,  # Should be dropped
        }

        result = normalize_params(ModelProvider.GEMINI, raw_params)

        expected = {
            "temperature": 0.7,
            "max_output_tokens": 100,  # Renamed from max_tokens
            "top_p": 0.9,
        }

        assert result == expected

        # Should log parameter rename
        # Enum now appears in log message (ModelProvider.GEMINI)
        mock_logger.debug.assert_any_call(
            "Parameter renamed for ModelProvider.GEMINI: max_tokens -> max_output_tokens"
        )

        # Should log dropped parameters
        mock_logger.debug.assert_any_call(
            "Parameter dropped for ModelProvider.GEMINI: frequency_penalty (value: 0.1)"
        )
        mock_logger.debug.assert_any_call(
            "Parameter dropped for ModelProvider.GEMINI: presence_penalty (value: 0.2)"
        )
        mock_logger.debug.assert_any_call(
            "Parameter dropped for ModelProvider.GEMINI: request_timeout (value: 30)"
        )

    @patch("app.llm.param_policy.logger")
    def test_claude_params_normalized(self, mock_logger):
        """Test Claude parameters are correctly normalized"""
        raw_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "frequency_penalty": 0.1,  # Should be dropped
            "presence_penalty": 0.2,  # Should be dropped
            "request_timeout": 30,  # Should be renamed to timeout
        }

        result = normalize_params(ModelProvider.CLAUDE, raw_params)

        expected = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "timeout": 30,  # Renamed from request_timeout
        }

        assert result == expected

        # Should log parameter rename
        mock_logger.debug.assert_any_call(
            "Parameter renamed for ModelProvider.CLAUDE: request_timeout -> timeout"
        )

        # Should log dropped parameters
        mock_logger.debug.assert_any_call(
            "Parameter dropped for ModelProvider.CLAUDE: frequency_penalty (value: 0.1)"
        )
        mock_logger.debug.assert_any_call(
            "Parameter dropped for ModelProvider.CLAUDE: presence_penalty (value: 0.2)"
        )

    @patch("app.llm.param_policy.logger")
    def test_none_values_skipped(self, mock_logger):
        """Test that None parameter values are skipped"""
        raw_params = {"temperature": 0.7, "max_tokens": None, "frequency_penalty": None}

        result = normalize_params(ModelProvider.OPENAI, raw_params)

        assert result == {"temperature": 0.7}
        mock_logger.debug.assert_not_called()
        mock_logger.warning.assert_not_called()

    @patch("app.llm.param_policy.logger")
    def test_unknown_parameter_warning(self, mock_logger):
        """Test warning logged for unknown parameters"""
        raw_params = {"temperature": 0.7, "unknown_param": "value"}

        result = normalize_params(ModelProvider.OPENAI, raw_params)

        assert result == {"temperature": 0.7}
        mock_logger.warning.assert_called_once_with(
            "Unknown parameter for ModelProvider.OPENAI: unknown_param (value: value) - dropping"
        )

    def test_unsupported_provider(self):
        """Test exception raised for unsupported provider"""
        # Create a fake provider that's not in POLICIES
        fake_provider = "fake_provider"

        with pytest.raises(ValueError, match="Unsupported provider: fake_provider"):
            normalize_params(fake_provider, {})

    def test_empty_params(self):
        """Test normalization with empty parameters"""
        result = normalize_params(ModelProvider.OPENAI, {})
        assert result == {}


class TestGetProviderPolicy:
    """Test get_provider_policy function"""

    def test_get_openai_policy(self):
        """Test getting OpenAI provider policy"""
        policy = get_provider_policy(ModelProvider.OPENAI)

        assert isinstance(policy, ProviderParamPolicy)
        assert "temperature" in policy.allowed
        assert policy.renamed == {}
        assert policy.dropped == set()

    def test_get_gemini_policy(self):
        """Test getting Gemini provider policy"""
        policy = get_provider_policy(ModelProvider.GEMINI)

        assert isinstance(policy, ProviderParamPolicy)
        assert "max_output_tokens" in policy.allowed
        assert policy.renamed == {"max_tokens": "max_output_tokens"}
        assert "frequency_penalty" in policy.dropped

    def test_unsupported_provider(self):
        """Test exception for unsupported provider"""
        fake_provider = "fake_provider"

        with pytest.raises(ValueError, match="Unsupported provider: fake_provider"):
            get_provider_policy(fake_provider)


class TestGetSupportedProviders:
    """Test get_supported_providers function"""

    def test_supported_providers(self):
        """Test getting set of supported providers"""
        providers = get_supported_providers()

        expected = {
            ModelProvider.OPENAI,
            ModelProvider.CUSTOM_OPENAI_COMPATIBLE,
            ModelProvider.GEMINI,
            ModelProvider.CLAUDE,
        }

        assert providers == expected


class TestValidateProviderParams:
    """Test validate_provider_params function"""

    def test_valid_openai_params(self):
        """Test validation of valid OpenAI parameters"""
        params = {"temperature": 0.7, "max_tokens": 100, "frequency_penalty": 0.1}

        assert validate_provider_params(ModelProvider.OPENAI, params) is True

    def test_invalid_openai_params(self):
        """Test validation of invalid OpenAI parameters"""
        params = {"temperature": 0.7, "unknown_param": "value"}

        assert validate_provider_params(ModelProvider.OPENAI, params) is False

    def test_valid_gemini_params_with_mapping(self):
        """Test validation of Gemini parameters that need mapping"""
        params = {
            "temperature": 0.7,
            "max_tokens": 100,  # Will be renamed to max_output_tokens
            "frequency_penalty": 0.1,  # Will be dropped
        }

        assert validate_provider_params(ModelProvider.GEMINI, params) is True

    def test_valid_claude_params_with_mapping(self):
        """Test validation of Claude parameters that need mapping"""
        params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "request_timeout": 30,  # Will be renamed to timeout
            "frequency_penalty": 0.1,  # Will be dropped
        }

        assert validate_provider_params(ModelProvider.CLAUDE, params) is True

    def test_none_values_in_validation(self):
        """Test validation ignores None values"""
        params = {"temperature": 0.7, "max_tokens": None, "unknown_param": None}

        assert validate_provider_params(ModelProvider.OPENAI, params) is True

    def test_unsupported_provider_validation(self):
        """Test validation returns False for unsupported provider"""
        fake_provider = "fake_provider"
        params = {"temperature": 0.7}

        assert validate_provider_params(fake_provider, params) is False


class TestParameterPolicyIntegration:
    """Test integration scenarios for parameter policy"""

    @patch("app.llm.param_policy.logger")
    def test_typical_gemini_workflow(self, mock_logger):
        """Test typical parameter normalization workflow for Gemini"""
        # Simulate parameters coming from service layer
        service_params = {
            "temperature": 0.8,
            "max_tokens": 200,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.1,
            "request_timeout": 60,
        }

        # Normalize for Gemini
        normalized = normalize_params(ModelProvider.GEMINI, service_params)

        # Should only have Gemini-compatible parameters
        expected = {"temperature": 0.8, "max_output_tokens": 200, "top_p": 0.95}

        assert normalized == expected

        # Verify logging occurred
        assert mock_logger.debug.call_count >= 4  # 1 rename + 3 drops

    @patch("app.llm.param_policy.logger")
    def test_typical_claude_workflow(self, mock_logger):
        """Test typical parameter normalization workflow for Claude"""
        # Simulate parameters coming from service layer
        service_params = {
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 1.0,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.1,
            "request_timeout": 45,
        }

        # Normalize for Claude
        normalized = normalize_params(ModelProvider.CLAUDE, service_params)

        # Should only have Claude-compatible parameters
        expected = {"temperature": 0.7, "max_tokens": 150, "top_p": 1.0, "timeout": 45}

        assert normalized == expected

        # Verify logging occurred
        assert mock_logger.debug.call_count >= 3  # 1 rename + 2 drops

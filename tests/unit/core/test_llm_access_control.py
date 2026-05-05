"""
Tests for shared model API access control settings.

This module tests the configurable access control setting that still gates
batch, vector, embedding, and agent-related model endpoints.
"""

import pytest

from inference_core.core.config import (
    build_pure_defaults_with_overrides,
    get_settings_pure_defaults,
)


class TestLLMAccessControlConfig:
    """Test LLM access control configuration settings"""

    def test_llm_api_access_mode_default_value(self):
        """Test that the default value for llm_api_access_mode is 'superuser'"""
        settings = get_settings_pure_defaults()
        assert settings.llm_api_access_mode == "superuser"

    # assert settings.llm_api_access_mode == "superuser"

    def test_llm_api_access_mode_from_env_public(self):
        """Test setting llm_api_access_mode to 'public' via environment"""
        settings = build_pure_defaults_with_overrides(llm_api_access_mode="public")
        assert settings.llm_api_access_mode == "public"

    def test_llm_api_access_mode_from_env_user(self):
        """Test setting llm_api_access_mode to 'user' via environment"""
        settings = build_pure_defaults_with_overrides(llm_api_access_mode="user")
        assert settings.llm_api_access_mode == "user"

    def test_llm_api_access_mode_from_env_superuser(self):
        """Test setting llm_api_access_mode to 'superuser' via environment"""
        settings = build_pure_defaults_with_overrides(llm_api_access_mode="superuser")
        assert settings.llm_api_access_mode == "superuser"

    def test_llm_api_access_mode_invalid_value_validation(self):
        """Test that invalid values for llm_api_access_mode raise validation error"""
        with pytest.raises(ValueError):
            build_pure_defaults_with_overrides(llm_api_access_mode="invalid_mode")

    def test_llm_api_access_mode_case_sensitivity(self):
        """Test that llm_api_access_mode is case sensitive"""
        # with patch.dict("os.environ", {"LLM_API_ACCESS_MODE": "PUBLIC"}):
        with pytest.raises(ValueError):
            build_pure_defaults_with_overrides(llm_api_access_mode="PUBLIC")

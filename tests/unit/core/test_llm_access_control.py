"""
Tests for LLM API access control functionality.

This module tests the configurable access control for LLM endpoints
based on the LLM_API_ACCESS_MODE environment variable.
"""

from unittest.mock import MagicMock, patch

import pytest

from inference_core.core.config import (
    build_pure_defaults_with_overrides,
    get_settings_pure_defaults,
)
from inference_core.core.dependecies import (
    get_current_active_user,
    get_current_superuser,
    get_llm_router_dependencies,
)


class TestLLMAccessControl:
    """Test suite for LLM access control configuration"""

    def test_get_llm_router_dependencies_public_mode(self):
        """Test that public mode returns no dependencies"""
        with patch("inference_core.core.dependecies.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.llm_api_access_mode = "public"
            mock_get_settings.return_value = mock_settings

            dependencies = get_llm_router_dependencies()

            assert dependencies == []

    def test_get_llm_router_dependencies_user_mode(self):
        """Test that user mode returns active user dependency"""
        with patch("inference_core.core.dependecies.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.llm_api_access_mode = "user"
            mock_get_settings.return_value = mock_settings

            dependencies = get_llm_router_dependencies()

            assert len(dependencies) == 1
            # Verify it's a Depends object with the correct dependency
            assert hasattr(dependencies[0], "dependency")
            assert dependencies[0].dependency == get_current_active_user

    def test_get_llm_router_dependencies_superuser_mode(self):
        """Test that superuser mode returns superuser dependency"""
        with patch("inference_core.core.dependecies.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.llm_api_access_mode = "superuser"
            mock_get_settings.return_value = mock_settings

            dependencies = get_llm_router_dependencies()

            assert len(dependencies) == 1
            # Verify it's a Depends object with the correct dependency
            assert hasattr(dependencies[0], "dependency")
            assert dependencies[0].dependency == get_current_superuser

    def test_get_llm_router_dependencies_default_mode(self):
        """Test that default mode (when not specified) returns superuser dependency"""
        with patch("inference_core.core.dependecies.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            # Test default behavior when llm_api_access_mode is not set
            mock_settings.llm_api_access_mode = (
                "superuser"  # This should be default in config
            )
            mock_get_settings.return_value = mock_settings

            dependencies = get_llm_router_dependencies()

            assert len(dependencies) == 1
            assert hasattr(dependencies[0], "dependency")
            assert dependencies[0].dependency == get_current_superuser

    def test_get_llm_router_dependencies_invalid_mode_fallback(self):
        """Test that invalid mode falls back to superuser for security"""
        with patch("inference_core.core.dependecies.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.llm_api_access_mode = "invalid_mode"
            mock_get_settings.return_value = mock_settings

            dependencies = get_llm_router_dependencies()

            # Should fallback to superuser for security
            assert len(dependencies) == 1
            assert hasattr(dependencies[0], "dependency")
            assert dependencies[0].dependency == get_current_superuser


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

"""
Tests for email configuration loading and validation
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from inference_core.core.email_config import (
    EmailConfig,
    EmailHostConfig,
    EmailSettings,
    FullEmailConfig,
    clear_email_config_cache,
    get_email_config,
    is_email_configured,
)


class TestEmailHostConfig:
    """Test EmailHostConfig validation"""

    def test_valid_ssl_config(self):
        """Test valid SSL configuration"""
        config = EmailHostConfig(
            host="smtp.gmail.com",
            port=465,
            use_ssl=True,
            use_starttls=False,
            username="test@example.com",
            password_env="TEST_PASSWORD",
            from_email="no-reply@example.com",
        )
        assert config.use_ssl is True
        assert config.use_starttls is False

    def test_valid_starttls_config(self):
        """Test valid STARTTLS configuration"""
        config = EmailHostConfig(
            host="smtp.mailgun.org",
            port=587,
            use_ssl=False,
            use_starttls=True,
            username="test@mailgun.org",
            password_env="TEST_PASSWORD",
            from_email="no-reply@example.com",
        )
        assert config.use_ssl is False
        assert config.use_starttls is True

    def test_ssl_and_starttls_mutually_exclusive(self):
        """Test that SSL and STARTTLS cannot both be enabled"""
        with pytest.raises(ValidationError, match="mutually exclusive"):
            EmailHostConfig(
                host="smtp.example.com",
                port=465,
                use_ssl=True,
                use_starttls=True,
                username="test@example.com",
                password_env="TEST_PASSWORD",
                from_email="no-reply@example.com",
            )

    def test_env_var_resolution_in_username(self):
        """Test environment variable resolution in username"""
        with patch.dict(os.environ, {"TEST_USERNAME": "resolved@example.com"}):
            config = EmailHostConfig(
                host="smtp.example.com",
                port=465,
                use_ssl=True,
                username="${TEST_USERNAME}",
                password_env="TEST_PASSWORD",
                from_email="no-reply@example.com",
            )
            assert config.username == "resolved@example.com"

    def test_password_retrieval(self):
        """Test password retrieval from environment"""
        with patch.dict(os.environ, {"TEST_PASSWORD": "secret123"}):
            config = EmailHostConfig(
                host="smtp.example.com",
                port=465,
                use_ssl=True,
                username="test@example.com",
                password_env="TEST_PASSWORD",
                from_email="no-reply@example.com",
            )
            assert config.get_password() == "secret123"

    def test_password_not_found(self):
        """Test behavior when password environment variable not found"""
        config = EmailHostConfig(
            host="smtp.example.com",
            port=465,
            use_ssl=True,
            username="test@example.com",
            password_env="NONEXISTENT_PASSWORD",
            from_email="no-reply@example.com",
        )
        assert config.get_password() is None


class TestEmailConfig:
    """Test EmailConfig validation and functionality"""

    def test_valid_config(self):
        """Test valid email configuration"""
        host_config = EmailHostConfig(
            host="smtp.gmail.com",
            port=465,
            use_ssl=True,
            username="test@example.com",
            password_env="TEST_PASSWORD",
            from_email="no-reply@example.com",
        )
        
        config = EmailConfig(
            default_host="primary",
            hosts={"primary": host_config}
        )
        
        assert config.default_host == "primary"
        assert "primary" in config.hosts

    def test_default_host_must_exist(self):
        """Test that default_host must exist in hosts"""
        host_config = EmailHostConfig(
            host="smtp.gmail.com",
            port=465,
            use_ssl=True,
            username="test@example.com",
            password_env="TEST_PASSWORD",
            from_email="no-reply@example.com",
        )
        
        with pytest.raises(ValidationError, match="not found in hosts"):
            EmailConfig(
                default_host="nonexistent",
                hosts={"primary": host_config}
            )

    def test_get_host_config(self):
        """Test getting host configuration by alias"""
        host_config = EmailHostConfig(
            host="smtp.gmail.com",
            port=465,
            use_ssl=True,
            username="test@example.com",
            password_env="TEST_PASSWORD",
            from_email="no-reply@example.com",
        )
        
        config = EmailConfig(
            default_host="primary",
            hosts={"primary": host_config, "backup": host_config}
        )
        
        # Test getting by alias
        assert config.get_host_config("primary") == host_config
        assert config.get_host_config("backup") == host_config
        
        # Test getting default
        assert config.get_host_config() == host_config

    def test_get_nonexistent_host(self):
        """Test error when getting nonexistent host"""
        host_config = EmailHostConfig(
            host="smtp.gmail.com",
            port=465,
            use_ssl=True,
            username="test@example.com",
            password_env="TEST_PASSWORD",
            from_email="no-reply@example.com",
        )
        
        config = EmailConfig(
            default_host="primary",
            hosts={"primary": host_config}
        )
        
        with pytest.raises(ValueError, match="not found in configuration"):
            config.get_host_config("nonexistent")

    def test_list_host_aliases(self):
        """Test listing available host aliases"""
        host_config = EmailHostConfig(
            host="smtp.gmail.com",
            port=465,
            use_ssl=True,
            username="test@example.com",
            password_env="TEST_PASSWORD",
            from_email="no-reply@example.com",
        )
        
        config = EmailConfig(
            default_host="primary",
            hosts={"primary": host_config, "backup": host_config}
        )
        
        aliases = config.list_host_aliases()
        assert "primary" in aliases
        assert "backup" in aliases
        assert len(aliases) == 2


class TestEmailSettings:
    """Test EmailSettings validation"""

    def test_default_settings(self):
        """Test default settings values"""
        settings = EmailSettings()
        assert settings.app_public_url == "http://localhost:8000"
        assert settings.default_timeout == 10
        assert settings.default_verify_hostname is True

    def test_app_url_env_resolution(self):
        """Test environment variable resolution in app URL"""
        with patch.dict(os.environ, {"APP_URL": "https://production.com"}):
            settings = EmailSettings(app_public_url="${APP_URL}")
            assert settings.app_public_url == "https://production.com"


class TestConfigLoading:
    """Test configuration loading from YAML files"""

    def test_load_valid_config(self):
        """Test loading valid configuration from YAML"""
        yaml_content = {
            "email": {
                "default_host": "primary",
                "hosts": {
                    "primary": {
                        "host": "smtp.gmail.com",
                        "port": 465,
                        "use_ssl": True,
                        "use_starttls": False,
                        "username": "test@example.com",
                        "password_env": "TEST_PASSWORD",
                        "from_email": "no-reply@example.com",
                    }
                }
            },
            "settings": {
                "app_public_url": "http://localhost:8000"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name
        
        try:
            with patch("inference_core.core.email_config._get_config_path", return_value=Path(temp_path)):
                clear_email_config_cache()
                config = get_email_config()
                
                assert config.email.default_host == "primary"
                assert "primary" in config.email.hosts
                assert config.settings.app_public_url == "http://localhost:8000"
        finally:
            os.unlink(temp_path)

    def test_load_config_file_not_found(self):
        """Test behavior when config file not found"""
        with patch("inference_core.core.email_config._get_config_path", return_value=Path("/nonexistent/file.yaml")):
            clear_email_config_cache()
            with pytest.raises(FileNotFoundError):
                get_email_config()

    def test_is_email_configured_true(self):
        """Test is_email_configured returns True when config is valid"""
        yaml_content = {
            "email": {
                "default_host": "primary",
                "hosts": {
                    "primary": {
                        "host": "smtp.gmail.com",
                        "port": 465,
                        "use_ssl": True,
                        "username": "test@example.com",
                        "password_env": "TEST_PASSWORD",
                        "from_email": "no-reply@example.com",
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name
        
        try:
            with patch("inference_core.core.email_config._get_config_path", return_value=Path(temp_path)):
                clear_email_config_cache()
                assert is_email_configured() is True
        finally:
            os.unlink(temp_path)

    def test_is_email_configured_false(self):
        """Test is_email_configured returns False when config is invalid"""
        with patch("inference_core.core.email_config._get_config_path", return_value=Path("/nonexistent/file.yaml")):
            clear_email_config_cache()
            assert is_email_configured() is False

    def test_invalid_yaml_syntax(self):
        """Test behavior with invalid YAML syntax"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with patch("inference_core.core.email_config._get_config_path", return_value=Path(temp_path)):
                clear_email_config_cache()
                with pytest.raises(ValueError, match="Invalid YAML syntax"):
                    get_email_config()
        finally:
            os.unlink(temp_path)
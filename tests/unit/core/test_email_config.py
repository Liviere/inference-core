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
    ImapHostConfig,
    SmtpHostConfig,
    clear_email_config_cache,
    get_email_config,
    is_email_configured,
)


def make_smtp_config(**overrides) -> SmtpHostConfig:
    """Helper to create SmtpHostConfig with sensible defaults."""
    defaults = {
        "host": "smtp.example.com",
        "port": 465,
        "use_ssl": True,
        "use_starttls": False,
        "username": "test@example.com",
        "password_env": "TEST_PASSWORD",
        "from_email": "no-reply@example.com",
    }
    defaults.update(overrides)
    return SmtpHostConfig(**defaults)


def make_host_config(smtp_overrides: dict = None, **host_overrides) -> EmailHostConfig:
    """Helper to create EmailHostConfig with sensible defaults."""
    smtp = make_smtp_config(**(smtp_overrides or {}))
    return EmailHostConfig(smtp=smtp, **host_overrides)


class TestSmtpHostConfig:
    """Test SmtpHostConfig validation"""

    def test_valid_ssl_config(self):
        """Test valid SSL configuration"""
        config = SmtpHostConfig(
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
        config = SmtpHostConfig(
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
            SmtpHostConfig(
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
            config = SmtpHostConfig(
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
            config = SmtpHostConfig(
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
        config = SmtpHostConfig(
            host="smtp.example.com",
            port=465,
            use_ssl=True,
            username="test@example.com",
            password_env="NONEXISTENT_PASSWORD",
            from_email="no-reply@example.com",
        )
        assert config.get_password() is None



def make_imap_config(**overrides) -> ImapHostConfig:
    return ImapHostConfig(host="imap.example.com", username="test@example.com", **overrides)


class TestConnectAddress:
    """connect_address: an address the caller resolved and checked itself."""

    def test_unset_by_default(self):
        assert make_smtp_config().connect_address is None
        assert make_imap_config().connect_address is None

    @pytest.mark.parametrize(
        ("address", "stored"),
        [("203.0.113.7", "203.0.113.7"), ("2001:0db8::0007", "2001:db8::7")],
    )
    def test_accepts_an_ip_literal(self, address, stored):
        assert make_smtp_config(connect_address=address).connect_address == stored
        assert make_imap_config(connect_address=address).connect_address == stored

    @pytest.mark.parametrize("address", ["smtp.example.com", "203.0.113.300", ""])
    def test_refuses_anything_else(self, address):
        """A host name would be resolved again, which is what the field avoids."""
        with pytest.raises(ValidationError, match="must be an IP address"):
            make_smtp_config(connect_address=address)
        with pytest.raises(ValidationError, match="must be an IP address"):
            make_imap_config(connect_address=address)



class TestInMemoryPassword:
    """password: for hosts built at runtime, without touching the environment."""

    def test_takes_precedence_over_password_env(self, monkeypatch):
        monkeypatch.setenv("TEST_PASSWORD", "from-env")
        assert make_smtp_config(password="in-memory").get_password() == "in-memory"
        assert make_imap_config(
            password="in-memory", password_env="TEST_PASSWORD"
        ).get_password() == "in-memory"

    def test_password_env_still_works_without_it(self, monkeypatch):
        monkeypatch.setenv("TEST_PASSWORD", "from-env")
        assert make_smtp_config().get_password() == "from-env"

    def test_is_left_out_of_dumps_and_repr(self):
        smtp = make_smtp_config(password="s3cret")
        imap = make_imap_config(password="s3cret")

        for config in (smtp, imap):
            assert "password" not in config.model_dump()
            assert "s3cret" not in config.model_dump_json()
            assert "s3cret" not in repr(config)
        host = EmailHostConfig(smtp=smtp, imap=imap)
        assert "s3cret" not in host.model_dump_json()
        assert host.get_password() == "s3cret"


class TestEmailHostConfig:
    """Test EmailHostConfig validation and property accessors"""

    def test_property_accessors(self):
        """Test that EmailHostConfig exposes SMTP properties for compatibility"""
        host_config = make_host_config(
            smtp_overrides={
                "host": "smtp.gmail.com",
                "port": 465,
                "use_ssl": True,
                "from_email": "no-reply@example.com",
            }
        )
        # Verify property accessors work
        assert host_config.host == "smtp.gmail.com"
        assert host_config.port == 465
        assert host_config.use_ssl is True
        assert host_config.from_email == "no-reply@example.com"

    def test_get_password_delegates_to_smtp(self):
        """Test that get_password() delegates to SMTP config"""
        with patch.dict(os.environ, {"TEST_PASSWORD": "secret123"}):
            host_config = make_host_config()
            assert host_config.get_password() == "secret123"

    def test_has_imap_false_by_default(self):
        """Test has_imap() returns False when IMAP not configured"""
        host_config = make_host_config()
        assert host_config.has_imap() is False

    def test_has_imap_true_when_configured(self):
        """Test has_imap() returns True when IMAP is configured"""
        from inference_core.core.email_config import ImapHostConfig

        imap_config = ImapHostConfig(
            host="imap.example.com",
            username="test@example.com",
            password_env="IMAP_PASSWORD",
        )
        host_config = make_host_config(imap=imap_config)
        assert host_config.has_imap() is True


class TestEmailConfig:
    """Test EmailConfig validation and functionality"""

    def test_valid_config(self):
        """Test valid email configuration"""
        host_config = make_host_config(
            smtp_overrides={
                "host": "smtp.gmail.com",
                "port": 465,
                "use_ssl": True,
            }
        )

        config = EmailConfig(default_host="primary", hosts={"primary": host_config})

        assert config.default_host == "primary"
        assert "primary" in config.hosts

    def test_default_host_must_exist(self):
        """Test that default_host must exist in hosts"""
        host_config = make_host_config()

        with pytest.raises(ValidationError, match="not found in hosts"):
            EmailConfig(default_host="nonexistent", hosts={"primary": host_config})

    def test_get_host_config(self):
        """Test getting host configuration by alias"""
        host_config = make_host_config()

        config = EmailConfig(
            default_host="primary",
            hosts={"primary": host_config, "backup": host_config},
        )

        # Test getting by alias
        assert config.get_host_config("primary") == host_config
        assert config.get_host_config("backup") == host_config

        # Test getting default
        assert config.get_host_config() == host_config

    def test_get_nonexistent_host(self):
        """Test error when getting nonexistent host"""
        host_config = make_host_config()

        config = EmailConfig(default_host="primary", hosts={"primary": host_config})

        with pytest.raises(ValueError, match="not found in configuration"):
            config.get_host_config("nonexistent")

    def test_list_host_aliases(self):
        """Test listing available host aliases"""
        host_config = make_host_config()

        config = EmailConfig(
            default_host="primary",
            hosts={"primary": host_config, "backup": host_config},
        )

        aliases = config.list_host_aliases()
        assert "primary" in aliases
        assert "backup" in aliases
        assert len(aliases) == 2

    def test_processed_ttl_seconds_default(self):
        """Test default processed_ttl_seconds value (7 days)"""
        host_config = make_host_config()

        config = EmailConfig(default_host="primary", hosts={"primary": host_config})

        assert config.processed_ttl_seconds == 7 * 24 * 3600  # 7 days in seconds

    def test_processed_ttl_seconds_custom(self):
        """Test custom processed_ttl_seconds value"""
        host_config = make_host_config()

        config = EmailConfig(
            default_host="primary",
            hosts={"primary": host_config},
            processed_ttl_seconds=3 * 24 * 3600,  # 3 days
        )

        assert config.processed_ttl_seconds == 3 * 24 * 3600

    def test_processed_ttl_seconds_validation_min(self):
        """Test processed_ttl_seconds minimum validation (1 hour)"""
        host_config = make_host_config()

        with pytest.raises(ValidationError):
            EmailConfig(
                default_host="primary",
                hosts={"primary": host_config},
                processed_ttl_seconds=1800,  # 30 minutes - below minimum
            )

    def test_processed_ttl_seconds_validation_max(self):
        """Test processed_ttl_seconds maximum validation (30 days)"""
        host_config = make_host_config()

        with pytest.raises(ValidationError):
            EmailConfig(
                default_host="primary",
                hosts={"primary": host_config},
                processed_ttl_seconds=31 * 24 * 3600,  # 31 days - above maximum
            )


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
        """Test loading valid configuration from YAML with new smtp structure"""
        yaml_content = {
            "email": {
                "default_host": "primary",
                "hosts": {
                    "primary": {
                        "smtp": {
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
            },
            "settings": {"app_public_url": "http://localhost:8000"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            with patch(
                "inference_core.core.email_config._get_config_path",
                return_value=Path(temp_path),
            ):
                clear_email_config_cache()
                config = get_email_config()

                assert config.email.default_host == "primary"
                assert "primary" in config.email.hosts
                # Verify SMTP settings are accessible via property accessors
                assert config.email.hosts["primary"].host == "smtp.gmail.com"
                assert config.email.hosts["primary"].smtp.host == "smtp.gmail.com"
                assert config.settings.app_public_url == "http://localhost:8000"
        finally:
            os.unlink(temp_path)

    def test_load_config_file_not_found(self):
        """Test behavior when config file not found"""
        with patch(
            "inference_core.core.email_config._get_config_path",
            return_value=Path("/nonexistent/file.yaml"),
        ):
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
                        "smtp": {
                            "host": "smtp.gmail.com",
                            "port": 465,
                            "use_ssl": True,
                            "username": "test@example.com",
                            "password_env": "TEST_PASSWORD",
                            "from_email": "no-reply@example.com",
                        }
                    }
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            with patch(
                "inference_core.core.email_config._get_config_path",
                return_value=Path(temp_path),
            ):
                clear_email_config_cache()
                assert is_email_configured() is True
        finally:
            os.unlink(temp_path)

    def test_is_email_configured_false(self):
        """Test is_email_configured returns False when config is invalid"""
        with patch(
            "inference_core.core.email_config._get_config_path",
            return_value=Path("/nonexistent/file.yaml"),
        ):
            clear_email_config_cache()
            assert is_email_configured() is False

    def test_invalid_yaml_syntax(self):
        """Test behavior with invalid YAML syntax"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            with patch(
                "inference_core.core.email_config._get_config_path",
                return_value=Path(temp_path),
            ):
                clear_email_config_cache()
                with pytest.raises(ValueError, match="Invalid YAML syntax"):
                    get_email_config()
        finally:
            os.unlink(temp_path)

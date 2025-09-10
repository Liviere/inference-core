"""
Email Configuration

Loads and validates email configuration from YAML files with environment variable support.
"""

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class EmailHostConfig(BaseModel):
    """Configuration for a single SMTP host"""
    
    host: str = Field(..., description="SMTP server hostname")
    port: int = Field(..., ge=1, le=65535, description="SMTP server port")
    use_ssl: bool = Field(default=False, description="Use SSL connection")
    use_starttls: bool = Field(default=False, description="Use STARTTLS after connection")
    username: str = Field(..., description="SMTP username (supports ${ENV_VAR} syntax)")
    password_env: str = Field(..., description="Environment variable name containing password")
    from_email: str = Field(..., description="From email address")
    from_name: Optional[str] = Field(default=None, description="From display name")
    timeout: int = Field(default=10, ge=1, le=300, description="Connection timeout in seconds")
    verify_hostname: bool = Field(default=True, description="Verify SSL hostname")
    rate_limit_per_minute: Optional[int] = Field(default=None, ge=1, description="Rate limit per minute")
    max_attachment_mb: int = Field(default=10, ge=1, le=100, description="Max attachment size in MB")
    
    @field_validator('username')
    @classmethod
    def resolve_username(cls, v: str) -> str:
        """Resolve environment variables in username"""
        return _resolve_env_vars(v)
    
    @model_validator(mode='after')
    def validate_ssl_options(self):
        """Ensure SSL and STARTTLS are mutually exclusive"""
        if self.use_ssl and self.use_starttls:
            raise ValueError("use_ssl and use_starttls are mutually exclusive")
        if not self.use_ssl and not self.use_starttls:
            logger.warning(f"Host {self.host} configured without SSL or STARTTLS - emails will be sent insecurely")
        return self
    
    def get_password(self) -> Optional[str]:
        """Get password from environment variable"""
        return os.getenv(self.password_env)


class EmailSettings(BaseModel):
    """Global email settings"""
    
    app_public_url: str = Field(default="http://localhost:8000", description="Public URL for email links")
    default_timeout: int = Field(default=10, ge=1, le=300, description="Default timeout for hosts")
    default_verify_hostname: bool = Field(default=True, description="Default hostname verification")
    default_max_attachment_mb: int = Field(default=10, ge=1, le=100, description="Default max attachment size")
    template_directory: str = Field(default="templates/email", description="Email template directory")
    development: Dict[str, Any] = Field(default_factory=dict, description="Development settings")
    
    @field_validator('app_public_url')
    @classmethod
    def resolve_app_url(cls, v: str) -> str:
        """Resolve environment variables in app URL"""
        return _resolve_env_vars(v)


class EmailConfig(BaseModel):
    """Main email configuration"""
    
    default_host: str = Field(..., description="Default host alias to use")
    hosts: Dict[str, EmailHostConfig] = Field(..., description="SMTP host configurations")
    
    @model_validator(mode='after')
    def validate_default_host_exists(self):
        """Ensure default_host exists in hosts"""
        if self.default_host not in self.hosts:
            raise ValueError(f"default_host '{self.default_host}' not found in hosts configuration")
        return self
    
    def get_host_config(self, alias: Optional[str] = None) -> EmailHostConfig:
        """Get host configuration by alias or default"""
        alias = alias or self.default_host
        if alias not in self.hosts:
            raise ValueError(f"Host alias '{alias}' not found in configuration")
        return self.hosts[alias]
    
    def list_host_aliases(self) -> list[str]:
        """Get list of available host aliases"""
        return list(self.hosts.keys())


class FullEmailConfig(BaseModel):
    """Complete email configuration including settings"""
    
    email: EmailConfig = Field(..., description="Email configuration")
    settings: EmailSettings = Field(default_factory=EmailSettings, description="Global settings")


def _resolve_env_vars(value: str) -> str:
    """Resolve ${ENV_VAR} placeholders in string values"""
    if not isinstance(value, str):
        return value
        
    # Pattern to match ${VAR_NAME}
    pattern = re.compile(r'\$\{([^}]+)\}')
    
    def replace_var(match):
        var_name = match.group(1)
        env_value = os.getenv(var_name)
        if env_value is None:
            logger.warning(f"Environment variable '{var_name}' not found, using placeholder")
            return match.group(0)  # Return original if not found
        return env_value
    
    return pattern.sub(replace_var, value)


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load and parse YAML configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("Configuration file must contain a YAML object")
        return config
    except FileNotFoundError:
        logger.warning(f"Email config file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing email configuration YAML: {e}")
        raise ValueError(f"Invalid YAML syntax in configuration file: {e}")
    except Exception as e:
        logger.error(f"Error loading email configuration: {e}")
        raise


def _get_config_path() -> Path:
    """Get email configuration file path from environment or default"""
    config_path_env = os.getenv("EMAIL_CONFIG_PATH")
    if config_path_env:
        return Path(config_path_env)
    
    # Default path relative to project root
    default_path = Path(__file__).parent.parent.parent / "email_config.yaml"
    return default_path


@lru_cache(maxsize=1)
def get_email_config() -> FullEmailConfig:
    """
    Get email configuration with caching
    
    Returns:
        FullEmailConfig: Validated email configuration
        
    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If configuration file not found
    """
    config_path = _get_config_path()
    
    try:
        yaml_config = _load_yaml_config(config_path)
        config = FullEmailConfig.model_validate(yaml_config)
        
        logger.info(f"Loaded email configuration from {config_path}")
        logger.info(f"Available hosts: {config.email.list_host_aliases()}")
        logger.info(f"Default host: {config.email.default_host}")
        
        return config
        
    except FileNotFoundError:
        logger.error(f"Email configuration file not found at {config_path}")
        logger.info("Email functionality will be disabled")
        raise
    except Exception as e:
        logger.error(f"Failed to load email configuration: {e}")
        raise


def is_email_configured() -> bool:
    """Check if email is properly configured"""
    try:
        get_email_config()
        return True
    except (FileNotFoundError, ValueError):
        return False


def clear_email_config_cache():
    """Clear the cached email configuration (useful for testing)"""
    get_email_config.cache_clear()
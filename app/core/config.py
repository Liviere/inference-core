from functools import lru_cache
from typing import Any, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import (
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
)

###################################
#            Classess             #
###################################


class ListParsingEnvSource(EnvSettingsSource):
    """Custom settings source that handles comma-separated lists in environment variables"""

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        """
        Parse comma-separated strings into lists for specific CORS fields
        """
        # Handle CORS list fields
        if field_name in (
            "cors_methods",
            "cors_origins",
            "cors_headers",
        ) and isinstance(value, str):
            # Handle special case of "*" which should remain as single item
            if value.strip() == "*":
                return ["*"]
            # Split by comma and clean whitespace, remove quotes
            cleaned = value.strip().strip('"').strip("'")
            return [item.strip() for item in cleaned.split(",") if item.strip()]

        # For other complex types, use default JSON parsing
        if value_is_complex:
            return self.decode_complex_value(field_name, field, value)

        return value


class ListParsingDotEnvSource(DotEnvSettingsSource):
    """Custom dotenv settings source that handles comma-separated lists"""

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        """
        Parse comma-separated strings into lists for specific CORS fields
        """
        # Handle CORS list fields
        if field_name in (
            "cors_methods",
            "cors_origins",
            "cors_headers",
        ) and isinstance(value, str):
            # Handle special case of "*" which should remain as single item
            if value.strip() == "*":
                return ["*"]
            # Split by comma and clean whitespace, remove quotes
            cleaned = value.strip().strip('"').strip("'")
            return [item.strip() for item in cleaned.split(",") if item.strip()]

        # For other complex types, use default JSON parsing
        if value_is_complex:
            return self.decode_complex_value(field_name, field, value)

        return value


class Settings(BaseSettings):
    """Application settings - all configuration in one place"""

    ###################################
    #     APPLICATION SETTINGS        #
    ###################################
    app_name: str = Field(
        default="Backend Template API", description="Application name"
    )
    app_title: str = Field(
        default="Backend Template API", description="Application title"
    )
    app_description: str = Field(
        default="A production-ready FastAPI backend template with LLM integration",
        description="Application description",
    )
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(
        default="development", description="Application environment"
    )
    debug: bool = Field(default=True, description="Debug mode")
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    ###################################
    #           CORS SETTINGS         #
    ###################################

    cors_methods: List[str] = Field(default=["*"], description="CORS allowed methods")
    cors_headers: List[str] = Field(default=["*"], description="CORS allowed headers")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")

    ###################################
    #           DATABASE              #
    ###################################
    # database_url: str = Field(
    #     default="sqlite+aiosqlite:///./app.db",
    #     description="Database connection URL",
    # )

    database_service: str = Field(
        default="sqlite+aiosqlite",
        description="Database service type (sqlite, postgresql, mysql)",
    )

    database_host: str = Field(
        default="localhost",
        description="Database host",
    )

    database_port: int = Field(
        default=5432,
        description="Database port",
    )

    database_name: str = Field(
        default="app_db",
        description="Database name",
    )

    database_user: Optional[str] = Field(
        default="db_user",
        description="Database user",
    )
    database_password: Optional[str] = Field(
        default="db_password",
        description="Database password",
    )

    database_echo: bool = Field(
        default=False,
        description="Echo SQL queries (development only)",
    )
    database_pool_size: int = Field(
        default=20,
        description="Database connection pool size",
    )
    database_max_overflow: int = Field(
        default=30,
        description="Maximum database connection overflow",
    )
    database_pool_timeout: int = Field(
        default=30,
        description="Pool connection timeout in seconds",
    )
    database_pool_recycle: int = Field(
        default=3600,
        description="Connection recycle time in seconds",
    )
    database_mysql_charset: str = Field(
        default="utf8mb4",
        description="MySQL character set",
    )
    database_url: str = Field(
        default="sqlite+aiosqlite:///./app.db",
        description="Database connection URL",
    )

    @model_validator(mode="after")
    def set_database_url(self) -> "Settings":
        """Construct database URL based on service type and other parameters"""
        if self.database_service == "sqlite+aiosqlite":
            self.database_url = f"{self.database_service}:///./app.db"
        else:
            self.database_url = (
                f"{self.database_service}://{self.database_user}:{self.database_password}"
                f"@{self.database_host}:{self.database_port}/{self.database_name}"
            )
        return self

    ###################################
    #           MONITORING            #
    ###################################
    sentry_dsn: Optional[str] = Field(
        default=None, description="Sentry DSN for error monitoring"
    )
    sentry_traces_sample_rate: float = Field(
        default=1.0,
        description="Sentry traces sample rate (0.0 to 1.0)",
    )
    sentry_profiles_sample_rate: float = Field(
        default=1.0,
        description="Sentry profiles sample rate (0.0 to 1.0)",
    )

    ###################################
    #           VALIDATORS            #
    ###################################
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        allowed_envs = ["development", "staging", "production", "testing"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of: {allowed_envs}")
        return v

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str, values: dict) -> str:
        """Validate and enhance database URL"""
        if not v:
            raise ValueError("Database URL cannot be empty")

        # Ensure async drivers are used
        url_mappings = {
            "postgresql://": "postgresql+asyncpg://",
            "mysql://": "mysql+aiomysql://",
            "sqlite:///": "sqlite+aiosqlite:///",
        }

        for old_prefix, new_prefix in url_mappings.items():
            if v.startswith(old_prefix):
                v = v.replace(old_prefix, new_prefix, 1)
                break

        return v

    ###################################
    #           PROPERTIES            #
    ###################################
    # Security / Auth
    secret_key: str = Field(
        default="change-me-in-production",
        description="Secret key for JWT signing",
    )
    algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(
        default=30, description="Access token expiration in minutes"
    )
    refresh_token_expire_days: int = Field(
        default=7, description="Refresh token expiration in days"
    )

    ###################################
    #              REDIS              #
    ###################################
    redis_url: str = Field(
        default="redis://localhost:6379/10",
        description="Redis connection URL for sessions/locks",
    )
    redis_refresh_prefix: str = Field(
        default="auth:refresh:", description="Key prefix for refresh sessions"
    )

    ###################################
    #           PROPERTIES            #
    ###################################
    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_testing(self) -> bool:
        return self.environment == "testing"

    @property
    def is_sqlite(self) -> bool:
        return "sqlite" in self.database_url.lower()

    @property
    def is_mysql(self) -> bool:
        return "mysql" in self.database_url.lower()

    @property
    def is_postgresql(self) -> bool:
        return "postgresql" in self.database_url.lower()

    @property
    def database_type(self) -> str:
        if self.is_sqlite:
            return "sqlite"
        elif self.is_mysql:
            return "mysql"
        elif self.is_postgresql:
            return "postgresql"
        else:
            return "unknown"

    def get_database_engine_args(self) -> dict:
        """Get database engine arguments based on database type"""
        base_args = {
            "echo": self.database_echo,
        }

        # SQLite doesn't support pooling
        if self.is_sqlite:
            base_args.update({"connect_args": {"check_same_thread": False}})
        else:
            # PostgreSQL and MySQL pooling settings
            base_args.update(
                {
                    "pool_size": self.database_pool_size,
                    "max_overflow": self.database_max_overflow,
                    "pool_timeout": self.database_pool_timeout,
                    "pool_recycle": self.database_pool_recycle,
                    "pool_pre_ping": True,  # Validate connections
                }
            )

            # MySQL specific settings
            if self.is_mysql:
                base_args["connect_args"] = {
                    "charset": self.database_mysql_charset,
                    "use_unicode": True,
                    "autocommit": False,
                }

        return base_args

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize settings sources to use our custom list parsing for environment variables
        """
        return (
            init_settings,
            ListParsingEnvSource(settings_cls),
            ListParsingDotEnvSource(settings_cls),
            file_secret_settings,
        )


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached)

    Returns:
        Settings instance
    """
    return Settings()

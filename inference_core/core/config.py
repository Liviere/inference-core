import os
from functools import lru_cache
from typing import Any, List, Literal, Optional

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
        Parse comma-separated strings into lists for specific CORS and ALLOWED_HOSTS fields
        """

        # Handle CORS list fields
        if field_name in (
            "cors_methods",
            "cors_origins",
            "cors_headers",
            "allowed_hosts",
        ) and isinstance(value, str):
            # Handle special case of "*" which should remain as single item
            if value.strip() == "*":
                return ["*"]
            # Split by comma and clean whitespace, remove quotes
            cleaned = value.strip().strip('"').strip("'")
            return [item.strip() for item in cleaned.split(",") if item.strip()]

        # For other complex types, use default JSON parsing
        if value_is_complex and value is not None:
            return self.decode_complex_value(field_name, field, value)

        return value


class ListParsingDotEnvSource(DotEnvSettingsSource):
    """Custom dotenv settings source that handles comma-separated lists"""

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        """
        Parse comma-separated strings into lists for specific CORS and ALLOWED_HOSTS fields
        """
        # Handle CORS and ALLOWED_HOSTS list fields
        if field_name in (
            "cors_methods",
            "cors_origins",
            "cors_headers",
            "allowed_hosts",
        ) and isinstance(value, str):
            # Handle special case of "*" which should remain as single item
            if value.strip() == "*":
                return ["*"]
            # Split by comma and clean whitespace, remove quotes
            cleaned = value.strip().strip('"').strip("'")
            return [item.strip() for item in cleaned.split(",") if item.strip()]

        # For other complex types, use default JSON parsing
        if value_is_complex and value is not None:
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
    app_public_url: str = Field(
        default="http://localhost:8000",
        description="Public URL for the application (used in emails)",
    )

    ###################################
    #           CORS SETTINGS         #
    ###################################

    cors_methods: List[str] = Field(default=["*"], description="CORS allowed methods")
    cors_headers: List[str] = Field(default=["*"], description="CORS allowed headers")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")

    ###################################
    #       TRUSTED HOST SETTINGS     #
    ###################################

    allowed_hosts: Optional[List[str]] = Field(
        default=None,
        description="Trusted hosts for TrustedHostMiddleware (separate from CORS). "
        "If not set, will derive from cors_origins by normalizing hostnames.",
    )

    ###################################
    #           DATABASE              #
    ###################################
    # database_url: str = Field(
    #     default="sqlite+aiosqlite:///./inference_core.db",
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
        default="sqlite+aiosqlite:///./inference_core.db",
        description="Database connection URL",
    )

    @model_validator(mode="after")
    def set_database_url(self) -> "Settings":
        """Construct database URL based on service type and other parameters"""
        if self.database_service == "sqlite+aiosqlite":
            self.database_url = f"{self.database_service}:///./inference_core.db"
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

    # LLM API Access Control
    llm_api_access_mode: Literal["public", "user", "superuser"] = Field(
        default="superuser",
        description="Access control mode for LLM API endpoints. 'public' allows no authentication, 'user' requires authenticated active users, 'superuser' requires superuser privileges. Default: 'superuser' for security.",
    )

    ###################################
    #        VECTOR STORE             #
    ###################################
    vector_backend: Optional[Literal["qdrant", "memory"]] = Field(
        default=None,
        description="Vector store backend. None disables vector store features. 'qdrant' for production, 'memory' for testing/development.",
    )
    vector_collection_default: str = Field(
        default="default_documents",
        description="Default collection name for vector storage",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key for authentication (optional)",
    )
    vector_distance: Literal["cosine", "euclidean", "dot"] = Field(
        default="cosine",
        description="Distance metric for vector similarity",
    )
    vector_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for text vectorization",
    )
    vector_dim: int = Field(
        default=384,
        description="Vector dimension size - must match embedding model output",
        ge=1,
        le=4096,
    )
    vector_ingest_max_batch_size: int = Field(
        default=1000,
        description="Maximum batch size for vector ingestion",
        ge=1,
        le=10000,
    )

    ###################################
    #        AGENT MEMORY             #
    ###################################
    agent_memory_enabled: bool = Field(
        default=False,
        description="Enable long-term memory for agents. Requires vector_backend to be configured.",
    )
    agent_memory_collection: str = Field(
        default="agent_memory",
        description="Collection name for agent memory storage (separate from RAG documents)",
    )
    agent_memory_max_results: int = Field(
        default=5,
        description="Maximum number of memory items to retrieve during recall",
        ge=1,
        le=50,
    )
    agent_memory_upsert_by_similarity: bool = Field(
        default=False,
        description="When True, check similarity before adding memory to avoid duplicates",
    )
    agent_memory_similarity_threshold: float = Field(
        default=0.85,
        description="Similarity threshold for upsert deduplication (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    agent_memory_auto_recall: bool = Field(
        default=True,
        description="Automatically recall relevant memories in before_agent middleware hook",
    )

    # Cookie settings for refresh tokens
    refresh_cookie_name: str = Field(
        default="refresh_token", description="Name of the refresh token cookie"
    )
    refresh_cookie_path: str = Field(
        default="/", description="Path scope for refresh token cookie"
    )
    refresh_cookie_samesite: str = Field(
        default="lax", description="SameSite setting for refresh token cookie"
    )
    refresh_cookie_domain: Optional[str] = Field(
        default=None, description="Domain for refresh token cookie"
    )

    # User activation and verification settings
    auth_register_default_active: bool = Field(
        default=True,
        description="If false, newly registered users are inactive by default",
    )
    auth_send_verification_email_on_register: bool = Field(
        default=False, description="If true, send verification email after registration"
    )
    auth_login_require_active: bool = Field(
        default=True, description="If true, login is denied for inactive users"
    )
    auth_login_require_verified: bool = Field(
        default=False, description="If true, login is denied for unverified users"
    )
    auth_email_verification_token_ttl_minutes: int = Field(
        default=60, description="Token lifetime for email verification links in minutes"
    )
    auth_email_verification_url_base: Optional[str] = Field(
        default=None,
        description="Base URL for verification links (if not set, use backend endpoint)",
    )
    auth_email_verification_makes_active: bool = Field(
        default=True,
        description="If true, verifying email also activates the user",
    )

    ###################################
    #              REDIS              #
    ###################################
    redis_url: str = Field(
        default="redis://localhost:6379/0",
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
    def refresh_cookie_secure(self) -> bool:
        """Determine if refresh token cookie should be secure (HTTPS only)"""
        return self.is_production

    @property
    def refresh_cookie_max_age(self) -> int:
        """Calculate max age for refresh token cookie in seconds"""
        return self.refresh_token_expire_days * 24 * 60 * 60

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

    def _normalize_hostname(self, url_or_hostname: str) -> str:
        """
        Extract hostname from a URL or return hostname as-is.

        Examples:
            "https://app.example.com:8080" -> "app.example.com"
            "http://localhost:3000" -> "localhost"
            "example.com" -> "example.com"
            "*" -> "*"
        """
        if url_or_hostname == "*":
            return "*"

        # Remove scheme if present
        if "://" in url_or_hostname:
            url_or_hostname = url_or_hostname.split("://", 1)[1]

        # Remove path if present (do this before port handling)
        if "/" in url_or_hostname:
            url_or_hostname = url_or_hostname.split("/", 1)[0]

        # Remove port if present
        if ":" in url_or_hostname:
            # Split only on the first colon to handle cases like IPv6
            parts = url_or_hostname.split(":", 1)
            if len(parts) == 2:
                hostname, port_part = parts
                # Only treat as port if it's purely numeric
                if port_part.isdigit():
                    url_or_hostname = hostname
                # For non-numeric or complex cases, keep as is for now

        return url_or_hostname.strip()

    def get_effective_allowed_hosts(self) -> List[str]:
        """
        Get the effective allowed hosts for TrustedHostMiddleware.

        If allowed_hosts is explicitly set, use it.
        Otherwise, derive hostnames from cors_origins.

        Returns:
            List of hostnames/IPs for TrustedHostMiddleware
        """
        if self.allowed_hosts is not None and len(self.allowed_hosts) > 0:
            return self.allowed_hosts

        # Fallback: derive from cors_origins
        if self.cors_origins == ["*"]:
            return ["*"]

        normalized_hosts = []
        for origin in self.cors_origins:
            normalized = self._normalize_hostname(origin)
            if normalized and normalized not in normalized_hosts:
                normalized_hosts.append(normalized)

        # Always include localhost/127.0.0.1 for local development and health checks
        if not self.is_production:
            for local_host in ["localhost", "127.0.0.1"]:
                if local_host not in normalized_hosts:
                    normalized_hosts.append(local_host)

        return normalized_hosts if normalized_hosts else ["*"]

    @property
    def is_vector_store_enabled(self) -> bool:
        """Check if vector store is enabled"""
        return self.vector_backend is not None

    model_config = SettingsConfigDict(
        env_file=".env.test" if os.getenv("ENVIRONMENT") == "testing" else ".env",
        case_sensitive=False,
        extra="ignore",
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

        if os.getenv("ENVIRONMENT") == "testing":
            return (
                init_settings,
                ListParsingDotEnvSource(settings_cls, env_file=".env.test"),
                ListParsingEnvSource(settings_cls),
                file_secret_settings,
            )

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


# -------------------------------------------------------------
# Helpers for tests / scenarios requiring pure default settings
# -------------------------------------------------------------
class PureDefaultsSettings(Settings):  # type: ignore
    """Settings variant ignoring ENV, .env and secrets.

    Used in tests to obtain pure defaults (with validators) and
    to create instances with overrides that go through full validation.
    """

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Only use init_settings source – no ENV, no .env, no secrets
        return (init_settings,)


def get_settings_pure_defaults() -> Settings:
    """Return instance with default values only (validators active, no ENV/.env)."""
    return PureDefaultsSettings()


def build_pure_defaults_with_overrides(**overrides: Any) -> Settings:
    """Create PureDefaultsSettings instance with overrides, all validated."""
    return PureDefaultsSettings(**overrides)


from contextlib import contextmanager


@contextmanager
def isolated_settings_environment(clear: bool = True):
    """Test context isolating ENV and get_settings cache.

    Usage:
        with isolated_settings_environment():
            s = get_settings_pure_defaults()
            ... assertions ...

    Args:
        clear: if True, removes all existing environment variables (restored after exit)
    """
    from copy import deepcopy

    original_env = deepcopy(os.environ)
    try:
        if clear:
            os.environ.clear()
        # Clear cache so get_settings() does not return a stale instance
        try:
            get_settings.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_env)
        try:
            get_settings.cache_clear()  # refresh after exit
        except Exception:
            pass

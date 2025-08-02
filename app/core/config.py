from functools import lru_cache
from typing import Any, List

from pydantic import Field, field_validator
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
        default="Backend Template API", env="APP_NAME", description="Application name"
    )
    app_title: str = Field(
        default="Backend Template API", env="APP_TITLE", description="Application title"
    )
    app_description: str = Field(
        default="A production-ready FastAPI backend template with LLM integration",
        env="APP_DESCRIPTION",
        description="Application description",
    )
    app_version: str = Field(
        default="0.1.0", env="APP_VERSION", description="Application version"
    )
    environment: str = Field(
        default="development", env="ENVIRONMENT", description="Application environment"
    )
    debug: bool = Field(default=True, env="DEBUG", description="Debug mode")
    host: str = Field(default="0.0.0.0", env="HOST", description="Server host")
    port: int = Field(default=8000, env="PORT", description="Server port")

    ###################################
    #           CORS SETTINGS         #
    ###################################

    cors_methods: List[str] = Field(
        default=["*"], env="CORS_METHODS", description="CORS allowed methods"
    )
    cors_headers: List[str] = Field(
        default=["*"], env="CORS_HEADERS", description="CORS allowed headers"
    )
    cors_origins: List[str] = Field(
        default=["*"], env="CORS_ORIGINS", description="CORS allowed origins"
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

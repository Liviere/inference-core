from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import PydanticBaseSettingsSource

###################################
#            Classess             #
###################################


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
        return (init_settings, env_settings, dotenv_settings, file_secret_settings)


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached)

    Returns:
        Settings instance
    """
    return Settings()

"""
Database Models

This module imports all database models to ensure they are
registered with SQLAlchemy metadata for table creation.
"""

# Import all models here so SQLAlchemy can discover them
from .batch import BatchEvent, BatchItem, BatchJob
from .llm_config import (
    AllowedUserOverride,
    ConfigScope,
    LLMConfigOverride,
    UserLLMPreference,
    UserLLMPreferenceType,
)
from .llm_request_log import LLMRequestLog
from .model_catalog import LLMCatalogEvent, LLMCatalogModel, LLMCatalogProviderState
from .pricing_snapshot import LLMPricingSnapshot
from .user import User
from .user_agent_instance import UserAgentInstance

# Export all models for easy importing
__all__ = [
    "User",
    "BatchJob",
    "BatchItem",
    "BatchEvent",
    "LLMRequestLog",
    "LLMPricingSnapshot",
    # Model catalog
    "LLMCatalogModel",
    "LLMCatalogEvent",
    "LLMCatalogProviderState",
    # LLM Config models
    "LLMConfigOverride",
    "UserLLMPreference",
    "AllowedUserOverride",
    "ConfigScope",
    "UserLLMPreferenceType",
    # User Agent Instance
    "UserAgentInstance",
]

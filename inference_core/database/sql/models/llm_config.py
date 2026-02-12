"""
LLM Configuration Models

Database models for dynamic LLM configuration overrides.
Supports multi-layer configuration: YAML base → Admin DB overrides → User preferences.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import BaseModel, SmartJSON


class ConfigScope(str, Enum):
    """
    Defines scope levels for configuration overrides.

    WHY: Allows granular control over what config is being overridden -
    from global defaults to specific model/task/agent settings.
    """

    GLOBAL = "global"  # Affects all LLM operations
    MODEL = "model"  # Specific model (e.g., gpt-5-mini)
    TASK = "task"  # Specific task (e.g., chat, completion)
    AGENT = "agent"  # Specific agent configuration


class LLMConfigOverride(BaseModel):
    """
    Admin-level configuration overrides stored in database.

    WHY: Enables runtime configuration changes without server restart.
    Admins can adjust global settings, enable/disable models, or tune
    parameters for specific tasks without touching YAML files.

    Resolution order (later wins):
    1. YAML base config
    2. Admin DB overrides (this model)
    3. User preferences (UserLLMPreference)
    """

    __tablename__ = "llm_config_overrides"

    # Scope definition
    scope: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        doc="Override scope: global, model, task, or agent",
    )
    scope_key: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        doc="Scope identifier (e.g., 'gpt-5-mini' for model scope, 'chat' for task scope)",
    )

    # Configuration key-value
    config_key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        doc="Configuration key (e.g., 'temperature', 'max_tokens', 'enabled')",
    )
    config_value: Mapped[Dict[str, Any]] = mapped_column(
        SmartJSON,
        nullable=False,
        doc="Configuration value as JSON (supports complex nested values)",
    )

    # Priority for conflict resolution (higher = more important)
    priority: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        index=True,
        doc="Priority for resolving conflicts (higher wins)",
    )

    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        doc="Whether this override is currently active",
    )

    # Audit fields
    created_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        doc="Admin user who created this override",
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Human-readable description of why this override exists",
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Optional expiration timestamp for temporary overrides",
    )

    # Relationships
    created_by = relationship("User", foreign_keys=[created_by_id])

    __table_args__ = (
        # Ensure unique override per scope/key combination
        UniqueConstraint(
            "scope",
            "scope_key",
            "config_key",
            name="uq_llm_config_override_scope_key",
        ),
        Index("ix_llm_config_overrides_active_scope", "is_active", "scope"),
        Index("ix_llm_config_overrides_expires", "expires_at", "is_active"),
    )

    def is_expired(self) -> bool:
        """Check if this override has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(self.expires_at.tzinfo) > self.expires_at


class UserLLMPreferenceType(str, Enum):
    """
    Types of user preferences for LLM configuration.

    WHY: Categorizes preferences for easier querying and validation.
    """

    DEFAULT_MODEL = "default_model"  # User's preferred default model
    MODEL_PARAMS = "model_params"  # Parameters for specific model
    TASK_PARAMS = "task_params"  # Parameters for specific task
    AGENT_PARAMS = "agent_params"  # Parameters for specific agent


class UserLLMPreference(BaseModel):
    """
    User-specific LLM configuration preferences.

    WHY: Allows registered users to personalize their LLM experience.
    Users can set preferred models, default temperatures, token limits etc.
    within admin-defined boundaries (allowlist).

    SECURITY: User preferences are validated against an allowlist before
    being applied. Users cannot override admin-restricted settings.
    """

    __tablename__ = "user_llm_preferences"

    # User reference
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="User who owns this preference",
    )

    # Preference definition
    preference_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        index=True,
        doc="Type of preference: default_model, model_params, task_params, agent_params",
    )
    preference_key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        doc="Preference key (e.g., 'chat.temperature', 'completion.max_tokens')",
    )
    preference_value: Mapped[Dict[str, Any]] = mapped_column(
        SmartJSON,
        nullable=False,
        doc="Preference value as JSON",
    )

    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        doc="Whether this preference is currently active",
    )

    # Relationships
    user = relationship("User", foreign_keys=[user_id])

    __table_args__ = (
        # One preference per user/type/key combination
        UniqueConstraint(
            "user_id",
            "preference_type",
            "preference_key",
            name="uq_user_llm_preference_user_type_key",
        ),
        Index("ix_user_llm_preferences_user_active", "user_id", "is_active"),
    )


class AllowedUserOverride(BaseModel):
    """
    Defines which configuration keys users are allowed to override.

    WHY: Security boundary - admins define what users can customize.
    Prevents users from overriding sensitive settings like API keys,
    pricing, or security configurations.

    Supports constraints on allowed values (min/max for numbers,
    allowlist for strings).
    """

    __tablename__ = "allowed_user_overrides"

    # What can be overridden
    config_key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
        doc="Configuration key that users can override (e.g., 'temperature', 'max_tokens')",
    )

    # Value constraints (stored as JSON)
    constraints: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        SmartJSON,
        nullable=True,
        doc="Validation constraints: {min, max, allowed_values, type}",
    )

    # Scope restrictions
    allowed_scopes: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        SmartJSON,
        nullable=True,
        doc="Which scopes this override applies to (null = all)",
    )

    # Description for UI/documentation
    display_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Human-readable name for UI display",
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Description of what this setting does",
    )

    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        doc="Whether users can currently use this override",
    )

    __table_args__ = (
        Index("ix_allowed_user_overrides_active", "is_active", "config_key"),
    )

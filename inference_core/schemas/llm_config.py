"""
LLM Configuration Schemas

Pydantic schemas for LLM configuration API endpoints.
Handles validation for user preferences and admin overrides.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ============================================================
# Enums (mirror DB enums for API validation)
# ============================================================


class ConfigScopeEnum(str, Enum):
    """Configuration scope levels."""

    GLOBAL = "global"
    MODEL = "model"
    TASK = "task"
    AGENT = "agent"


class PreferenceTypeEnum(str, Enum):
    """User preference types."""

    DEFAULT_MODEL = "default_model"
    MODEL_PARAMS = "model_params"
    TASK_PARAMS = "task_params"
    AGENT_PARAMS = "agent_params"


# ============================================================
# Constraint Schemas (for AllowedUserOverride validation)
# ============================================================


class NumericConstraints(BaseModel):
    """Constraints for numeric configuration values."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["number"] = "number"
    min: Optional[float] = Field(None, description="Minimum allowed value")
    max: Optional[float] = Field(None, description="Maximum allowed value")
    step: Optional[float] = Field(None, description="Step increment for UI sliders")


class StringConstraints(BaseModel):
    """Constraints for string configuration values."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["string"] = "string"
    allowed_values: Optional[List[str]] = Field(
        None, description="List of allowed string values"
    )
    pattern: Optional[str] = Field(None, description="Regex pattern for validation")
    max_length: Optional[int] = Field(None, description="Maximum string length")


class BooleanConstraints(BaseModel):
    """Constraints for boolean configuration values."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["boolean"] = "boolean"


class SelectConstraints(BaseModel):
    """Constraints for select configuration values (enumerated choices)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["select"] = "select"
    allowed_values: List[Any] = Field(
        ..., description="List of allowed values (required for select type)"
    )


class ConstraintsUnion(BaseModel):
    """
    Union of possible constraint types.

    WHY: Provides flexible validation rules that can be defined by admins
    and enforced when users set preferences.
    """

    type: str = Field(..., description="Value type: number, string, boolean, select")
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    max_length: Optional[int] = None


# Discriminated union for type-safe constraints
ConstraintsUnion = Union[
    NumericConstraints, StringConstraints, BooleanConstraints, SelectConstraints
]


# ============================================================
# User Preference Schemas
# ============================================================


class UserPreferenceBase(BaseModel):
    """Base schema for user preferences."""

    preference_type: PreferenceTypeEnum = Field(..., description="Type of preference")
    preference_key: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Preference key (e.g., 'chat.temperature')",
    )
    preference_value: Dict[str, Any] = Field(
        ..., description="Preference value as JSON object"
    )


class UserPreferenceCreate(UserPreferenceBase):
    """Schema for creating a user preference."""

    pass


class UserPreferenceUpdate(BaseModel):
    """Schema for updating a user preference."""

    preference_value: Dict[str, Any] = Field(..., description="New preference value")
    is_active: Optional[bool] = Field(None, description="Whether preference is active")


class UserPreferenceResponse(UserPreferenceBase):
    """Schema for user preference API response."""

    id: UUID
    user_id: UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserPreferenceBulkUpdate(BaseModel):
    """Schema for bulk updating user preferences."""

    preferences: List[UserPreferenceCreate] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of preferences to create/update",
    )


# ============================================================
# Admin Override Schemas
# ============================================================


class AdminOverrideBase(BaseModel):
    """Base schema for admin configuration overrides."""

    scope: ConfigScopeEnum = Field(..., description="Override scope")
    scope_key: Optional[str] = Field(
        None,
        max_length=100,
        description="Scope identifier (required for non-global scopes)",
    )
    config_key: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Configuration key to override",
    )
    config_value: Dict[str, Any] = Field(..., description="Configuration value as JSON")
    priority: int = Field(
        default=0,
        ge=-100,
        le=100,
        description="Priority for conflict resolution",
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Human-readable description",
    )
    expires_at: Optional[datetime] = Field(
        None, description="Optional expiration timestamp"
    )

    @model_validator(mode="after")
    def validate_scope_key(self):
        """Ensure scope_key is provided for non-global scopes."""
        if self.scope != ConfigScopeEnum.GLOBAL and not self.scope_key:
            raise ValueError(f"scope_key is required for {self.scope.value} scope")
        return self


class AdminOverrideCreate(AdminOverrideBase):
    """Schema for creating an admin override."""

    pass


class AdminOverrideUpdate(BaseModel):
    """Schema for updating an admin override."""

    config_value: Optional[Dict[str, Any]] = None
    priority: Optional[int] = Field(None, ge=-100, le=100)
    description: Optional[str] = Field(None, max_length=500)
    expires_at: Optional[datetime] = None
    is_active: Optional[bool] = None


class AdminOverrideResponse(AdminOverrideBase):
    """Schema for admin override API response."""

    id: UUID
    is_active: bool
    created_by_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================
# Allowed Override Schemas (Admin manages what users can change)
# ============================================================


class AllowedOverrideBase(BaseModel):
    """Base schema for allowed user overrides."""

    config_key: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Configuration key users can override",
    )
    constraints: Optional[ConstraintsUnion] = Field(
        None, discriminator="type", description="Validation constraints for this key"
    )
    allowed_scopes: Optional[List[ConfigScopeEnum]] = Field(
        None, description="Which scopes this override applies to"
    )
    display_name: Optional[str] = Field(
        None, max_length=100, description="Human-readable name"
    )
    description: Optional[str] = Field(
        None, max_length=500, description="Description for users"
    )


class AllowedOverrideCreate(AllowedOverrideBase):
    """Schema for creating an allowed override definition."""

    pass


class AllowedOverrideUpdate(BaseModel):
    """Schema for updating an allowed override definition."""

    constraints: Optional[ConstraintsUnion] = Field(None, discriminator="type")
    allowed_scopes: Optional[List[ConfigScopeEnum]] = None
    display_name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None


class AllowedOverrideResponse(AllowedOverrideBase):
    """Schema for allowed override API response."""

    id: UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================
# Resolved Configuration Schemas
# ============================================================


class ResolvedModelConfig(BaseModel):
    """Resolved configuration for a specific model."""

    name: str
    provider: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    # Additional params that may come from user preferences
    extra_params: Optional[Dict[str, Any]] = None


class ResolvedTaskConfig(BaseModel):
    """Resolved configuration for a specific task."""

    primary_model: str
    fallback_models: Optional[List[str]] = None
    default_params: Optional[Dict[str, Any]] = None


class ResolvedConfigResponse(BaseModel):
    """
    Full resolved configuration for a user.

    WHY: Provides a complete view of effective configuration after
    merging YAML → admin overrides → user preferences.
    """

    # Source tracking for debugging
    sources: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of config keys to their source (yaml/admin/user)",
    )

    # Available models for this user
    available_models: List[str] = Field(
        default_factory=list,
        description="List of model names user can access",
    )

    # Task configurations
    tasks: Dict[str, ResolvedTaskConfig] = Field(
        default_factory=dict,
        description="Resolved task configurations",
    )

    # User's effective default settings
    defaults: Dict[str, Any] = Field(
        default_factory=dict,
        description="User's effective default parameters",
    )

    # Cache metadata
    cached_at: Optional[datetime] = None
    cache_ttl_seconds: Optional[int] = None


# ============================================================
# API Request/Response Wrappers
# ============================================================


class PreferenceListResponse(BaseModel):
    """Response for listing user preferences."""

    preferences: List[UserPreferenceResponse]
    total: int


class AdminOverrideListResponse(BaseModel):
    """Response for listing admin overrides."""

    overrides: List[AdminOverrideResponse]
    total: int


class AllowedOverrideListResponse(BaseModel):
    """Response for listing allowed overrides."""

    allowed_overrides: List[AllowedOverrideResponse]
    total: int


class AvailableOptionsResponse(BaseModel):
    """
    Response listing what options users can configure.

    WHY: Frontend needs to know what preferences are available
    and what constraints apply to build dynamic settings UI.
    """

    options: List[AllowedOverrideResponse] = Field(
        ..., description="List of configurable options with constraints"
    )
    available_models: List[str] = Field(
        ..., description="Models the user can select as defaults"
    )
    available_tasks: List[str] = Field(..., description="Tasks that can be configured")


class ConfigValidationError(BaseModel):
    """Validation error for configuration values."""

    key: str
    message: str
    constraint_violated: Optional[str] = None

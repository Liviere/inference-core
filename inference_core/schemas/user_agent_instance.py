"""
User Agent Instance Schemas

Pydantic schemas for the user agent instance API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ============================================================
# Request Schemas
# ============================================================


class AgentInstanceCreate(BaseModel):
    """Schema for creating a new user agent instance."""

    instance_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique instance name (slug-like identifier, e.g., 'creative-writer')",
    )
    display_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human-readable display name",
    )
    base_agent_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the base agent from config (e.g., 'assistant_agent')",
    )
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Description of what this agent instance is for",
    )
    primary_model: Optional[str] = Field(
        None,
        max_length=100,
        description="Override for the primary LLM model",
    )
    system_prompt_override: Optional[str] = Field(
        None,
        max_length=5000,
        description="Full system prompt override (replaces base prompt)",
    )
    system_prompt_append: Optional[str] = Field(
        None,
        max_length=2000,
        description="Text appended to base system prompt",
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Additional overrides: fallback (list), allowed_tools (list), "
            "mcp_profile (str), temperature (float), max_tokens (int)"
        ),
    )
    is_default: bool = Field(
        default=False,
        description="Set as default agent for new chats",
    )
    is_deepagent: bool = Field(
        default=False,
        description="Whether this instance is a deep agent",
    )
    subagent_ids: Optional[List[UUID]] = Field(
        None,
        description="List of subagent instance IDs (only valid if is_deepagent=True)",
    )

    @field_validator("instance_name")
    @classmethod
    def validate_instance_name(cls, v: str) -> str:
        """Ensure instance_name is a valid slug."""
        import re

        if not re.match(r"^[a-z0-9][a-z0-9_-]*$", v):
            raise ValueError(
                "instance_name must start with a letter/digit and contain only "
                "lowercase letters, digits, hyphens, and underscores"
            )
        return v


class AgentInstanceUpdate(BaseModel):
    """Schema for updating an existing agent instance."""

    display_name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    primary_model: Optional[str] = Field(None, max_length=100)
    system_prompt_override: Optional[str] = Field(None, max_length=5000)
    system_prompt_append: Optional[str] = Field(None, max_length=2000)
    config_overrides: Optional[Dict[str, Any]] = None
    is_default: Optional[bool] = None
    is_deepagent: Optional[bool] = None
    subagent_ids: Optional[List[UUID]] = None
    is_active: Optional[bool] = None


# ============================================================
# Response Schemas
# ============================================================


class AgentInstanceResponse(BaseModel):
    """Schema for agent instance API response."""

    id: UUID
    user_id: UUID
    instance_name: str
    display_name: str
    base_agent_name: str
    description: Optional[str] = None
    primary_model: Optional[str] = None
    system_prompt_override: Optional[str] = None
    system_prompt_append: Optional[str] = None
    config_overrides: Optional[Dict[str, Any]] = None
    is_default: bool
    is_deepagent: bool
    subagents: Optional[List[AgentInstanceResponse]] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AgentInstanceListResponse(BaseModel):
    """Response for listing agent instances."""

    instances: List[AgentInstanceResponse]
    total: int


# ============================================================
# Agent Template Schemas (read-only, from YAML config)
# ============================================================


class AgentTemplateResponse(BaseModel):
    """Schema describing a base agent template from YAML config."""

    agent_name: str = Field(..., description="Agent name from config")
    primary_model: str = Field(..., description="Default primary model")
    fallback_models: Optional[List[str]] = Field(None, description="Fallback models")
    description: str = Field(default="", description="Agent description")
    allowed_tools: Optional[List[str]] = Field(None, description="Available tools")
    mcp_profile: Optional[str] = Field(None, description="MCP profile name")
    local_tool_providers: Optional[List[str]] = Field(
        None, description="Local tool provider names"
    )
    skills: Optional[List[str]] = Field(
        None, description="Pre-defined skills in template"
    )
    subagents: Optional[List[str]] = Field(
        None, description="Pre-defined subagent names in template"
    )


class AgentTemplateListResponse(BaseModel):
    """Response for listing available agent templates."""

    templates: List[AgentTemplateResponse]
    available_models: List[str] = Field(
        default_factory=list,
        description="All models available for selection",
    )

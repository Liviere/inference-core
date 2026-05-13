"""
User Agent Instance Schemas

Pydantic schemas for the user agent instance API endpoints.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from inference_core.schemas.agent_prompt_limits import (
    validate_agent_prompt_length,
)

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_MAX_USER_SKILLS = 20
_MAX_SKILL_DESCRIPTION_LENGTH = 1024
_MAX_SKILL_CONTENT_LENGTH = 100_000


def _validate_skill_entries(
    skills: Optional[List[Dict[str, str]]],
) -> Optional[List[Dict[str, str]]]:
    """Validate user-defined skills before storing them as SKILL.md files."""
    if skills is None:
        return skills
    if len(skills) > _MAX_USER_SKILLS:
        raise ValueError(f"At most {_MAX_USER_SKILLS} skills are allowed")

    seen_names: set[str] = set()
    normalized_skills: list[dict[str, str]] = []
    for index, skill in enumerate(skills):
        if not isinstance(skill, dict):
            raise ValueError(f"Skill at index {index} must be a dict")

        normalized_skill = dict(skill)
        for key in ("name", "description", "content"):
            value = normalized_skill.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Skill at index {index} missing required field '{key}'"
                )
            normalized_skill[key] = value.strip() if key != "content" else value

        name = normalized_skill["name"]
        if not _SKILL_NAME_PATTERN.match(name) or "--" in name:
            raise ValueError(
                f"Skill at index {index} has invalid name '{name}'. "
                "Use lowercase letters, digits, and single hyphens only"
            )
        if name in seen_names:
            raise ValueError(f"Duplicate skill name '{name}'")
        seen_names.add(name)

        description = normalized_skill["description"]
        if len(description) > _MAX_SKILL_DESCRIPTION_LENGTH:
            raise ValueError(
                f"Skill '{name}' description exceeds "
                f"{_MAX_SKILL_DESCRIPTION_LENGTH} characters"
            )

        content = normalized_skill["content"]
        if len(content) > _MAX_SKILL_CONTENT_LENGTH:
            raise ValueError(
                f"Skill '{name}' content exceeds {_MAX_SKILL_CONTENT_LENGTH} characters"
            )

        normalized_skills.append(normalized_skill)

    return normalized_skills


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
        description="Full system prompt override (replaces base prompt)",
    )
    system_prompt_append: Optional[str] = Field(
        None,
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
    skills: Optional[List[Dict[str, str]]] = Field(
        None,
        description=(
            "User-defined skills. Each entry: "
            "{'name': str, 'description': str, 'content': str (SKILL.md content)}"
        ),
    )
    is_deepagent: bool = Field(
        default=False,
        description="Whether this instance is a deep agent",
    )
    subagent_ids: Optional[List[UUID]] = Field(
        None,
        description="List of subagent instance IDs (only valid if is_deepagent=True)",
    )

    @field_validator("skills")
    @classmethod
    def validate_skills(
        cls, v: Optional[List[Dict[str, str]]]
    ) -> Optional[List[Dict[str, str]]]:
        """Validate that each skill entry is safe to expose as a SKILL.md file."""
        return _validate_skill_entries(v)

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

    @field_validator("system_prompt_override", "system_prompt_append")
    @classmethod
    def validate_prompt_lengths(
        cls, value: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        """Validate prompt fields against optional runtime-configured limits."""

        validate_agent_prompt_length(info.field_name, value)
        return value


class AgentInstanceUpdate(BaseModel):
    """Schema for updating an existing agent instance."""

    display_name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    primary_model: Optional[str] = Field(None, max_length=100)
    system_prompt_override: Optional[str] = None
    system_prompt_append: Optional[str] = None
    config_overrides: Optional[Dict[str, Any]] = None
    skills: Optional[List[Dict[str, str]]] = None
    is_default: Optional[bool] = None
    is_deepagent: Optional[bool] = None
    subagent_ids: Optional[List[UUID]] = None
    is_active: Optional[bool] = None

    @field_validator("skills")
    @classmethod
    def validate_skills(
        cls, v: Optional[List[Dict[str, str]]]
    ) -> Optional[List[Dict[str, str]]]:
        """Validate that each skill entry is safe to expose as a SKILL.md file."""
        return _validate_skill_entries(v)

    @field_validator("system_prompt_override", "system_prompt_append")
    @classmethod
    def validate_prompt_lengths(
        cls, value: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        """Validate prompt fields against optional runtime-configured limits."""

        validate_agent_prompt_length(info.field_name, value)
        return value


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
    skills: Optional[List[Dict[str, str]]] = None
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


# ============================================================
# Agent Run Schemas
# ============================================================


class AgentInstanceRunRequest(BaseModel):
    """Schema for running an agent instance."""

    user_input: str = Field(
        ..., min_length=1, description="User message to send to the agent"
    )
    system_prompt: Optional[str] = Field(
        None,
        max_length=5000,
        description="Optional base system prompt (DB overrides still apply on top)",
    )


class AgentInstanceRunResponse(BaseModel):
    """Schema for agent instance run response."""

    result: Dict[str, Any]
    steps: List[Dict[str, Any]]
    model_name: str
    instance_id: UUID
    instance_name: str
    cost_metrics: Optional[Dict[str, Any]] = None


# ============================================================
# Run Bundle (frontend handshake)
# ============================================================


class RunBundleConfig(BaseModel):
    """Wrapper around the ``configurable`` dict consumed by the Agent Server.

    Mirrors the LangGraph SDK ``RunnableConfig`` shape so the frontend can
    feed it straight into ``useStream({ defaultConfig: bundle.config })``.
    """

    configurable: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Runtime overrides forwarded to the Agent Server via "
            "``config.configurable`` — read by InstanceConfigMiddleware, "
            "MemoryMiddleware, SubagentConfigMiddleware, etc."
        ),
    )


class RunBundleResponse(BaseModel):
    """Everything the frontend needs to open a ``useStream`` connection.

    WHY: Hides backend wiring from the frontend.  When the user picks an
    agent instance, the UI calls this endpoint once and receives:

      * ``assistant_id`` — graph identifier on the Agent Server
      * ``agent_server_url`` — base URL where ``useStream`` should connect
      * ``access_token`` — short-lived JWT for the LangGraph auth handler
      * ``config.configurable`` — instance overrides (model, prompts,
        subagent_configs, memory flags) that the middleware stack honours

    The frontend then opens a ``useStream`` session.  No additional backend
    round-trips are required for the lifetime of that conversation.
    """

    instance_id: UUID
    instance_name: str
    display_name: str
    base_agent_name: str
    description: Optional[str] = None

    assistant_id: str = Field(
        ...,
        description=(
            "Graph identifier on the Agent Server (matches the YAML "
            "``remote_graph_id`` or the agent name)."
        ),
    )
    agent_server_url: str = Field(
        ...,
        description="Base URL of the LangGraph Agent Server.",
    )
    access_token: Optional[str] = Field(
        None,
        description=(
            "JWT to attach as ``Authorization: Bearer <token>`` when "
            "connecting to the Agent Server.  Echoed from the inbound "
            "Authorization header so the same token flows through."
        ),
    )

    config: RunBundleConfig = Field(
        default_factory=RunBundleConfig,
        description="Runtime overrides for this instance.",
    )

    is_remote: bool = Field(
        ...,
        description=(
            "Whether the agent will be served by the Agent Server "
            "(execution_mode='remote' + AGENT_SERVER_ENABLED=True)."
        ),
    )

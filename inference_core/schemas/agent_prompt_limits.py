"""Optional runtime- or environment-configured limits for agent prompts."""

from dataclasses import dataclass
import os
from typing import Optional


@dataclass(frozen=True)
class AgentPromptLimits:
    """Optional max-length constraints for configurable prompt fields."""

    system_prompt_override: Optional[int] = None
    system_prompt_append: Optional[int] = None


AGENT_SYSTEM_PROMPT_OVERRIDE_MAX_LENGTH_ENV_VAR = (
    "INFERENCE_CORE_AGENT_SYSTEM_PROMPT_OVERRIDE_MAX_LENGTH"
)
AGENT_SYSTEM_PROMPT_APPEND_MAX_LENGTH_ENV_VAR = (
    "INFERENCE_CORE_AGENT_SYSTEM_PROMPT_APPEND_MAX_LENGTH"
)


_FIELD_TO_LIMIT_ATTR = {
    "system_prompt_override": "system_prompt_override",
    "system_prompt_append": "system_prompt_append",
}

_FIELD_TO_ENV_VAR = {
    "system_prompt_override": AGENT_SYSTEM_PROMPT_OVERRIDE_MAX_LENGTH_ENV_VAR,
    "system_prompt_append": AGENT_SYSTEM_PROMPT_APPEND_MAX_LENGTH_ENV_VAR,
}

_configured_agent_prompt_limits = AgentPromptLimits()
_has_runtime_agent_prompt_limits = False


def _validate_limit_value(field_name: str, value: Optional[int]) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{field_name} limit must be a positive integer")


def _read_agent_prompt_limit_from_env(field_name: str) -> Optional[int]:
    env_name = _FIELD_TO_ENV_VAR[field_name]
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return None

    stripped_value = raw_value.strip()
    if not stripped_value:
        return None

    try:
        parsed_value = int(stripped_value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be an integer") from exc

    if parsed_value < 1:
        raise ValueError(f"{env_name} must be a positive integer")

    return parsed_value


def get_agent_prompt_limits_from_env() -> AgentPromptLimits:
    """Return prompt limits derived only from environment variables."""

    return AgentPromptLimits(
        system_prompt_override=_read_agent_prompt_limit_from_env(
            "system_prompt_override"
        ),
        system_prompt_append=_read_agent_prompt_limit_from_env("system_prompt_append"),
    )


def clear_agent_prompt_limits_runtime_override() -> None:
    """Clear runtime overrides so env-based defaults become visible again."""

    global _configured_agent_prompt_limits, _has_runtime_agent_prompt_limits
    _configured_agent_prompt_limits = AgentPromptLimits()
    _has_runtime_agent_prompt_limits = False


def configure_agent_prompt_limits(
    *,
    system_prompt_override: Optional[int] = None,
    system_prompt_append: Optional[int] = None,
) -> None:
    """Configure optional prompt limits for the current runtime."""

    _validate_limit_value("system_prompt_override", system_prompt_override)
    _validate_limit_value("system_prompt_append", system_prompt_append)

    global _configured_agent_prompt_limits, _has_runtime_agent_prompt_limits
    _configured_agent_prompt_limits = AgentPromptLimits(
        system_prompt_override=system_prompt_override,
        system_prompt_append=system_prompt_append,
    )
    _has_runtime_agent_prompt_limits = True


def get_agent_prompt_limits() -> AgentPromptLimits:
    """Return the currently configured prompt limits for this runtime."""

    if not _has_runtime_agent_prompt_limits:
        return get_agent_prompt_limits_from_env()

    return _configured_agent_prompt_limits


def validate_agent_prompt_length(field_name: str, value: Optional[str]) -> None:
    """Raise a ValueError when a prompt field exceeds its configured limit."""

    if value is None:
        return

    limit_attr = _FIELD_TO_LIMIT_ATTR[field_name]
    max_length = getattr(get_agent_prompt_limits(), limit_attr)
    if max_length is None:
        return

    if len(value) > max_length:
        raise ValueError(f"{field_name} exceeds {max_length} characters")


def validate_agent_prompt_lengths(
    *,
    system_prompt_override: Optional[str] = None,
    system_prompt_append: Optional[str] = None,
) -> None:
    """Validate all user-configurable prompt fields against shared limits."""

    validate_agent_prompt_length("system_prompt_override", system_prompt_override)
    validate_agent_prompt_length("system_prompt_append", system_prompt_append)

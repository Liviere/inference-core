"""
User Agent Instance Service

CRUD service for managing user-created agent configuration instances.
Handles validation, default management, and template listing.
"""

import logging
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Tuple,
)
from uuid import UUID

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from inference_core.agents.middleware.model_fallback import (
    canonicalize_fallback_overrides,
)
from inference_core.database.sql.models.user_agent_instance import UserAgentInstance
from inference_core.llm.config import get_llm_config
from inference_core.schemas.agent_prompt_limits import validate_agent_prompt_lengths
from inference_core.services.llm_config_service import LLMConfigService

logger = logging.getLogger(__name__)


INSTANCE_LOAD_OPTIONS = (
    selectinload(UserAgentInstance.subagents).selectinload(UserAgentInstance.subagents),
)


@dataclass(frozen=True)
class InstanceCheck:
    """What an instance is about to become, for the checks an application adds.

    ``instance_id``, ``previous_base_agent_name`` and ``previous_overrides``
    are None when the instance is being created. ``actor_is_superuser`` is
    None when the caller did not say who is acting.

    ``primary_model``, ``is_deepagent``, ``skills`` and ``subagent_ids`` are
    what the instance will have once saved: the new value where it changes,
    the stored one where it does not. ``previous_primary_model`` is None when
    the instance is being created.
    """

    user_id: UUID
    instance_id: Optional[UUID]
    base_agent_name: str
    config_overrides: Dict[str, Any]
    previous_base_agent_name: Optional[str]
    previous_overrides: Optional[Dict[str, Any]]
    actor_is_superuser: Optional[bool]
    primary_model: Optional[str] = None
    previous_primary_model: Optional[str] = None
    is_deepagent: bool = False
    skills: Optional[Tuple[Dict[str, str], ...]] = None
    subagent_ids: Tuple[UUID, ...] = ()

    def changed(self, key: str) -> bool:
        """Whether ``key`` of config_overrides is new, changed or removed."""
        previous = self.previous_overrides or {}
        return self.config_overrides.get(key) != previous.get(key)

    @property
    def type_changed(self) -> bool:
        return (
            self.previous_base_agent_name is not None
            and self.base_agent_name != self.previous_base_agent_name
        )


# What an application adds to the life of an instance: checks of its own on
# config_overrides, models it lets a user's instances name besides the
# configured ones, and work to do once an instance is deleted (what it kept
# elsewhere under its id).
OverridesValidator = Callable[[AsyncSession, UUID, Dict[str, Any]], Awaitable[None]]
InstanceValidator = Callable[[AsyncSession, InstanceCheck], Awaitable[None]]
InstanceDeletedHook = Callable[[UUID, UUID], Awaitable[None]]
ExtraModelsSource = Callable[[AsyncSession, UUID], Awaitable[Collection[str]]]

#: What else, besides config_overrides and the base agent, may make an
#: instance validator run again (see :func:`register_instance_validator`).
INSTANCE_CHANGES: FrozenSet[str] = frozenset(
    {"primary_model", "is_deepagent", "skills", "subagents"}
)

_overrides_validators: List[OverridesValidator] = []
_instance_validators: List[InstanceValidator] = []
_instance_validator_triggers: Dict[InstanceValidator, FrozenSet[str]] = {}
_instance_deleted_hooks: List[InstanceDeletedHook] = []
_extra_models_sources: List[ExtraModelsSource] = []


def register_config_overrides_validator(validator: OverridesValidator) -> None:
    """Check ``config_overrides`` when an instance is created or changed.

    ``validator(db, user_id, overrides)`` raises ``ValueError`` to refuse;
    the message reaches the caller as it is.
    """
    if validator not in _overrides_validators:
        _overrides_validators.append(validator)


def register_instance_validator(
    validator: InstanceValidator, *, also_on: Iterable[str] = ()
) -> None:
    """Check an instance when it is created, or when its config_overrides or
    its base agent changes.

    ``validator(db, check)`` gets the :class:`InstanceCheck` and raises
    ``ValueError`` to refuse; the message reaches the caller as it is.

    ``also_on`` names other changes that run it too, from
    :data:`INSTANCE_CHANGES`: ``primary_model``, ``is_deepagent``, ``skills``
    and ``subagents``. By default a change of those alone runs no check.
    """
    triggers = frozenset(also_on)
    unknown = triggers - INSTANCE_CHANGES
    if unknown:
        raise ValueError(f"Unknown instance changes: {sorted(unknown)}")
    if validator not in _instance_validators:
        _instance_validators.append(validator)
    _instance_validator_triggers[validator] = triggers


def register_extra_models_source(source: ExtraModelsSource) -> None:
    """Let a user's instances name models the configuration does not list.

    ``source(db, user_id)`` returns the model names that user's instances may
    have as their primary model besides the available ones, e.g. models the
    application registers for that user at run time. It is asked only for a
    model the configuration does not have. Which instance may use which model
    is for an instance validator to say (``also_on={"primary_model"}``).
    """
    if source not in _extra_models_sources:
        _extra_models_sources.append(source)


def register_instance_deleted_hook(hook: InstanceDeletedHook) -> None:
    """Call ``hook(user_id, instance_id)`` after an instance is deleted.

    It runs after the commit, so a failure cannot keep the instance; it is
    logged and the others still run.
    """
    if hook not in _instance_deleted_hooks:
        _instance_deleted_hooks.append(hook)


async def _run_overrides_validators(
    db: AsyncSession, user_id: UUID, overrides: Optional[Dict[str, Any]]
) -> None:
    for validator in list(_overrides_validators):
        await validator(db, user_id, dict(overrides or {}))


def _check_mcp_profile(check: InstanceCheck) -> None:
    """Refuse an MCP profile the instance may not use.

    Asked only when the profile is set and it, or the base agent, changes: an
    instance keeps a profile that was allowed when it was saved.
    """
    profile = check.config_overrides.get("mcp_profile")
    if not profile:
        return
    if not isinstance(profile, str):
        raise ValueError("config_overrides.mcp_profile must be a profile name")
    if not (check.changed("mcp_profile") or check.type_changed):
        return
    problem = get_llm_config().mcp_profile_problem(
        profile, check.base_agent_name, is_superuser=check.actor_is_superuser
    )
    if problem:
        raise ValueError(problem)


async def _run_instance_checks(
    db: AsyncSession,
    check: InstanceCheck,
    only_on: Optional[FrozenSet[str]] = None,
) -> None:
    """Run the checks on what an instance is about to become.

    ``only_on`` is set when nothing changes but some of
    :data:`INSTANCE_CHANGES`: then only the validators registered for one of
    those run.
    """
    if only_on is None:
        _check_mcp_profile(check)
    for validator in list(_instance_validators):
        if only_on is not None and not (
            _instance_validator_triggers.get(validator, frozenset()) & only_on
        ):
            continue
        await validator(db, check)


async def _model_is_admitted(
    db: AsyncSession, user_id: UUID, model: str, available: Collection[str]
) -> bool:
    """Whether ``user_id``'s instance may name ``model`` as its primary model."""
    if model in available:
        return True
    for source in list(_extra_models_sources):
        if model in set(await source(db, user_id)):
            return True
    return False


async def run_instance_deleted_hooks(user_id: UUID, instance_id: UUID) -> None:
    """Run what was registered for a deleted instance; for any place that deletes one."""
    for hook in list(_instance_deleted_hooks):
        try:
            await hook(user_id, instance_id)
        except Exception as exc:
            logger.error(
                "Instance-deleted hook %r failed for instance %s: %s",
                hook,
                instance_id,
                exc,
                exc_info=True,
            )


class UserAgentInstanceService:
    """
    Service for managing user agent instances.

    WHY: Provides CRUD operations for user-created agent configurations,
    validates against available base agents and models, and manages
    the "default" flag (only one default per user).
    """

    def __init__(self, db: AsyncSession, llm_config_service: LLMConfigService):
        self.db = db
        self.llm_config_service = llm_config_service

    # =========================================================
    # Read Operations
    # =========================================================

    async def list_instances(
        self,
        user_id: UUID,
        active_only: bool = True,
    ) -> List[UserAgentInstance]:
        """List all agent instances for a user."""
        filters = [
            UserAgentInstance.user_id == user_id,
        ]
        if active_only:
            filters.append(UserAgentInstance.is_active == True)  # noqa: E712

        query = (
            select(UserAgentInstance)
            .options(*INSTANCE_LOAD_OPTIONS)
            .where(and_(*filters))
            .order_by(
                UserAgentInstance.is_default.desc(), UserAgentInstance.display_name
            )
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_instance(
        self,
        user_id: UUID,
        instance_id: UUID,
    ) -> Optional[UserAgentInstance]:
        """Get a specific agent instance by ID."""
        query = (
            select(UserAgentInstance)
            .options(*INSTANCE_LOAD_OPTIONS)
            .where(
                and_(
                    UserAgentInstance.id == instance_id,
                    UserAgentInstance.user_id == user_id,
                )
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_instance_by_name(
        self,
        user_id: UUID,
        instance_name: str,
    ) -> Optional[UserAgentInstance]:
        """Get a specific agent instance by name."""
        query = (
            select(UserAgentInstance)
            .options(*INSTANCE_LOAD_OPTIONS)
            .where(
                and_(
                    UserAgentInstance.instance_name == instance_name,
                    UserAgentInstance.user_id == user_id,
                )
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_default_instance(
        self,
        user_id: UUID,
    ) -> Optional[UserAgentInstance]:
        """Get the user's default agent instance."""
        query = (
            select(UserAgentInstance)
            .options(*INSTANCE_LOAD_OPTIONS)
            .where(
                and_(
                    UserAgentInstance.user_id == user_id,
                    UserAgentInstance.is_default == True,  # noqa: E712
                    UserAgentInstance.is_active == True,  # noqa: E712
                )
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    # =========================================================
    # Write Operations
    # =========================================================

    async def create_instance(
        self,
        user_id: UUID,
        instance_name: str,
        display_name: str,
        base_agent_name: str,
        description: Optional[str] = None,
        primary_model: Optional[str] = None,
        system_prompt_override: Optional[str] = None,
        system_prompt_append: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        skills: Optional[List[Dict[str, str]]] = None,
        is_default: bool = False,
        is_deepagent: bool = False,
        subagent_ids: Optional[List[UUID]] = None,
        actor_is_superuser: Optional[bool] = None,
    ) -> UserAgentInstance:
        """Create a new agent instance for a user.

        Validates that:
        - base_agent_name exists in resolved config (with admin overrides)
        - primary_model (if provided) exists in available models from resolved config
        - instance_name is unique for this user
        - subagent_ids (if provided) exist and belong to the user
        - an MCP profile in config_overrides is one the instance may use

        ``actor_is_superuser`` says whether whoever acts is a superuser; None
        counts as not one. If is_default=True, clears any existing default
        for the user.
        """
        validate_agent_prompt_lengths(
            system_prompt_override=system_prompt_override,
            system_prompt_append=system_prompt_append,
        )

        # Get resolved config for validation
        resolved = await self.llm_config_service.get_resolved_config(user_id)

        # Validate base agent exists
        if base_agent_name not in resolved.agents:
            available = list(resolved.agents.keys())
            raise ValueError(
                f"Base agent '{base_agent_name}' not found. "
                f"Available agents: {available}"
            )

        # Reject internal-only agents (Agent Builder stage agents, etc.).
        # WHY: ``user_selectable=False`` means the agent is reserved for system
        # orchestration and must never back a ``UserAgentInstance`` even though
        # it appears in the resolved config registry.
        base_agent_config = resolved.agents[base_agent_name]
        if not getattr(base_agent_config, "user_selectable", True):
            raise ValueError(
                f"Base agent '{base_agent_name}' is not available for user " "instances"
            )

        # Validate primary model if provided
        if primary_model and not await _model_is_admitted(
            self.db, user_id, primary_model, resolved.available_models
        ):
            available = resolved.available_models
            raise ValueError(
                f"Model '{primary_model}' not found. " f"Available models: {available}"
            )

        config_overrides = self._normalize_and_validate_config_overrides(
            config_overrides,
            resolved.available_models,
        )
        await _run_overrides_validators(self.db, user_id, config_overrides)
        await _run_instance_checks(
            self.db,
            InstanceCheck(
                user_id=user_id,
                instance_id=None,
                base_agent_name=base_agent_name,
                config_overrides=dict(config_overrides or {}),
                previous_base_agent_name=None,
                previous_overrides=None,
                actor_is_superuser=actor_is_superuser,
                primary_model=primary_model,
                is_deepagent=bool(is_deepagent),
                skills=tuple(skills) if skills is not None else None,
                subagent_ids=(
                    tuple(subagent_ids) if is_deepagent and subagent_ids else ()
                ),
            ),
        )

        # Check uniqueness
        existing = await self.get_instance_by_name(user_id, instance_name)
        if existing:
            raise ValueError(
                f"Agent instance '{instance_name}' already exists for this user"
            )

        # Validate subagents
        subagents = []
        if is_deepagent and subagent_ids:
            query = select(UserAgentInstance).where(
                and_(
                    UserAgentInstance.id.in_(subagent_ids),
                    UserAgentInstance.user_id == user_id,
                )
            )
            result = await self.db.execute(query)
            subagents = list(result.scalars().all())
            if len(subagents) != len(subagent_ids):
                raise ValueError(
                    "One or more subagents not found or do not belong to the user"
                )

        # Clear existing default if setting new one
        if is_default:
            await self._clear_user_defaults(user_id)

        instance = UserAgentInstance(
            user_id=user_id,
            instance_name=instance_name,
            display_name=display_name,
            base_agent_name=base_agent_name,
            description=description,
            primary_model=primary_model,
            system_prompt_override=system_prompt_override,
            system_prompt_append=system_prompt_append,
            config_overrides=config_overrides,
            skills=skills,
            is_default=is_default,
            is_deepagent=is_deepagent,
            is_active=True,
            subagents=subagents,
        )

        self.db.add(instance)
        await self.db.commit()

        # Refresh with eager loading of subagents to avoid MissingGreenlet error
        query = (
            select(UserAgentInstance)
            .options(*INSTANCE_LOAD_OPTIONS)
            .where(UserAgentInstance.id == instance.id)
        )
        result = await self.db.execute(query)
        instance = result.scalar_one()

        logger.info(
            f"Created agent instance '{instance_name}' (base={base_agent_name}) "
            f"for user {user_id}"
        )
        return instance

    async def update_instance(
        self,
        user_id: UUID,
        instance_id: UUID,
        *,
        actor_is_superuser: Optional[bool] = None,
        **updates: Any,
    ) -> Optional[UserAgentInstance]:
        """Update an existing agent instance.

        Validates model names and manages default flag. The instance checks
        run again when config_overrides or the base agent changes, with
        ``actor_is_superuser`` as in :meth:`create_instance`; a validator
        registered ``also_on`` other changes runs on those too.
        """
        validate_agent_prompt_lengths(
            system_prompt_override=updates.get("system_prompt_override"),
            system_prompt_append=updates.get("system_prompt_append"),
        )

        # We need to eager load subagents if we are going to update them
        query = (
            select(UserAgentInstance)
            .options(*INSTANCE_LOAD_OPTIONS)
            .where(
                and_(
                    UserAgentInstance.id == instance_id,
                    UserAgentInstance.user_id == user_id,
                )
            )
        )
        result = await self.db.execute(query)
        instance = result.scalar_one_or_none()

        if not instance:
            return None

        # Validate primary_model if being updated
        if "primary_model" in updates and updates["primary_model"] is not None:
            resolved = await self.llm_config_service.get_resolved_config(user_id)
            if not await _model_is_admitted(
                self.db, user_id, updates["primary_model"], resolved.available_models
            ):
                available = resolved.available_models
                raise ValueError(
                    f"Model '{updates['primary_model']}' not found. "
                    f"Available models: {available}"
                )

        # Validate base_agent_name if being updated (mirrors create_instance):
        # ensure the new base exists and is user-selectable. Without this guard
        # callers could swap a valid instance over to an internal-only agent
        # (e.g. an Agent Builder stage agent) via PATCH.
        if "base_agent_name" in updates and updates["base_agent_name"] is not None:
            resolved = await self.llm_config_service.get_resolved_config(user_id)
            new_base = updates["base_agent_name"]
            if new_base not in resolved.agents:
                available = list(resolved.agents.keys())
                raise ValueError(
                    f"Base agent '{new_base}' not found. "
                    f"Available agents: {available}"
                )
            new_base_config = resolved.agents[new_base]
            if not getattr(new_base_config, "user_selectable", True):
                raise ValueError(
                    f"Base agent '{new_base}' is not available for user " "instances"
                )

        overrides_change = (
            "config_overrides" in updates and updates["config_overrides"] is not None
        )
        if overrides_change:
            resolved = await self.llm_config_service.get_resolved_config(user_id)
            updates["config_overrides"] = self._normalize_and_validate_config_overrides(
                updates["config_overrides"],
                resolved.available_models,
            )
            await _run_overrides_validators(
                self.db, user_id, updates["config_overrides"]
            )

        new_base = updates.get("base_agent_name") or instance.base_agent_name
        # What else changes, for the validators registered ``also_on`` it.
        # A value of None leaves a field as it is (see "Apply updates").
        new_model = updates.get("primary_model") or instance.primary_model
        new_deep = updates.get("is_deepagent")
        new_deep = instance.is_deepagent if new_deep is None else new_deep
        new_skills = updates.get("skills")
        new_skills = instance.skills if new_skills is None else new_skills
        current_subagent_ids = tuple(sub.id for sub in instance.subagents or [])
        new_subagent_ids = (
            tuple(updates["subagent_ids"])
            if updates.get("subagent_ids")
            else current_subagent_ids
        )
        other_changes = frozenset(
            name
            for name, changed in (
                ("primary_model", new_model != instance.primary_model),
                ("is_deepagent", bool(new_deep) != bool(instance.is_deepagent)),
                ("skills", new_skills != instance.skills),
                ("subagents", new_subagent_ids != current_subagent_ids),
            )
            if changed
        )
        main_change = overrides_change or new_base != instance.base_agent_name
        if main_change or other_changes:
            previous_overrides = dict(instance.config_overrides or {})
            await _run_instance_checks(
                self.db,
                InstanceCheck(
                    user_id=user_id,
                    instance_id=instance.id,
                    base_agent_name=new_base,
                    config_overrides=dict(
                        updates["config_overrides"]
                        if overrides_change
                        else previous_overrides
                    ),
                    previous_base_agent_name=instance.base_agent_name,
                    previous_overrides=previous_overrides,
                    actor_is_superuser=actor_is_superuser,
                    primary_model=new_model,
                    previous_primary_model=instance.primary_model,
                    is_deepagent=bool(new_deep),
                    skills=tuple(new_skills) if new_skills is not None else None,
                    subagent_ids=new_subagent_ids,
                ),
                only_on=None if main_change else other_changes,
            )

        # Handle default flag
        if updates.get("is_default") is True and not instance.is_default:
            await self._clear_user_defaults(user_id)

        # Handle subagents update
        if updates.get("subagent_ids"):
            subagent_ids = updates.pop("subagent_ids")
            if subagent_ids is not None:
                if not updates.get("is_deepagent", instance.is_deepagent):
                    raise ValueError("Cannot assign subagents to a non-deep agent")

                subagents_query = select(UserAgentInstance).where(
                    and_(
                        UserAgentInstance.id.in_(subagent_ids),
                        UserAgentInstance.user_id == user_id,
                    )
                )
                subagents_result = await self.db.execute(subagents_query)
                subagents = list(subagents_result.scalars().all())
                if len(subagents) != len(subagent_ids):
                    raise ValueError(
                        "One or more subagents not found or do not belong to the user"
                    )
                instance.subagents = subagents
            else:
                instance.subagents = []

        # Apply updates
        for key, value in updates.items():
            if value is not None and hasattr(instance, key):
                setattr(instance, key, value)

        await self.db.commit()

        # Refresh with eager loading of subagents to avoid MissingGreenlet error
        query = (
            select(UserAgentInstance)
            .options(*INSTANCE_LOAD_OPTIONS)
            .where(UserAgentInstance.id == instance.id)
        )
        result = await self.db.execute(query)
        instance = result.scalar_one()

        logger.info(
            f"Updated agent instance '{instance.instance_name}' "
            f"(id={instance_id}) for user {user_id}"
        )
        return instance

    async def delete_instance(
        self,
        user_id: UUID,
        instance_id: UUID,
    ) -> bool:
        """Hard-delete an agent instance."""
        instance = await self.get_instance(user_id, instance_id)
        if not instance:
            return False

        await self.db.delete(instance)
        await self.db.commit()

        logger.info(
            f"Deleted agent instance '{instance.instance_name}' "
            f"(id={instance_id}) for user {user_id}"
        )
        await run_instance_deleted_hooks(user_id, instance_id)
        return True

    # =========================================================
    # Template Operations (read-only from YAML)
    # =========================================================

    async def list_templates(self) -> List[Dict[str, Any]]:
        """List available agent templates from resolved config.

        Returns list of dicts with agent config details that can be
        used as base for creating instances, filtered by admin overrides.
        """
        resolved = await self.llm_config_service.get_resolved_config(None)
        templates = []
        for agent_name, agent_config in resolved.agents.items():
            # Skip internal-only agents so the user-facing template list never
            # exposes Agent Builder stage agents or other system agents.
            if not getattr(agent_config, "user_selectable", True):
                continue
            templates.append(
                {
                    "agent_name": agent_name,
                    "primary_model": agent_config.primary_model,
                    "fallback_models": agent_config.fallback_models,
                    "description": agent_config.description,
                    "allowed_tools": agent_config.allowed_tools,
                    "mcp_profile": agent_config.mcp_profile,
                    "local_tool_providers": agent_config.local_tool_providers or [],
                    "skills": agent_config.skills,
                    "subagents": agent_config.subagents,
                }
            )
        return templates

    async def list_available_models(self) -> List[str]:
        """List available model names from resolved config."""
        resolved = await self.llm_config_service.get_resolved_config(None)
        return resolved.available_models

    # =========================================================
    # Helpers
    # =========================================================

    async def _clear_user_defaults(self, user_id: UUID) -> None:
        """Clear the is_default flag for all user's instances."""
        stmt = (
            update(UserAgentInstance)
            .where(
                and_(
                    UserAgentInstance.user_id == user_id,
                    UserAgentInstance.is_default == True,  # noqa: E712
                )
            )
            .values(is_default=False)
        )
        await self.db.execute(stmt)

    @staticmethod
    def _normalize_and_validate_config_overrides(
        config_overrides: Optional[Dict[str, Any]],
        available_models: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Normalize fallback aliases and validate referenced fallback models."""
        if config_overrides is None:
            return None

        normalized = canonicalize_fallback_overrides(config_overrides)

        # Validate per-assistant memory gates (app-layer flags stored in the
        # config_overrides bag). Reject malformed values at the API boundary.
        if "memory_access_enabled" in normalized:
            if not isinstance(normalized["memory_access_enabled"], bool):
                raise ValueError(
                    "config_overrides.memory_access_enabled must be a boolean"
                )
        if "memory_scope" in normalized:
            if normalized["memory_scope"] not in ("shared", "dedicated"):
                raise ValueError(
                    "config_overrides.memory_scope must be 'shared' or 'dedicated'"
                )

        # Validate response_format: must be a non-empty JSON Schema dict.
        # WHY: keep DB overrides aligned with AgentConfig.validate_response_format
        # so invalid schemas are rejected at the API boundary, not at agent build.
        if "response_format" in normalized:
            rf = normalized["response_format"]
            if rf is not None:
                if not isinstance(rf, dict) or not rf:
                    raise ValueError(
                        "config_overrides.response_format must be a non-empty "
                        "JSON Schema dict or null"
                    )
                if not any(
                    key in rf for key in ("type", "$ref", "oneOf", "anyOf", "allOf")
                ):
                    raise ValueError(
                        "config_overrides.response_format must look like a JSON "
                        "Schema: expected at least one of 'type', '$ref', "
                        "'oneOf', 'anyOf', or 'allOf' at the top level"
                    )

        if "fallback" not in normalized:
            return normalized

        fallback = normalized.get("fallback")
        if fallback is None:
            fallback = []
            normalized["fallback"] = fallback
            normalized["fallback_models"] = fallback

        if not isinstance(fallback, list) or not all(
            isinstance(model, str) for model in fallback
        ):
            raise ValueError("config_overrides.fallback must be a list of model names")

        invalid = [model for model in fallback if model not in available_models]
        if invalid:
            raise ValueError(
                f"Fallback model(s) not found: {invalid}. "
                f"Available models: {available_models}"
            )

        return normalized


# =========================================================
# Factory / Dependency
# =========================================================


def get_user_agent_instance_service(
    db: AsyncSession, llm_config_service: LLMConfigService
) -> UserAgentInstanceService:
    """Factory for creating UserAgentInstanceService instances."""
    return UserAgentInstanceService(db, llm_config_service)

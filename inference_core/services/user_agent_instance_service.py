"""
User Agent Instance Service

CRUD service for managing user-created agent configuration instances.
Handles validation, default management, and template listing.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from inference_core.database.sql.models.user_agent_instance import UserAgentInstance
from inference_core.services.llm_config_service import LLMConfigService

logger = logging.getLogger(__name__)


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
            UserAgentInstance.is_deleted == False,  # noqa: E712
        ]
        if active_only:
            filters.append(UserAgentInstance.is_active == True)  # noqa: E712

        query = (
            select(UserAgentInstance)
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
        query = select(UserAgentInstance).where(
            and_(
                UserAgentInstance.id == instance_id,
                UserAgentInstance.user_id == user_id,
                UserAgentInstance.is_deleted == False,  # noqa: E712
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
        query = select(UserAgentInstance).where(
            and_(
                UserAgentInstance.instance_name == instance_name,
                UserAgentInstance.user_id == user_id,
                UserAgentInstance.is_deleted == False,  # noqa: E712
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_default_instance(
        self,
        user_id: UUID,
    ) -> Optional[UserAgentInstance]:
        """Get the user's default agent instance."""
        query = select(UserAgentInstance).where(
            and_(
                UserAgentInstance.user_id == user_id,
                UserAgentInstance.is_default == True,  # noqa: E712
                UserAgentInstance.is_active == True,  # noqa: E712
                UserAgentInstance.is_deleted == False,  # noqa: E712
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
        is_default: bool = False,
    ) -> UserAgentInstance:
        """Create a new agent instance for a user.

        Validates that:
        - base_agent_name exists in resolved config (with admin overrides)
        - primary_model (if provided) exists in available models from resolved config
        - instance_name is unique for this user

        If is_default=True, clears any existing default for the user.
        """
        # Get resolved config for validation
        resolved = await self.llm_config_service.get_resolved_config(user_id)

        # Validate base agent exists
        if base_agent_name not in resolved.agents:
            available = list(resolved.agents.keys())
            raise ValueError(
                f"Base agent '{base_agent_name}' not found. "
                f"Available agents: {available}"
            )

        # Validate primary model if provided
        if primary_model and primary_model not in resolved.available_models:
            available = resolved.available_models
            raise ValueError(
                f"Model '{primary_model}' not found. " f"Available models: {available}"
            )

        # Check uniqueness
        existing = await self.get_instance_by_name(user_id, instance_name)
        if existing:
            raise ValueError(
                f"Agent instance '{instance_name}' already exists for this user"
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
            is_default=is_default,
            is_active=True,
        )

        self.db.add(instance)
        await self.db.commit()
        await self.db.refresh(instance)

        logger.info(
            f"Created agent instance '{instance_name}' (base={base_agent_name}) "
            f"for user {user_id}"
        )
        return instance

    async def update_instance(
        self,
        user_id: UUID,
        instance_id: UUID,
        **updates: Any,
    ) -> Optional[UserAgentInstance]:
        """Update an existing agent instance.

        Validates model names and manages default flag.
        """
        instance = await self.get_instance(user_id, instance_id)
        if not instance:
            return None

        # Validate primary_model if being updated
        if "primary_model" in updates and updates["primary_model"] is not None:
            resolved = await self.llm_config_service.get_resolved_config(user_id)
            if updates["primary_model"] not in resolved.available_models:
                available = resolved.available_models
                raise ValueError(
                    f"Model '{updates['primary_model']}' not found. "
                    f"Available models: {available}"
                )

        # Handle default flag
        if updates.get("is_default") is True and not instance.is_default:
            await self._clear_user_defaults(user_id)

        # Apply updates
        for key, value in updates.items():
            if value is not None and hasattr(instance, key):
                setattr(instance, key, value)

        await self.db.commit()
        await self.db.refresh(instance)

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
        """Soft-delete an agent instance."""
        instance = await self.get_instance(user_id, instance_id)
        if not instance:
            return False

        instance.soft_delete()
        instance.is_active = False
        await self.db.commit()

        logger.info(
            f"Deleted agent instance '{instance.instance_name}' "
            f"(id={instance_id}) for user {user_id}"
        )
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
            templates.append(
                {
                    "agent_name": agent_name,
                    "primary_model": agent_config.primary_model,
                    "fallback_models": agent_config.fallback_models,
                    "description": agent_config.description,
                    "allowed_tools": agent_config.allowed_tools,
                    "mcp_profile": agent_config.mcp_profile,
                    "local_tool_providers": [],  # Not in resolved, keep empty or from base if needed
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
                    UserAgentInstance.is_deleted == False,  # noqa: E712
                )
            )
            .values(is_default=False)
        )
        await self.db.execute(stmt)


# =========================================================
# Factory / Dependency
# =========================================================


def get_user_agent_instance_service(
    db: AsyncSession, llm_config_service: LLMConfigService
) -> UserAgentInstanceService:
    """Factory for creating UserAgentInstanceService instances."""
    return UserAgentInstanceService(db, llm_config_service)

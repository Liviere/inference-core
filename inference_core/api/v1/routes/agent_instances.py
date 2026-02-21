"""
User Agent Instance API Router

Endpoints for managing user-created agent configuration instances.
Users can create, list, update, and delete personalized agent configurations
based on available base agent templates.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from inference_core.core.dependecies import (
    get_current_active_user,
    get_db,
    get_llm_config_service,
)
from inference_core.schemas.user_agent_instance import (
    AgentInstanceCreate,
    AgentInstanceListResponse,
    AgentInstanceResponse,
    AgentInstanceUpdate,
    AgentTemplateListResponse,
    AgentTemplateResponse,
)
from inference_core.services.llm_config_service import LLMConfigService
from inference_core.services.user_agent_instance_service import (
    get_user_agent_instance_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent-instances", tags=["Agent Instances"])


# =========================================================
# Template Endpoints (read-only, from YAML config)
# =========================================================


@router.get(
    "/templates",
    response_model=AgentTemplateListResponse,
    summary="List available agent templates",
)
async def list_agent_templates(
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    llm_config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    List base agent templates available for creating instances.

    Returns agent configurations from YAML config along with
    available models that can be selected.
    """
    service = get_user_agent_instance_service(db, llm_config_service)

    templates = await service.list_templates()
    available_models = await service.list_available_models()

    return AgentTemplateListResponse(
        templates=[AgentTemplateResponse(**t) for t in templates],
        available_models=available_models,
    )


# =========================================================
# Instance CRUD Endpoints
# =========================================================


@router.get(
    "",
    response_model=AgentInstanceListResponse,
    summary="List user agent instances",
)
async def list_agent_instances(
    include_inactive: bool = False,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    llm_config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    List all agent instances for the current user.

    Returns active instances by default. Set include_inactive=True to see all.
    """
    user_id = UUID(current_user["id"])
    service = get_user_agent_instance_service(db, llm_config_service)

    instances = await service.list_instances(
        user_id=user_id,
        active_only=not include_inactive,
    )

    return AgentInstanceListResponse(
        instances=[AgentInstanceResponse.model_validate(i) for i in instances],
        total=len(instances),
    )


@router.post(
    "",
    response_model=AgentInstanceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create agent instance",
)
async def create_agent_instance(
    data: AgentInstanceCreate,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    llm_config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    Create a new personalized agent instance.

    The instance is based on a base agent template (from YAML config)
    and can override model, system prompt, tools, and other parameters.
    """
    user_id = UUID(current_user["id"])
    service = get_user_agent_instance_service(db, llm_config_service)

    try:
        instance = await service.create_instance(
            user_id=user_id,
            instance_name=data.instance_name,
            display_name=data.display_name,
            base_agent_name=data.base_agent_name,
            description=data.description,
            primary_model=data.primary_model,
            system_prompt_override=data.system_prompt_override,
            system_prompt_append=data.system_prompt_append,
            config_overrides=data.config_overrides,
            is_default=data.is_default,
            is_deepagent=data.is_deepagent,
            subagent_ids=data.subagent_ids,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return AgentInstanceResponse.model_validate(instance)


@router.get(
    "/{instance_id}",
    response_model=AgentInstanceResponse,
    summary="Get agent instance",
)
async def get_agent_instance(
    instance_id: UUID,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    llm_config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """Get details of a specific agent instance."""
    user_id = UUID(current_user["id"])
    service = get_user_agent_instance_service(db, llm_config_service)

    instance = await service.get_instance(user_id, instance_id)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent instance not found",
        )

    return AgentInstanceResponse.model_validate(instance)


@router.patch(
    "/{instance_id}",
    response_model=AgentInstanceResponse,
    summary="Update agent instance",
)
async def update_agent_instance(
    instance_id: UUID,
    data: AgentInstanceUpdate,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    llm_config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """Update an existing agent instance."""
    user_id = UUID(current_user["id"])
    service = get_user_agent_instance_service(db, llm_config_service)

    updates = data.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update",
        )

    try:
        instance = await service.update_instance(
            user_id=user_id,
            instance_id=instance_id,
            **updates,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent instance not found",
        )

    return AgentInstanceResponse.model_validate(instance)


@router.delete(
    "/{instance_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete agent instance",
)
async def delete_agent_instance(
    instance_id: UUID,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    llm_config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """Delete (soft-delete) an agent instance."""
    user_id = UUID(current_user["id"])
    service = get_user_agent_instance_service(db, llm_config_service)

    deleted = await service.delete_instance(user_id, instance_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent instance not found",
        )

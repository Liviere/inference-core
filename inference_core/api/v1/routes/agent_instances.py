"""
User Agent Instance API Router

Endpoints for managing user-created agent configuration instances.
Users can create, list, update, and delete personalized agent configurations
based on available base agent templates.
"""

import logging
import uuid as _uuid
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from inference_core.core.config import get_settings
from inference_core.core.dependecies import (
    get_current_active_user,
    get_db,
    get_llm_config_service,
)
from inference_core.schemas.user_agent_instance import (
    AgentInstanceCreate,
    AgentInstanceListResponse,
    AgentInstanceResponse,
    AgentInstanceRunRequest,
    AgentInstanceRunResponse,
    AgentInstanceUpdate,
    AgentTemplateListResponse,
    AgentTemplateResponse,
    RunBundleConfig,
    RunBundleResponse,
)
from inference_core.services.agents_service import AgentService, DeepAgentService
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
            skills=data.skills,
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


# =========================================================
# Run Endpoint
# =========================================================


@router.post(
    "/{instance_id}/run",
    response_model=AgentInstanceRunResponse,
    summary="Run agent instance",
)
async def run_agent_instance(
    instance_id: UUID,
    data: AgentInstanceRunRequest,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    llm_config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """
    Run an agent instance synchronously and return the result.

    For deep agents (is_deepagent=True or has subagents), uses DeepAgentService
    with from_user_instance() which applies all DB overrides and resolves subagents.
    For regular agents, uses AgentService with DB config overrides applied.
    """
    user_id = UUID(current_user["id"])
    service = get_user_agent_instance_service(db, llm_config_service)

    instance = await service.get_instance(user_id, instance_id)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent instance not found",
        )
    if not instance.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent instance is not active",
        )

    try:
        # Resolve config with admin overrides + user preferences applied.
        # Instance-specific DB overrides are then layered on top via
        # build_config_for_instance(), giving the full resolution chain:
        # YAML → admin overrides → user preferences → instance DB overrides.
        base_config = await llm_config_service.get_config_with_overrides(user_id)

        if instance.is_deepagent or instance.subagents:
            agent_svc = await DeepAgentService.from_user_instance(
                instance,
                user_id=user_id,
                base_config=base_config,
            )
        else:
            agent_svc = AgentService.from_user_instance(
                instance,
                user_id=user_id,
                base_config=base_config,
            )

        with agent_svc:
            await agent_svc.create_agent(system_prompt=data.system_prompt)
            response = await agent_svc.arun_agent_steps(data.user_input)

    except Exception as e:
        logger.exception("Error running agent instance %s: %s", instance_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {e}",
        )

    cost_metrics = None
    if response.cost_metrics:
        cost_metrics = response.cost_metrics.model_dump()

    return AgentInstanceRunResponse(
        result=response.result,
        steps=response.steps,
        model_name=response.metadata.model_name,
        instance_id=instance.id,
        instance_name=instance.instance_name,
        cost_metrics=cost_metrics,
    )


# =========================================================
# Run Bundle (frontend handshake for direct Agent Server use)
# =========================================================


def _extract_bearer_from_request(request: Request) -> Optional[str]:
    """Pull the bearer token off the inbound request, if any.

    WHY: We forward the same JWT to the frontend so it can authenticate
    the subsequent ``useStream`` connection to the Agent Server.  No new
    token is minted — token TTL stays under user/admin control.
    """
    auth_header = request.headers.get("authorization") or request.headers.get(
        "Authorization"
    )
    if not auth_header:
        return None
    parts = auth_header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


@router.get(
    "/{instance_id}/run-bundle",
    response_model=RunBundleResponse,
    summary="Get run bundle for a frontend Agent Server connection",
)
async def get_agent_instance_run_bundle(
    instance_id: UUID,
    request: Request,
    session_id: Optional[str] = None,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    llm_config_service: LLMConfigService = Depends(get_llm_config_service),
):
    """Return the handshake payload for a direct frontend → Agent Server connection.

    The frontend (``@langchain/react`` ``useStream``) needs three things to
    open a streaming session:

      1. **assistantId** — which graph to invoke on the Agent Server.
      2. **agentServerUrl** — where the Agent Server lives.
      3. **config.configurable** — per-instance overrides (model, prompts,
         subagent_configs, memory toggles) that drive the middleware stack.

    Computing this bundle backend-side keeps the resolution logic
    (admin overrides → user preferences → DB instance overrides → reasoning
    flags → memory flags → subagent_configs) in one place and prevents the
    frontend from needing to know the YAML schema.

    A pre-existing ``session_id`` (resumable thread) can be passed via query
    string; otherwise the frontend is responsible for creating threads via
    the Agent Server's own ``/threads`` endpoint.
    """
    user_id = UUID(current_user["id"])
    service = get_user_agent_instance_service(db, llm_config_service)

    instance = await service.get_instance(user_id, instance_id)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent instance not found",
        )
    if not instance.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent instance is not active",
        )

    settings = get_settings()
    if not settings.agent_server_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Agent Server URL is not configured. "
                "Set AGENT_SERVER_URL in the backend environment."
            ),
        )

    # Reuse AgentService.from_user_instance to keep the override resolution
    # chain identical to the live runtime path.  We never call create_agent()
    # so no LLM clients, tools, or graphs are instantiated — only metadata.
    base_config = await llm_config_service.get_config_with_overrides(user_id)

    try:
        agent_svc = AgentService.from_user_instance(
            instance,
            user_id=user_id,
            session_id=session_id,
            request_id=str(_uuid.uuid4()),
            base_config=base_config,
        )
        configurable = agent_svc._build_remote_metadata()
    except Exception as exc:
        logger.exception(
            "Failed to build run bundle for instance %s: %s", instance_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build run bundle: {exc}",
        )

    return RunBundleResponse(
        instance_id=instance.id,
        instance_name=instance.instance_name,
        display_name=instance.display_name,
        base_agent_name=instance.base_agent_name,
        description=instance.description,
        assistant_id=instance.base_agent_name,
        agent_server_url=settings.agent_server_url,
        access_token=_extract_bearer_from_request(request),
        config=RunBundleConfig(configurable=configurable),
        is_remote=agent_svc._is_remote,
    )

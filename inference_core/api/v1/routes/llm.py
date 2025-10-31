"""
LLM Router

API endpoints for Large Language Model operations.
Provides endpoints for story generation, analysis, and other LLM-powered features.
"""

import logging
from typing import Annotated, Any, Dict, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, BaseModel, Field

from inference_core.core.dependecies import (
    get_llm_router_dependencies,
    get_optional_current_user,
)
from inference_core.schemas.tasks_responses import TaskResponse
from inference_core.services.llm_service import get_llm_service
from inference_core.services.task_service import TaskService, get_task_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/llm", tags=["LLM"], dependencies=get_llm_router_dependencies()
)


class BaseLLMRequest(BaseModel):
    """Base request model for LLM operations"""

    model_name: Optional[str] = Field(
        None, description="Optional model name to override default"
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Model temperature"
    )
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens")

    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Top-p sampling parameter"
    )

    frequency_penalty: Optional[float] = Field(
        default=None, ge=-2.0, le=2.0, description="Frequency penalty"
    )

    presence_penalty: Optional[float] = Field(
        default=None, ge=-2.0, le=2.0, description="Presence penalty"
    )

    request_timeout: Optional[int] = Field(
        default=None, ge=1, description="Request timeout in seconds"
    )

    # GPT-5+ experimental params (temperature/top_p deprecated there)
    reasoning_effort: Optional[str] = Field(
        default=None,
        description="Reasoning effort level for advanced models (e.g., low|medium|high)",
    )
    verbosity: Optional[str] = Field(
        default=None,
        description="Verbosity of the response (e.g., low|medium|high)",
    )
    as_user_id: Optional[UUID] = Field(
        default=None,
        description=(
            "Optional user ID to impersonate (requires superuser). If provided and caller is superuser, usage will be logged under this user."
        ),
    )


class CompletionRequest(BaseLLMRequest):
    """Request model for completion endpoint"""

    # Accept both 'prompt' (preferred) and legacy 'question' from clients
    prompt: Annotated[
        str,
        Field(
            description="Text prompt to generate from",
            validation_alias=AliasChoices("prompt", "question"),
            serialization_alias="prompt",
        ),
    ]
    input_vars: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary of template variables replacing single 'prompt'",
    )


class ChatRequest(BaseLLMRequest):
    """Request model for chat endpoint"""

    session_id: Optional[str] = Field(
        default=None,
        description="Chat session identifier. If omitted, a new session will be created.",
    )
    user_input: str = Field(..., description="User message to the assistant")
    input_vars: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary of template variables. user_input remains the value used for history.",
    )


class ModelsListResponse(BaseModel):
    """Response model for available models list"""

    models: Dict[str, bool]
    success: bool = True


class LLMStatsResponse(BaseModel):
    """Response model for LLM usage statistics"""

    stats: Dict[str, Any]
    success: bool = True


# Dependency to get LLM service
def get_llm_service_dependency():
    """Dependency to get LLM service instance"""
    return get_llm_service()


###################################
#            Endpoints            #
###################################


@router.post("/completion", response_model=TaskResponse)
async def completion(
    request: CompletionRequest,
    task_service: TaskService = Depends(get_task_service),
    current_user: Optional[dict] = Depends(get_optional_current_user),
) -> TaskResponse:
    """
    Generate a completion-style answer for a given question using the specified model.

    Args:
    request: CompletionRequest containing the question and optional parameters
        task_service: TaskService instance for managing async tasks

    Returns:
        TaskResponse with task ID and status
    """
    try:
        payload = request.model_dump(exclude_none=True)
        effective_user_id: Optional[str] = None
        if current_user:
            effective_user_id = current_user.get("id")
            if request.as_user_id and current_user.get("is_superuser"):
                effective_user_id = str(request.as_user_id)
        if effective_user_id:
            payload["user_id"] = effective_user_id
        task_id = await task_service.completion_submit_async(**payload)

        return TaskResponse(
            task_id=task_id,
            status="PENDING",
            message="Completion task submitted successfully",
        )

    except Exception as e:
        logger.error(f"Completion task submission failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Completion task submission failed: {str(e)}"
        )


@router.post("/chat", response_model=TaskResponse)
async def chat(
    request: ChatRequest,
    task_service: TaskService = Depends(get_task_service),
    current_user: Optional[dict] = Depends(get_optional_current_user),
) -> TaskResponse:
    """Submit a chat turn as a Celery task.

    If session_id is not provided, a new session will be created on the worker.
    """
    try:
        # Build kwargs for Celery task
        kwargs: Dict[str, Any] = {**request.model_dump(exclude_none=True)}
        if current_user:
            effective_user_id = current_user.get("id")
            if request.as_user_id and current_user.get("is_superuser"):
                effective_user_id = str(request.as_user_id)
            kwargs["user_id"] = effective_user_id
        task_id = await task_service.chat_submit_async(**kwargs)

        return TaskResponse(
            task_id=task_id,
            status="PENDING",
            message="Chat task submitted successfully",
        )
    except Exception as e:
        logger.error(f"Chat task submission failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat task submission failed: {str(e)}",
        )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    llm_service=Depends(get_llm_service_dependency),
    current_user: Optional[dict] = Depends(get_optional_current_user),
):
    """
    Stream a chat turn using Server-Sent Events.

    Provides real-time token-by-token streaming for chat responses.
    If session_id is not provided, a new session will be created.

    Returns:
        StreamingResponse with Server-Sent Events (text/event-stream)

    Event Format:
        data: {"event":"start","model":"gpt-4o-mini","session_id":"abc123"}
        data: {"event":"token","content":"Hello"}
        data: {"event":"token","content":" world"}
        data: {"event":"usage","usage":{"input_tokens":123,"output_tokens":45,"total_tokens":168}}
        data: {"event":"end"}
    """
    try:
        # Build kwargs for streaming
        stream_kwargs = {}
        for field, value in request.model_dump(exclude_none=True).items():
            if field not in ["session_id", "user_input", "model_name"]:
                stream_kwargs[field] = value

        effective_user_id: Optional[str] = None
        if current_user:
            effective_user_id = current_user.get("id")
            if request.as_user_id and current_user.get("is_superuser"):
                effective_user_id = str(request.as_user_id)

        request_id = http_request.headers.get("X-Request-ID") or str(uuid4())

        async_generator = llm_service.stream_chat(
            session_id=request.session_id,
            user_input=request.user_input,
            model_name=request.model_name,
            request=http_request,
            user_id=effective_user_id,
            request_id=request_id,
            **stream_kwargs,
        )

        # Source: FastAPI docs – StreamingResponse, 2025-08 snapshot
        return StreamingResponse(
            async_generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    except Exception as e:
        logger.error(f"Chat streaming failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat streaming failed: {str(e)}",
        )


@router.post("/completion/stream")
async def completion_stream(
    request: CompletionRequest,
    http_request: Request,
    llm_service=Depends(get_llm_service_dependency),
    current_user: Optional[dict] = Depends(get_optional_current_user),
):
    """
    Stream a completion using Server-Sent Events.

    Provides real-time token-by-token streaming for completion responses.

    Returns:
        StreamingResponse with Server-Sent Events (text/event-stream)

    Event Format:
        data: {"event":"start","model":"gpt-4o-mini"}
        data: {"event":"token","content":"Hello"}
        data: {"event":"token","content":" world"}
        data: {"event":"usage","usage":{"input_tokens":123,"output_tokens":45,"total_tokens":168}}
        data: {"event":"end"}
    """
    try:
        # Build kwargs for streaming
        stream_kwargs = {}
        for field, value in request.model_dump(exclude_none=True).items():
            if field not in ["prompt", "model_name"]:
                stream_kwargs[field] = value

        effective_user_id: Optional[str] = None
        if current_user:
            effective_user_id = current_user.get("id")
            if request.as_user_id and current_user.get("is_superuser"):
                effective_user_id = str(request.as_user_id)

        request_id = http_request.headers.get("X-Request-ID") or str(uuid4())

        async_generator = llm_service.stream_completion(
            prompt=request.prompt,
            model_name=request.model_name,
            request=http_request,
            user_id=effective_user_id,
            request_id=request_id,
            **stream_kwargs,
        )

        # Source: FastAPI docs – StreamingResponse, 2025-08 snapshot
        return StreamingResponse(
            async_generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    except Exception as e:
        logger.error(f"Completion streaming failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Completion streaming failed: {str(e)}",
        )


@router.get("/models", response_model=ModelsListResponse)
async def list_available_models(llm_service=Depends(get_llm_service_dependency)):
    """
    Get list of available LLM models.

    Returns all configured models and their availability status.
    """
    try:
        models = llm_service.get_available_models()
        return ModelsListResponse(models=models)

    except Exception as e:
        logger.error(f"Failed to get available models: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get available models: {str(e)}"
        )


@router.get("/stats", response_model=LLMStatsResponse)
async def get_llm_statistics(llm_service=Depends(get_llm_service_dependency)):
    """
    Get LLM usage statistics.

    Returns usage metrics including request counts, tokens used, cost information, and error rates.
    """
    try:
        stats = await llm_service.get_usage_stats()
        return LLMStatsResponse(stats=stats)

    except Exception as e:
        logger.error(f"Failed to get LLM statistics: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get LLM statistics: {str(e)}"
        )


@router.get("/health")
async def health_check(llm_service=Depends(get_llm_service_dependency)):
    """
    Health check endpoint for LLM services.

    Verifies that LLM services are operational and models are accessible.
    """
    try:
        models = llm_service.get_available_models()
        available_count = sum(1 for available in models.values() if available)
        total_count = len(models)

        return {
            "status": "healthy" if available_count > 0 else "degraded",
            "available_models": available_count,
            "total_models": total_count,
            "models": models,
        }

    except Exception as e:
        logger.error(f"LLM health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "available_models": 0,
            "total_models": 0,
        }


# Debug endpoint for parameter policies (only available in DEBUG mode)
from inference_core.core.config import Settings, get_settings


@router.get("/param-policy/{provider}")
async def get_param_policy(
    provider: str,
    model: Optional[str] = None,
    settings: Settings = Depends(get_settings),
):
    """
    Get parameter policy for a specific LLM provider.

    Only available when DEBUG=True in settings.
    Useful for inspecting parameter normalization rules.
    """
    if not settings.debug:
        raise HTTPException(
            status_code=404, detail="Debug endpoints are only available in DEBUG mode"
        )

    try:
        from inference_core.llm.config import ModelProvider
        from inference_core.llm.param_policy import (
            get_model_policy,
            get_provider_policy,
            get_supported_providers,
        )

        # Convert string to ModelProvider enum
        try:
            provider_enum = ModelProvider(provider)
        except ValueError:
            supported = [p.value for p in get_supported_providers()]
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported provider '{provider}'. Supported: {supported}",
            )

        provider_policy = get_provider_policy(provider_enum)

        response = {
            "provider": provider,
            "policy": {
                "allowed_parameters": sorted(list(provider_policy.allowed)),
                "parameter_mappings": dict(provider_policy.renamed),
                "dropped_parameters": sorted(list(provider_policy.dropped)),
                "passthrough_prefixes": sorted(
                    list(provider_policy.passthrough_prefixes)
                ),
            },
            "description": f"Parameter normalization policy for {provider} provider",
        }

        if model:
            # Determine provider from model config if mismatch not handled here (simple attempt)
            try:
                from inference_core.llm.config import llm_config

                model_cfg = llm_config.get_model_config(model)
                if model_cfg and model_cfg.provider != provider_enum:
                    response["model_provider_mismatch"] = {
                        "model": model,
                        "model_provider": model_cfg.provider.value,
                        "requested_provider": provider_enum.value,
                        "note": "Using model's actual provider for effective policy merge",
                    }
                    effective_provider = model_cfg.provider
                else:
                    effective_provider = provider_enum
            except Exception:
                effective_provider = provider_enum

            model_policy = get_model_policy(model, effective_provider)
            response["model"] = {
                "name": model,
                "effective_policy": {
                    "allowed_parameters": sorted(list(model_policy.allowed)),
                    "parameter_mappings": dict(model_policy.renamed),
                    "dropped_parameters": sorted(list(model_policy.dropped)),
                    "passthrough_prefixes": sorted(
                        list(model_policy.passthrough_prefixes)
                    ),
                },
            }

        return response

    except Exception as e:
        logger.error(f"Failed to get parameter policy: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get parameter policy: {str(e)}"
        )

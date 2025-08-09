"""
LLM Router

API endpoints for Large Language Model operations.
Provides endpoints for story generation, analysis, and other LLM-powered features.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.celery.task_service import TaskService, get_task_service
from app.llm.llm_service import LLMResponse, get_llm_service
from app.schemas.tasks_responses import TaskResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM"])


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


class ExplainRequest(BaseLLMRequest):
    """Request model for explanation endpoint"""

    question: str = Field(..., description="The question to explain")


class ConversationRequest(BaseLLMRequest):
    """Request model for conversation endpoint"""

    session_id: Optional[str] = Field(
        default=None,
        description="Conversation session identifier. If omitted, a new session will be created.",
    )
    user_input: str = Field(..., description="User message to the assistant")


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


@router.post("/explain", response_model=TaskResponse)
async def explain(
    request: ExplainRequest, task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    """
    Generate an explanation for a given question using the specified model.

    Args:
        request: ExplainRequest containing the question and optional parameters
        task_service: TaskService instance for managing async tasks

    Returns:
        TaskResponse with task ID and status
    """
    try:
        task_id = task_service.explain_async(**request.model_dump(exclude_none=True))

        return TaskResponse(
            task_id=task_id,
            status="PENDING",
            message="Explanation task submitted successfully",
        )

    except Exception as e:
        logger.error(f"Explenation task submission failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Explenation task submission failed: {str(e)}"
        )


@router.post("/conversation", response_model=TaskResponse)
async def conversation(
    request: ConversationRequest, task_service: TaskService = Depends(get_task_service)
) -> TaskResponse:
    """Submit a conversation turn as a Celery task.

    If session_id is not provided, a new session will be created on the worker.
    """
    try:
        # Build kwargs for Celery task
        kwargs: Dict[str, Any] = {
            **request.model_dump(exclude_none=True),
        }
        task_id = task_service.conversation_async(**kwargs)

        return TaskResponse(
            task_id=task_id,
            status="PENDING",
            message="Conversation task submitted successfully",
        )
    except Exception as e:
        logger.error(f"Conversation task submission failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Conversation task submission failed: {str(e)}",
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

    Returns usage metrics including request counts, tokens used, and error rates.
    """
    try:
        stats = llm_service.get_usage_stats()
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

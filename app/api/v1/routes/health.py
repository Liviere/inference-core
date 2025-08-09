"""
Health Check and System Status Endpoints

FastAPI endpoints for monitoring application health,
system status, and diagnostic information.
"""

from datetime import UTC, datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.dependecies import get_db
from app.database.sql.connection import db_manager
from app.services.task_service import TaskService, get_task_service

router = APIRouter(prefix="/health", tags=["Health Check"])


###################################
#             Schemas            #
###################################
class StatusResponse(BaseModel):
    """Generic status response schema"""

    status: str = Field(..., description="Current status")
    details: Optional[Dict[str, Any]] = Field(None, description="Status details")
    last_updated: str = Field(..., description="Last status update timestamp")


class HealthCheckResponse(BaseModel):
    """Health check response schema"""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(..., description="Check timestamp")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component statuses")


###################################
#             Routes             #
###################################


@router.get("/", response_model=HealthCheckResponse)
async def health_check(
    settings=Depends(get_settings),
    db_session=Depends(get_db),
    task_service: TaskService = Depends(get_task_service),
) -> HealthCheckResponse:
    """
    Overall application health check

    Args:
        settings: Application settings dependency

    Returns:
        Health status of all components
    """
    timestamp = str(datetime.now(UTC).isoformat())

    # Check database health
    db_healthy = await db_manager.health_check(db_session)

    # Placeholder for service health checks (overall based on critical deps only)
    overall_status = "healthy" if all([db_healthy]) else "unhealthy"

    # Tasks/Celery health (non-critical for overall status here)
    try:
        worker_stats = task_service.get_worker_stats()
        ping_responses = worker_stats.get("ping") or {}
        active_workers = (
            len([w for w in ping_responses.values() if w.get("ok") == "pong"])
            if isinstance(ping_responses, dict)
            else 0
        )
        tasks_status = (
            "healthy"
            if active_workers > 0
            else ("degraded" if ping_responses else "unavailable")
        )
        tasks_component = {
            "status": tasks_status,
            "active_workers": active_workers,
            "celery_available": bool(ping_responses),
            "checked_at": timestamp,
        }
    except Exception as e:
        tasks_component = {
            "status": "unavailable",
            "active_workers": 0,
            "celery_available": False,
            "message": f"Task service error: {str(e)}",
            "checked_at": timestamp,
        }

    components = {
        "database": {
            "status": "healthy" if db_healthy else "unhealthy",
            "checked_at": timestamp,
        },
        "application": {
            "status": "healthy",
            "version": settings.app_version,
            "environment": settings.environment,
            "checked_at": timestamp,
        },
        "tasks": tasks_component,
    }

    return HealthCheckResponse(
        status=overall_status,
        version=settings.app_version,
        timestamp=timestamp,
        components=components,
    )


@router.get("/database", response_model=StatusResponse)
async def database_health(db_session: AsyncSession = Depends(get_db)) -> StatusResponse:
    """
    Database-specific health check

    Args:
        db: Database session

    Returns:
        Database health status
    """
    db_info = await db_manager.get_database_info(db_session)

    return StatusResponse(
        status=db_info["status"],
        details=db_info,
        last_updated=str(datetime.now(UTC).isoformat()),
    )


@router.get("/ping")
async def ping() -> Dict[str, Any]:
    """
    Simple ping endpoint for basic health checking

    Returns:
        Simple pong response
    """
    return {
        "message": "pong",
        "timestamp": str(datetime.now(UTC).isoformat()),
        "status": "ok",
    }

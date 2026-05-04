"""
Task Management Router

API endpoints for managing asynchronous tasks.
Provides endpoints for task status checking, cancellation, and monitoring.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from inference_core.schemas.tasks_responses import (
    ActiveTasksResponse,
    TaskCancelResponse,
    TaskStatusResponse,
    WorkerStatsResponse,
)
from inference_core.services.task_service import TaskService, get_task_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["Task Management"])

TASK_INSPECT_TIMEOUT_SECONDS = 1.0
TASK_HEALTH_CACHE_TTL_SECONDS = 5.0
TASK_ACTIVE_CACHE_TTL_SECONDS = 2.0
TASK_FAILURE_CACHE_TTL_SECONDS = 30.0


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str, task_service: TaskService = Depends(get_task_service)
):
    """
    Get the status of a specific task.

    Returns detailed information about task state, result, and any errors.
    """
    try:
        status = await task_service.get_task_status_async(task_id)

        return TaskStatusResponse(
            task_id=task_id,
            status=status["status"],
            result=status["result"],
            info=status["info"],
            traceback=status["traceback"],
            successful=status["successful"],
            failed=status["failed"],
        )

    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get task status: {str(e)}"
        )


@router.get("/{task_id}/result")
async def get_task_result(
    task_id: str,
    timeout: Optional[float] = Query(default=None, description="Timeout in seconds"),
    task_service: TaskService = Depends(get_task_service),
):
    """
    Get the result of a completed task.

    Waits for task completion if still running (up to timeout).
    """
    try:
        result = await task_service.get_task_result_async(task_id, timeout=timeout)

        return {"task_id": task_id, "result": result, "success": True}

    except Exception as e:
        logger.error(f"Failed to get task result for {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get task result: {str(e)}"
        )


@router.delete("/{task_id}", response_model=TaskCancelResponse)
async def cancel_task(
    task_id: str, task_service: TaskService = Depends(get_task_service)
):
    """
    Cancel a pending or running task.

    Attempts to gracefully terminate the task.
    """
    try:
        cancelled = await task_service.cancel_task_async(task_id)

        return TaskCancelResponse(
            task_id=task_id,
            cancelled=cancelled,
            message=(
                "Task cancelled successfully" if cancelled else "Failed to cancel task"
            ),
        )

    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")


@router.get("/active", response_model=ActiveTasksResponse)
async def get_active_tasks(task_service: TaskService = Depends(get_task_service)):
    """
    Get information about currently active tasks.

    Returns lists of active, scheduled, and reserved tasks across all workers.
    """
    try:
        active_info = await task_service.get_active_tasks_async(
            timeout=TASK_INSPECT_TIMEOUT_SECONDS,
            cache_ttl=TASK_ACTIVE_CACHE_TTL_SECONDS,
            failure_cache_ttl=TASK_FAILURE_CACHE_TTL_SECONDS,
        )

        return ActiveTasksResponse(
            active=active_info["active"] or {},
            scheduled=active_info["scheduled"] or {},
            reserved=active_info["reserved"] or {},
        )

    except Exception as e:
        logger.error(f"Failed to get active tasks: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get active tasks: {str(e)}"
        )


@router.get("/workers/stats", response_model=WorkerStatsResponse)
async def get_worker_stats(task_service: TaskService = Depends(get_task_service)):
    """
    Get statistics about Celery workers.

    Returns performance metrics, registered tasks, and worker health information.
    """
    try:
        stats = await task_service.get_worker_stats_async(
            timeout=TASK_INSPECT_TIMEOUT_SECONDS,
            cache_ttl=TASK_HEALTH_CACHE_TTL_SECONDS,
            failure_cache_ttl=TASK_FAILURE_CACHE_TTL_SECONDS,
        )

        return WorkerStatsResponse(
            stats=stats["stats"] or {},
            ping=stats["ping"] or {},
            registered=stats["registered"] or {},
        )

    except Exception as e:
        logger.error(f"Failed to get worker stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get worker stats: {str(e)}"
        )


@router.get("/health")
async def health_check(task_service: TaskService = Depends(get_task_service)):
    """
    Check the health of the task system.

    Returns basic connectivity and worker availability information.
    """
    try:
        # Try to ping workers without collecting expensive worker metadata.
        ping_responses = await task_service.get_worker_ping_async(
            timeout=TASK_INSPECT_TIMEOUT_SECONDS,
            cache_ttl=TASK_HEALTH_CACHE_TTL_SECONDS,
            failure_cache_ttl=TASK_FAILURE_CACHE_TTL_SECONDS,
        )
        ping_responses = ping_responses or {}

        # Count active workers
        active_workers = (
            len(
                [
                    worker_response
                    for worker_response in ping_responses.values()
                    if isinstance(worker_response, dict)
                    and worker_response.get("ok") == "pong"
                ]
            )
            if isinstance(ping_responses, dict)
            else 0
        )

        return {
            "status": "healthy" if active_workers > 0 else "degraded",
            "active_workers": active_workers,
            "message": f"{active_workers} worker(s) available",
            "celery_available": bool(ping_responses),
        }

    except Exception as e:
        logger.error(f"Task health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "active_workers": 0,
            "message": f"Celery unavailable: {str(e)}",
            "celery_available": False,
        }

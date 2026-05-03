"""
Task Service for managing Celery tasks
"""

import time
from typing import Any, Dict, Optional

from celery.result import AsyncResult
from fastapi.concurrency import run_in_threadpool

from inference_core.celery.celery_main import celery_app


class TaskService:
    """Service for managing asynchronous tasks"""

    def __init__(self):
        self.celery_app = celery_app
        self._worker_stats_cache: tuple[float, Dict[str, Any]] | None = None

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task

        Args:
            task_id: Task ID to check

        Returns:
            Dictionary with task status information
        """
        task_result = AsyncResult(task_id, app=self.celery_app)

        return {
            "task_id": task_id,
            "status": task_result.status,
            "result": task_result.result if task_result.ready() else None,
            "info": task_result.info,
            "traceback": task_result.traceback,
            "successful": task_result.successful() if task_result.ready() else None,
            "failed": task_result.failed() if task_result.ready() else None,
        }

    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get the result of a completed task

        Args:
            task_id: Task ID to get result for
            timeout: Maximum time to wait for result

        Returns:
            Task result or raises exception if task failed
        """
        task_result = AsyncResult(task_id, app=self.celery_app)
        return task_result.get(timeout=timeout)

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task

        Args:
            task_id: Task ID to cancel

        Returns:
            True if task was cancelled, False otherwise
        """
        task_result = AsyncResult(task_id, app=self.celery_app)
        task_result.revoke(terminate=True)
        return True

    def get_active_tasks(self) -> Dict[str, Any]:
        """
        Get information about currently active tasks

        Returns:
            Dictionary with active tasks information
        """
        inspect = self.celery_app.control.inspect()
        return {
            "active": inspect.active(),
            "scheduled": inspect.scheduled(),
            "reserved": inspect.reserved(),
        }

    def get_worker_stats(
        self, timeout: Optional[float] = 1.0, cache_ttl: float = 0.0
    ) -> Dict[str, Any]:
        """
        Get statistics about Celery workers with bounded broker RPCs.

        WHY: Health endpoints can call Celery inspection frequently. A timeout
        and optional short cache prevent slow or missing workers from making the
        API process spend repeated blocking waits on the broker.

        Returns:
            Dictionary with worker statistics
        """
        now = time.monotonic()
        if cache_ttl > 0 and self._worker_stats_cache is not None:
            cached_at, cached_stats = self._worker_stats_cache
            if now - cached_at <= cache_ttl:
                return cached_stats

        inspect_kwargs = {}
        if timeout is not None:
            inspect_kwargs["timeout"] = timeout

        inspect = self.celery_app.control.inspect(**inspect_kwargs)
        worker_stats = {
            "stats": inspect.stats(),
            "ping": inspect.ping(),
            "registered": inspect.registered(),
        }
        if cache_ttl > 0:
            self._worker_stats_cache = (now, worker_stats)

        return worker_stats

    # -------------------------
    # Async wrappers (threadpool)
    # -------------------------

    async def get_task_status_async(self, task_id: str) -> Dict[str, Any]:
        """Async wrapper for get_task_status to avoid blocking the event loop."""
        return await run_in_threadpool(self.get_task_status, task_id)

    async def get_task_result_async(
        self, task_id: str, timeout: Optional[float] = None
    ) -> Any:
        """Async wrapper for get_task_result to avoid blocking the event loop."""
        return await run_in_threadpool(self.get_task_result, task_id, timeout)

    async def cancel_task_async(self, task_id: str) -> bool:
        """Async wrapper for cancel_task to avoid blocking the event loop."""
        return await run_in_threadpool(self.cancel_task, task_id)

    async def get_active_tasks_async(self) -> Dict[str, Any]:
        """Async wrapper for get_active_tasks to avoid blocking the event loop."""
        return await run_in_threadpool(self.get_active_tasks)

    async def get_worker_stats_async(
        self, timeout: Optional[float] = 1.0, cache_ttl: float = 0.0
    ) -> Dict[str, Any]:
        """Async wrapper for bounded worker inspection from async routes."""
        return await run_in_threadpool(
            self.get_worker_stats,
            timeout=timeout,
            cache_ttl=cache_ttl,
        )

    def completion_async(self, **kwargs) -> str:
        """Submit completion task"""
        task = self.celery_app.send_task("llm.completion", kwargs=kwargs)
        return task.id

    def chat_async(self, **kwargs) -> str:
        """Submit chat task (one turn)"""
        task = self.celery_app.send_task("llm.chat", kwargs=kwargs)
        return task.id

    async def completion_submit_async(self, **kwargs) -> str:
        """Async wrapper for completion_async to avoid blocking the event loop."""
        return await run_in_threadpool(self.completion_async, **kwargs)

    async def chat_submit_async(self, **kwargs) -> str:
        """Async wrapper for chat_async to avoid blocking the event loop."""
        return await run_in_threadpool(self.chat_async, **kwargs)


# Global task service instance
task_service = TaskService()


def get_task_service() -> TaskService:
    """Get the global task service instance"""
    return task_service

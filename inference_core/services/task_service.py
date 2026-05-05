"""
Task Service for managing Celery tasks
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, Optional

from celery.result import AsyncResult
from fastapi.concurrency import run_in_threadpool

from inference_core.celery.celery_main import celery_app


@dataclass(slots=True)
class _InspectCacheEntry:
    """Store Celery inspect outcomes so broker outages do not fan out.

    WHY: health checks can run frequently and concurrently. Caching both
    successful responses and short-lived failures prevents a broker outage from
    creating a new remote-control connection for every request.
    """

    cached_at: float
    value: Dict[str, Any] | None = None
    error: Exception | None = None


class TaskService:
    """Service for managing asynchronous tasks"""

    def __init__(self):
        self.celery_app = celery_app
        self._inspect_cache: dict[str, _InspectCacheEntry] = {}
        self._inspect_lock = threading.Lock()

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

    def _read_cached_inspect_result(
        self,
        cache_key: str,
        now: float,
        cache_ttl: float,
        failure_cache_ttl: float,
    ) -> Dict[str, Any] | None:
        """Read a cached inspect result or re-raise a cached inspect failure.

        WHY: Celery remote-control calls create broker-side resources. Keeping
        the cache decision in one place makes both success and failure paths
        bounded under repeated health polling.
        """
        entry = self._inspect_cache.get(cache_key)
        if entry is None:
            return None

        ttl = failure_cache_ttl if entry.error is not None else cache_ttl
        if ttl <= 0 or now - entry.cached_at > ttl:
            self._inspect_cache.pop(cache_key, None)
            return None

        if entry.error is not None:
            raise entry.error

        return dict(entry.value or {})

    def _inspect_once(
        self,
        timeout: Optional[float],
        collect: Callable[[Any], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run one Celery inspect operation inside an explicit connection scope.

        WHY: `control.inspect()` can otherwise acquire broker connections from
        Celery's pool for each broadcast call. A short-lived read connection
        keeps all commands for one inspection under a single cleanup boundary.
        """
        inspect_kwargs: dict[str, Any] = {}
        if timeout is not None:
            inspect_kwargs["timeout"] = timeout

        with self.celery_app.connection_for_read() as connection:
            inspect = self.celery_app.control.inspect(
                **inspect_kwargs,
                connection=connection,
            )
            return collect(inspect)

    def _inspect_with_cache(
        self,
        cache_key: str,
        timeout: Optional[float],
        cache_ttl: float,
        failure_cache_ttl: float,
        collect: Callable[[Any], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Serialize and cache Celery inspect calls across health requests.

        WHY: without singleflight behavior, concurrent health checks can all
        miss cache and create separate broker RPCs. Holding this lock around
        the short bounded inspect call trades a small wait for stable resource
        usage during outages.
        """
        now = time.monotonic()
        with self._inspect_lock:
            cached_result = self._read_cached_inspect_result(
                cache_key,
                now,
                cache_ttl,
                failure_cache_ttl,
            )
            if cached_result is not None:
                return cached_result

            try:
                result = self._inspect_once(timeout, collect)
            except Exception as exc:
                if failure_cache_ttl > 0:
                    self._inspect_cache[cache_key] = _InspectCacheEntry(
                        cached_at=time.monotonic(),
                        error=exc,
                    )
                raise

            if cache_ttl > 0:
                self._inspect_cache[cache_key] = _InspectCacheEntry(
                    cached_at=time.monotonic(),
                    value=dict(result),
                )

            return result

    def get_active_tasks(
        self,
        timeout: Optional[float] = 1.0,
        cache_ttl: float = 0.0,
        failure_cache_ttl: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Get information about currently active tasks with bounded broker RPCs.

        WHY: this endpoint is diagnostic, but it still runs from API request
        handlers. Timeout and short caching prevent slow or missing workers
        from creating unbounded broker waits.

        Returns:
            Dictionary with active tasks information
        """

        def collect(inspect: Any) -> Dict[str, Any]:
            return {
                "active": inspect.active(),
                "scheduled": inspect.scheduled(),
                "reserved": inspect.reserved(),
            }

        return self._inspect_with_cache(
            cache_key=f"active_tasks:{timeout}",
            timeout=timeout,
            cache_ttl=cache_ttl,
            failure_cache_ttl=failure_cache_ttl,
            collect=collect,
        )

    def get_worker_ping(
        self,
        timeout: Optional[float] = 1.0,
        cache_ttl: float = 0.0,
        failure_cache_ttl: float = 1.0,
    ) -> Any:
        """Ping Celery workers without collecting expensive worker metadata.

        WHY: API and orchestrator health checks only need to know whether a
        worker answers. Using ping-only health avoids the extra `stats` and
        `registered` RPCs that are unnecessary for liveness checks.
        """

        def collect(inspect: Any) -> Dict[str, Any]:
            return {"ping": inspect.ping()}

        result = self._inspect_with_cache(
            cache_key=f"worker_ping:{timeout}",
            timeout=timeout,
            cache_ttl=cache_ttl,
            failure_cache_ttl=failure_cache_ttl,
            collect=collect,
        )
        return result.get("ping")

    def get_worker_stats(
        self,
        timeout: Optional[float] = 1.0,
        cache_ttl: float = 0.0,
        failure_cache_ttl: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Get statistics about Celery workers with bounded broker RPCs.

        WHY: Health endpoints can call Celery inspection frequently. A timeout
        and optional short cache prevent slow or missing workers from making the
        API process spend repeated blocking waits on the broker.

        Returns:
            Dictionary with worker statistics
        """

        def collect(inspect: Any) -> Dict[str, Any]:
            return {
                "stats": inspect.stats(),
                "ping": inspect.ping(),
                "registered": inspect.registered(),
            }

        return self._inspect_with_cache(
            cache_key=f"worker_stats:{timeout}",
            timeout=timeout,
            cache_ttl=cache_ttl,
            failure_cache_ttl=failure_cache_ttl,
            collect=collect,
        )

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

    async def get_active_tasks_async(
        self,
        timeout: Optional[float] = 1.0,
        cache_ttl: float = 0.0,
        failure_cache_ttl: float = 1.0,
    ) -> Dict[str, Any]:
        """Async wrapper for get_active_tasks to avoid blocking the event loop."""
        return await run_in_threadpool(
            self.get_active_tasks,
            timeout=timeout,
            cache_ttl=cache_ttl,
            failure_cache_ttl=failure_cache_ttl,
        )

    async def get_worker_ping_async(
        self,
        timeout: Optional[float] = 1.0,
        cache_ttl: float = 0.0,
        failure_cache_ttl: float = 1.0,
    ) -> Any:
        """Async wrapper for lightweight worker ping from health routes."""
        return await run_in_threadpool(
            self.get_worker_ping,
            timeout=timeout,
            cache_ttl=cache_ttl,
            failure_cache_ttl=failure_cache_ttl,
        )

    async def get_worker_stats_async(
        self,
        timeout: Optional[float] = 1.0,
        cache_ttl: float = 0.0,
        failure_cache_ttl: float = 1.0,
    ) -> Dict[str, Any]:
        """Async wrapper for bounded worker inspection from async routes."""
        return await run_in_threadpool(
            self.get_worker_stats,
            timeout=timeout,
            cache_ttl=cache_ttl,
            failure_cache_ttl=failure_cache_ttl,
        )


# Global task service instance
task_service = TaskService()


def get_task_service() -> TaskService:
    """Get the global task service instance"""
    return task_service

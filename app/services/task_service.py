"""
Task Service for managing Celery tasks
"""

from typing import Any, Dict, Optional

from celery.result import AsyncResult

from app.celery.celery_main import celery_app


class TaskService:
    """Service for managing asynchronous tasks"""

    def __init__(self):
        self.celery_app = celery_app

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

    def get_worker_stats(self) -> Dict[str, Any]:
        """
        Get statistics about Celery workers

        Returns:
            Dictionary with worker statistics
        """
        inspect = self.celery_app.control.inspect()
        return {
            "stats": inspect.stats(),
            "ping": inspect.ping(),
            "registered": inspect.registered(),
        }

    def explain_async(self, **kwargs) -> str:
        """Submit explenation task"""
        task = self.celery_app.send_task("llm.explain", kwargs=kwargs)
        return task.id

    def conversation_async(self, **kwargs) -> str:
        """Submit conversation task (one turn)"""
        task = self.celery_app.send_task("llm.conversation", kwargs=kwargs)
        return task.id


# Global task service instance
task_service = TaskService()


def get_task_service() -> TaskService:
    """Get the global task service instance"""
    return task_service

"""
Celery Application Instance
"""

# Async / DB / Redis lifecycle helpers
import asyncio
import logging
import os

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from dotenv import load_dotenv

from inference_core.celery.config import CeleryConfig
from inference_core.core import redis_client
from inference_core.core.config import get_settings
from inference_core.database.sql import connection as db_connection

_worker_loop = None  # Dedicated asyncio loop per worker process


@worker_process_init.connect
def _on_worker_process_init(**_):
    """Initialize per-process resources (event loop, DB/Redis) after fork.

    Rationale:
      - Avoid using engine/clients created in parent process (fork safety)
      - Provide stable event loop for async DB and LLM operations
    """
    global _worker_loop
    # Reset cached Redis clients (created pre-fork)
    try:
        redis_client.get_redis.cache_clear()  # type: ignore[attr-defined]
        redis_client.get_sync_redis.cache_clear()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive
        pass

    # Flag if there is an inherited engine to dispose after loop creation
    inherited_engine_present = db_connection.has_engine()

    # Create dedicated event loop (avoid asyncio.run per task)
    _worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_worker_loop)

    # If there was an inherited engine, dispose it safely inside new loop
    if inherited_engine_present:

        async def _dispose_inherited():
            await db_connection.dispose_current_engine()

        _worker_loop.run_until_complete(_dispose_inherited())


@worker_process_shutdown.connect
def _on_worker_process_shutdown(**_):
    """Gracefully close per-process async resources."""
    global _worker_loop
    if _worker_loop and not _worker_loop.is_closed():

        async def _shutdown():
            try:
                await db_connection.close_database()
            except Exception:  # pragma: no cover
                pass
            # Close async Redis if supported
            try:
                r = redis_client.get_redis()
                await r.close()  # type: ignore[attr-defined]
                # For redis-py 5.x also: await r.connection_pool.disconnect()
                pool = getattr(r, "connection_pool", None)
                if pool and hasattr(pool, "disconnect"):
                    await pool.disconnect()  # type: ignore
            except Exception:  # pragma: no cover
                pass

        _worker_loop.run_until_complete(_shutdown())
        _worker_loop.close()
        _worker_loop = None


logger = logging.getLogger(__name__)


def create_celery_app() -> Celery:
    """Create and configure Celery application"""
    load_dotenv()
    load_dotenv("../../.env", override=True)
    settings = get_settings()

    celery_app = Celery(settings.app_name)

    # Load configuration
    celery_app.config_from_object(CeleryConfig)

    # Auto-discover tasks
    celery_app.autodiscover_tasks(["inference_core.celery.tasks"])

    # Explicitly include tasks modules (useful for testing/packaging)
    celery_app.conf.update(
        include=[
            "inference_core.celery.tasks.llm_tasks",
            "inference_core.celery.tasks.batch_tasks",
            "inference_core.celery.tasks.email_tasks",
            "inference_core.celery.tasks.vector_tasks",
        ]
    )

    return celery_app


# Create the Celery app instance
celery_app = create_celery_app()


# Task base configuration for shared functionality
class BaseTask(celery_app.Task):
    """Base task class with common functionality"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Task {task_id} failed: {exc}", exc_info=einfo)
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        logger.debug(f"Task {task_id} succeeded")
        super().on_success(retval, task_id, args, kwargs)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry"""
        logger.warning(f"Task {task_id} retrying: {exc}", exc_info=einfo)
        super().on_retry(exc, task_id, args, kwargs, einfo)


def get_worker_loop():
    """Get the dedicated event loop for the current worker process"""
    return _worker_loop


# Set the base task class
celery_app.Task = BaseTask

if os.getenv("DEBUG_CELERY") and int(os.getenv("DEBUG_CELERY")) == 1:
    import debugpy

    debugpy.listen(("0.0.0.0", 5678))
    print("🔴 Debugging is enabled. Waiting for debugger to attach...")
    debugpy.wait_for_client()

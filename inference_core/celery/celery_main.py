"""
Celery Application Instance
"""

# Async / DB / Redis lifecycle helpers
import asyncio
import logging
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Optional, Union

from dotenv import load_dotenv

load_dotenv()
load_dotenv("../../.env", override=True)

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown

from inference_core.celery.config import CeleryConfig
from inference_core.core import redis_client
from inference_core.core.config import Settings, get_settings
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

DEFAULT_TASK_MODULES: list[str] = [
    "inference_core.celery.tasks.llm_tasks",
    "inference_core.celery.tasks.batch_tasks",
    "inference_core.celery.tasks.email_tasks",
    "inference_core.celery.tasks.vector_tasks",
]
DEFAULT_AUTODISCOVER: list[Union[str, Sequence[str], Callable[[], Sequence[str]]]] = [
    "inference_core.celery.tasks",
]


def _merge_unique(
    base: Iterable[str], extras: Optional[Iterable[str]] = None
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for collection in (base, extras or []):
        for module in collection:
            if module not in seen:
                seen.add(module)
                ordered.append(module)
    return ordered


def _apply_autodiscover(
    celery_app: Celery,
    autodiscover: Iterable[Union[str, Sequence[str], Callable[[], Sequence[str]]]],
) -> None:
    packages: list[str] = []
    callable_entries: list[Callable[[], Sequence[str]]] = []

    for entry in autodiscover:
        if callable(entry):  # type: ignore[arg-type]
            callable_entries.append(entry)  # type: ignore[arg-type]
        elif isinstance(entry, str):
            packages.append(entry)
        else:
            packages.extend(entry)

    if packages:
        celery_app.autodiscover_tasks(packages)

    for resolver in callable_entries:
        celery_app.autodiscover_tasks(resolver)


def create_celery_app(
    *,
    custom_settings: Optional[Settings] = None,
    app_name: Optional[str] = None,
    include_modules: Optional[Iterable[str]] = None,
    autodiscover: Optional[
        Iterable[Union[str, Sequence[str], Callable[[], Sequence[str]]]]
    ] = None,
    extra_task_routes: Optional[Mapping[str, Any]] = None,
    beat_schedule_overrides: Optional[Mapping[str, Any]] = None,
    config_object: Union[type, str, object] = CeleryConfig,
    post_configure: Optional[Iterable[Callable[[Celery], None]]] = None,
) -> Celery:
    """Create and configure Celery application.

    Args:
        custom_settings: Preloaded settings instance. If not provided the global
            settings factory will be used.
        app_name: Optional Celery application name override.
        include_modules: Additional task modules to register explicitly.
        autodiscover: Extra packages or callables passed to Celery autodiscovery.
        extra_task_routes: Additional task routing rules merged with defaults.
        beat_schedule_overrides: Additional/overridden periodic schedule entries.
        config_object: Celery configuration object or dotted path.
        post_configure: Callbacks executed after configuration allowing further tweaks.
    """

    load_dotenv()
    load_dotenv("../../.env", override=True)

    settings = custom_settings or get_settings()

    celery_app = Celery(app_name or settings.app_name)

    # Load configuration first so we can safely mutate after merges.
    celery_app.config_from_object(config_object)

    include = _merge_unique(DEFAULT_TASK_MODULES, include_modules)
    celery_app.conf.update(include=include)

    autodiscover_entries = list(DEFAULT_AUTODISCOVER)
    if autodiscover:
        autodiscover_entries.extend(autodiscover)
    _apply_autodiscover(celery_app, autodiscover_entries)

    if extra_task_routes:
        merged_routes = dict(getattr(celery_app.conf, "task_routes", {}) or {})
        merged_routes.update(extra_task_routes)
        celery_app.conf.task_routes = merged_routes

    if beat_schedule_overrides:
        merged_schedule = dict(getattr(celery_app.conf, "beat_schedule", {}) or {})
        merged_schedule.update(beat_schedule_overrides)
        celery_app.conf.beat_schedule = merged_schedule

    if post_configure:
        for callback in post_configure:
            callback(celery_app)

    return celery_app


# Create the Celery app instance
celery_app = create_celery_app()


class BaseTaskMixin:
    """Reusable mixin with shared task lifecycle logging hooks."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):  # type: ignore[override]
        """Handle task failure"""
        logger.error(f"Task {task_id} failed: {exc}", exc_info=einfo)
        super().on_failure(exc, task_id, args, kwargs, einfo)  # type: ignore[misc]

    def on_success(self, retval, task_id, args, kwargs):  # type: ignore[override]
        """Handle task success"""
        logger.debug(f"Task {task_id} succeeded")
        super().on_success(retval, task_id, args, kwargs)  # type: ignore[misc]

    def on_retry(self, exc, task_id, args, kwargs, einfo):  # type: ignore[override]
        """Handle task retry"""
        logger.warning(f"Task {task_id} retrying: {exc}", exc_info=einfo)
        super().on_retry(exc, task_id, args, kwargs, einfo)  # type: ignore[misc]


def attach_base_task_class(app: Celery) -> type:
    """Attach the shared BaseTask implementation to a Celery app.

    Returns the created task class so external apps can subclass it if needed.
    """

    class _BaseTask(BaseTaskMixin, app.Task):  # type: ignore[misc]
        abstract = True

    app.Task = _BaseTask
    return _BaseTask


# Task base configuration for shared functionality on the default app
BaseTask = attach_base_task_class(celery_app)


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

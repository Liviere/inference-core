"""Async utilities for Celery tasks.

Provides a unified helper for running coroutines inside the persistent
per-worker event loop initialized in `celery_main`.

Rationale:
  - Avoids creating/destroying a new asyncio loop per task invocation.
  - Prevents cross-loop errors with asyncpg / SQLAlchemy pooled connections.
  - Central place to extend logic (e.g. instrumentation, timeout wrappers).
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_in_worker_loop(coro: Coroutine[Any, Any, T]) -> T:
    """Run coroutine on the worker's persistent event loop.

    Fallback: if loop is missing/closed (shouldn't occur if signals configured),
    create a temporary loop just for this execution.
    """
    from inference_core.celery.celery_main import _worker_loop  # type: ignore

    loop = _worker_loop
    if loop is None or loop.is_closed():  # Fallback
        tmp_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(tmp_loop)
            return tmp_loop.run_until_complete(coro)
        finally:
            tmp_loop.close()
    return loop.run_until_complete(coro)

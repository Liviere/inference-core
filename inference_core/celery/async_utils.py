"""Async utilities for Celery tasks.

Provides a unified helper for running coroutines inside the persistent
per-worker event loop initialized in `celery_main`.

Rationale:
  - Avoids creating/destroying a new asyncio loop per task invocation.
  - Prevents cross-loop errors with asyncpg / SQLAlchemy pooled connections.
  - Central place to extend logic (e.g. instrumentation, timeout wrappers).

WHY this module exists:
  When AgentService runs inside Celery, various components (MCP tools, memory
  middleware, usage logging) need to execute async code from sync callbacks.
  Previously each component created its own event loop in a thread, causing
  conflicts with the worker's persistent loop and shared resources (DB, Redis).

  This module provides:
  1. `run_in_worker_loop()` - run async code on the Celery worker loop
  2. `run_async_safely()` - unified sync-to-async wrapper that reuses worker
     loop when available, avoiding nested loop creation
  3. `get_current_loop()` - get the appropriate loop for the current context
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import logging
from typing import Any, Coroutine, Optional, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Context variable to propagate the current event loop to nested calls.
# This allows sync wrappers in tools/middleware to detect if they're running
# inside a Celery worker and reuse its loop instead of creating a new one.
_current_loop_ctx: contextvars.ContextVar[Optional[asyncio.AbstractEventLoop]] = (
    contextvars.ContextVar("celery_current_loop", default=None)
)


def get_worker_loop_safe() -> Optional[asyncio.AbstractEventLoop]:
    """Get the Celery worker loop if available, without raising.

    Returns:
        The worker event loop, or None if not in a Celery worker context
        or loop is closed.
    """
    try:
        from inference_core.celery.celery_main import get_worker_loop

        loop = get_worker_loop()
        if loop is not None and not loop.is_closed():
            return loop
    except Exception:
        pass
    return None


def get_current_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Get the current event loop for this execution context.

    Priority:
    1. Loop set via context variable (propagated from run_in_worker_loop)
    2. Celery worker loop (if running inside Celery)

    NOTE: Does NOT return running loops from async contexts
    because those cannot be safely used with run_until_complete() from sync code.

    Returns:
        The appropriate event loop for current context, or None.
    """
    # Check context variable first (set by run_in_worker_loop)
    ctx_loop = _current_loop_ctx.get()
    if ctx_loop is not None and not ctx_loop.is_closed():
        return ctx_loop

    # Check Celery worker loop
    worker_loop = get_worker_loop_safe()
    if worker_loop is not None:
        return worker_loop

    return None


def is_in_worker_context() -> bool:
    """Check if code is executing within Celery worker context.

    Returns:
        True if a valid worker loop is available.
    """
    return get_worker_loop_safe() is not None


def run_in_worker_loop(coro: Coroutine[Any, Any, T]) -> T:
    """Run coroutine on the worker's persistent event loop.

    Sets the context variable so nested sync-to-async calls can detect
    they're inside a worker and reuse the loop via `run_async_safely()`.

    Fallback: if loop is missing/closed (shouldn't occur if signals configured),
    create a temporary loop just for this execution.
    """
    from inference_core.celery.celery_main import get_worker_loop

    loop = get_worker_loop()
    if loop is None or loop.is_closed():  # Fallback
        tmp_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(tmp_loop)
            _current_loop_ctx.set(tmp_loop)
            return tmp_loop.run_until_complete(coro)
        finally:
            _current_loop_ctx.set(None)
            tmp_loop.close()

    # Set context variable for nested calls
    token = _current_loop_ctx.set(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        _current_loop_ctx.reset(token)


def run_async_safely(
    coro: Coroutine[Any, Any, T],
    timeout: Optional[float] = 30.0,
) -> T:
    """Run async code safely from sync context, reusing worker loop if available.

    This is the recommended way to call async code from sync tool `_run()` methods,
    middleware hooks, or any sync callback that may execute inside Celery.

    Behavior:
    - If inside Celery worker with context variable set and loop NOT running
      in current thread: runs directly on worker loop
    - If loop is running in current thread:
      creates temporary loop in a thread pool worker to avoid deadlock
    - Otherwise: creates temporary loop in a thread pool worker

    Args:
        coro: The coroutine to execute
        timeout: Timeout in seconds (default 30s). None for no timeout.

    Returns:
        The coroutine result

    Raises:
        TimeoutError: If execution exceeds timeout
        Exception: Any exception raised by the coroutine
    """
    # Check if we're in Celery worker context (via context variable)
    # This is the ONLY case where we can safely reuse the loop
    ctx_loop = _current_loop_ctx.get()
    worker_loop = get_worker_loop_safe()

    # Only reuse loop if:
    # 1. We have a context variable set (meaning we're inside run_in_worker_loop)
    # 2. The loop is NOT running (so we can call run_until_complete)
    if ctx_loop is not None and not ctx_loop.is_closed() and not ctx_loop.is_running():
        try:
            return ctx_loop.run_until_complete(
                _with_timeout(coro, timeout) if timeout else coro
            )
        except RuntimeError as e:
            logger.debug(f"Context loop reuse failed, falling back to thread: {e}")

    # Check worker loop (without context var) - only if not running
    if worker_loop is not None and not worker_loop.is_running():
        try:
            return worker_loop.run_until_complete(
                _with_timeout(coro, timeout) if timeout else coro
            )
        except RuntimeError as e:
            logger.debug(f"Worker loop reuse failed, falling back to thread: {e}")

    # For ALL other cases use thread fallback.
    # This avoids deadlocks when the current thread's loop is running and we'd
    # block waiting for run_coroutine_threadsafe().
    return _run_in_new_thread_loop(coro, timeout)


async def _with_timeout(coro: Coroutine[Any, Any, T], timeout: float) -> T:
    """Wrap coroutine with timeout."""
    return await asyncio.wait_for(coro, timeout=timeout)


def _run_in_new_thread_loop(
    coro: Coroutine[Any, Any, T],
    timeout: Optional[float] = 30.0,
) -> T:
    """Run coroutine in a new event loop in a thread pool worker.

    This is the fallback when no existing loop is available. Creates an
    isolated loop that won't conflict with any existing async resources.

    Args:
        coro: The coroutine to execute
        timeout: Timeout for the entire operation

    Returns:
        The coroutine result
    """

    def _runner() -> T:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if timeout:
                return loop.run_until_complete(_with_timeout(coro, timeout))
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_runner)
        # Use slightly longer timeout for the future to allow inner timeout to fire first
        effective_timeout = (timeout + 5.0) if timeout else None
        return future.result(timeout=effective_timeout)

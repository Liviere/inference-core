"""
Async Redis client

Provides a shared, lazily-initialized Async Redis connection for the inference_core.
"""

from functools import lru_cache
from typing import Any

import redis
import redis.asyncio as aioredis

from inference_core.core.config import get_settings
from inference_core.core.resource_lifecycle import close_resource, close_resource_sync


@lru_cache()
def get_redis() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.from_url(settings.redis_url, decode_responses=True)


@lru_cache()
def get_sync_redis() -> redis.Redis:
    """Get synchronous Redis client for use in non-async contexts like Celery tasks"""
    settings = get_settings()
    return redis.from_url(settings.redis_url, decode_responses=True)


def _cached_redis_clients() -> list[Any]:
    """Return already-created Redis clients without creating new connections.

    WHY: Shutdown paths must close pools that exist, but they should not create
    a brand new Redis client merely to close it immediately.
    """
    clients: list[Any] = []
    for getter in (get_redis, get_sync_redis):
        try:
            if getter.cache_info().currsize == 0:  # type: ignore[attr-defined]
                continue
            clients.append(getter())
        except Exception:
            continue
    return clients


async def close_redis_clients() -> None:
    """Close cached async and sync Redis clients, then clear their caches.

    WHY: Both FastAPI and Celery reuse cached Redis clients.  Clearing the
    caches without disconnecting the underlying pools can leave sockets open
    until garbage collection, especially in worker fork/init boundaries.
    """
    await close_resource(_cached_redis_clients(), label="redis")
    get_redis.cache_clear()
    get_sync_redis.cache_clear()


def reset_redis_clients() -> None:
    """Synchronous Redis cleanup for Celery signal handlers.

    WHY: Celery init/fork hooks are sync functions.  They still need to close
    any inherited cached Redis pools before discarding cache references.
    """
    close_resource_sync(_cached_redis_clients(), label="redis")
    get_redis.cache_clear()
    get_sync_redis.cache_clear()


async def ensure_redis_connection() -> bool:
    """Ping Redis once to ensure connection is available."""
    try:
        redis = get_redis()
        pong = await redis.ping()
        return bool(pong)
    except Exception:
        return False

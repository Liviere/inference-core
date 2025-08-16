"""
Async Redis client

Provides a shared, lazily-initialized Async Redis connection for the app.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Optional

import redis
import redis.asyncio as aioredis

from app.core.config import get_settings


@lru_cache()
def get_redis() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.from_url(settings.redis_url, decode_responses=True)


@lru_cache()
def get_sync_redis() -> redis.Redis:
    """Get synchronous Redis client for use in non-async contexts like Celery tasks"""
    settings = get_settings()
    return redis.from_url(settings.redis_url, decode_responses=True)


async def ensure_redis_connection() -> bool:
    """Ping Redis once to ensure connection is available."""
    try:
        redis = get_redis()
        pong = await redis.ping()
        return bool(pong)
    except Exception:
        return False

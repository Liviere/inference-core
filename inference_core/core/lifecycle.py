"""Application resource lifecycle utilities.

Provides centralized initialization and shutdown for infrastructure resources:
- Database engine & tables
- Redis connectivity warm-up
- Vector store (Qdrant) provider initialization & health check
- Graceful shutdown of vector store, DB and Redis

These functions are invoked from FastAPI lifespan as part of application
startup / shutdown. Consolidating this logic here avoids duplication and
keeps `main_factory` focused on assembling the ASGI app.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from inference_core.core.config import Settings
from inference_core.core.redis_client import ensure_redis_connection, get_redis
from inference_core.database.sql.connection import (
    close_database,
    create_tables,
    get_engine,
)
from inference_core.services.vector_store_service import get_vector_store_service

logger = logging.getLogger(__name__)


async def init_resources(settings: Settings) -> Dict[str, Any]:
    """Initialize core resources during application startup.

    Returns a dictionary with status information for observability / debugging.
    Non-fatal failures are logged and surfaced but do not stop startup (except
    for database table creation which is considered critical unless testing).
    """
    statuses: Dict[str, Any] = {
        "database": None,
        "redis": None,
        "vector_store": None,
    }

    # Database (critical unless testing)
    if not settings.is_testing:
        try:
            get_engine()  # warm engine / create engine singleton
            await create_tables()
            statuses["database"] = {"status": "ready"}
            logger.info("✅ Database ready (engine initialized & tables ensured)")
        except Exception as e:  # pragma: no cover (critical path)
            logger.error(f"❌ Database initialization failed: {e}")
            statuses["database"] = {"status": "error", "error": str(e)}
            raise
    else:
        statuses["database"] = {"status": "skipped_test_mode"}

    # Redis warm-up (non-fatal)
    try:
        if await ensure_redis_connection():
            statuses["redis"] = {"status": "healthy"}
            logger.info("✅ Redis ping successful (connection warm)")
        else:
            statuses["redis"] = {"status": "unreachable"}
            logger.warning("⚠️ Redis ping failed (continuing; tasks may error)")
    except Exception as e:  # pragma: no cover
        statuses["redis"] = {"status": "error", "error": str(e)}
        logger.warning(f"⚠️ Redis warm-up failed: {e}")

    # Vector store (only if explicitly configured)
    if settings.vector_backend == "qdrant":
        try:
            vs_service = get_vector_store_service()
            if vs_service.provider:
                health = await vs_service.health_check()
                statuses["vector_store"] = health
                if health.get("status") == "healthy":
                    logger.info("✅ Qdrant vector store healthy: %s", health.get("url"))
                else:
                    logger.warning(
                        "⚠️ Qdrant reported unhealthy status: %s",
                        health.get("error") or health,
                    )
        except Exception as e:  # pragma: no cover
            statuses["vector_store"] = {"status": "error", "error": str(e)}
            logger.warning(f"⚠️ Qdrant initialization/health check failed: {e}")
    else:
        statuses["vector_store"] = {"status": "disabled"}

    return statuses


async def shutdown_resources(settings: Settings) -> None:
    """Gracefully shut down resources on application termination."""
    # Vector store shutdown first (may depend on DB-less ops)
    if not settings.is_testing and settings.vector_backend == "qdrant":
        try:
            vs_service = get_vector_store_service()
            if vs_service.provider and hasattr(vs_service.provider, "close"):
                await vs_service.provider.close()  # type: ignore
                logger.info("✅ Qdrant connections closed")
        except Exception as e:  # pragma: no cover
            logger.warning(f"⚠️ Failed to close Qdrant provider: {e}")

    # Database shutdown
    if not settings.is_testing:
        try:
            await close_database()
            logger.info("✅ Database connections closed successfully")
        except Exception as e:  # pragma: no cover
            logger.error(f"❌ Failed to close database connections: {e}")

    # Redis shutdown (optional; aioredis auto handles, but explicit close is tidy)
    try:
        redis_client = get_redis()
        # Some redis clients expose .close() coroutine, others require .close() + .wait_closed()
        close_method = getattr(redis_client, "close", None)
        if callable(close_method):
            result = close_method()
            # If coroutine await it
            try:
                import inspect

                if inspect.iscoroutine(result):  # type: ignore
                    await result  # type: ignore
            except Exception:
                pass
        logger.info("✅ Redis client closed (or no-op)")
    except Exception:  # pragma: no cover
        pass

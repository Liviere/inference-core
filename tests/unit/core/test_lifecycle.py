from unittest.mock import AsyncMock, patch

import pytest

from inference_core.core.config import Settings
from inference_core.core.lifecycle import init_resources, shutdown_resources

# IMPORTANT: Check these tests below later


class TestLifecycle:

    @patch(
        "inference_core.core.lifecycle.ensure_redis_connection",
        new_callable=AsyncMock,
    )
    @patch(
        "inference_core.core.dependecies.get_settings",
        return_value=Settings(environment="testing", vector_backend=None),
    )
    @pytest.mark.asyncio
    async def test_init_resources_testing_mode_redis_healthy(
        self, mock_get_settings, mock_redis
    ):

        settings = mock_get_settings.return_value
        mock_redis.return_value = True

        statuses = await init_resources(settings)
        assert statuses["database"]["status"] == "skipped_test_mode"
        assert statuses["redis"]["status"] == "healthy"
        assert statuses["vector_store"]["status"] == "disabled"

    @patch(
        "inference_core.core.lifecycle.get_vector_store_service",
    )
    @patch(
        "inference_core.core.lifecycle.ensure_redis_connection",
        new_callable=AsyncMock,
    )
    @patch(
        "inference_core.core.dependecies.get_settings",
        return_value=Settings(environment="testing", vector_backend="qdrant"),
    )
    @pytest.mark.asyncio
    async def test_init_resources_vector_qdrant_healthy(
        self, mock_get_settings, mock_redis, mock_vector_service
    ):
        settings = mock_get_settings.return_value
        mock_redis.return_value = True

        class FakeVS:
            provider = object()

            async def health_check(self):
                return {
                    "status": "healthy",
                    "backend": "qdrant",
                    "url": "http://qdrant:6333",
                }

        mock_vector_service.return_value = FakeVS()

        statuses = await init_resources(settings)
        assert statuses["vector_store"]["status"] == "healthy"
        assert statuses["vector_store"]["backend"] == "qdrant"

    @patch("inference_core.core.lifecycle.get_redis")
    @patch("inference_core.core.lifecycle.close_database", new_callable=AsyncMock)
    @patch("inference_core.core.lifecycle.get_vector_store_service")
    @pytest.mark.asyncio
    async def test_shutdown_resources_calls_all(
        self,
        mock_get_vector_store_service,
        mock_close_database,
        mock_get_redis,
    ):
        # Production-like settings to exercise shutdown paths
        settings = Settings(environment="production", vector_backend="qdrant")

        # Fake vector provider with close()
        calls = {
            "provider_close": 0,
            "db_close": 0,
            "redis_close": 0,
            "pool_disconnect": 0,
        }

        class Provider:
            async def close(self):
                calls["provider_close"] += 1

        class FakeVS:
            provider = Provider()

        async def _db_close():
            calls["db_close"] += 1

        class Pool:
            def disconnect(self):
                calls["pool_disconnect"] += 1

        class Redis:
            def __init__(self):
                self.connection_pool = Pool()

            def close(self):
                calls["redis_close"] += 1

        mock_get_vector_store_service.return_value = FakeVS()
        mock_close_database.side_effect = _db_close
        mock_get_redis.return_value = Redis()

        await shutdown_resources(settings)

        assert calls["provider_close"] == 1
        assert calls["db_close"] == 1
        assert calls["redis_close"] == 1
        assert calls["pool_disconnect"] == 1

"""
Unit tests for inference_core.core.redis_client module

Tests Redis connection management and health check functionality.
"""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as aioredis

from inference_core.core.redis_client import (
    close_redis_clients,
    ensure_redis_connection,
    get_redis,
    get_sync_redis,
    reset_redis_clients,
)


class TestGetRedis:
    """Test get_redis function"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from inference_core.core.config import get_settings

        get_settings.cache_clear()
        # Clear the LRU cache for get_redis
        get_redis.cache_clear()
        get_sync_redis.cache_clear()

    @patch("inference_core.core.redis_client.get_settings")
    @patch("redis.asyncio.from_url")
    def test_get_redis_creates_connection(self, mock_from_url, mock_get_settings):
        """Test get_redis creates Redis connection with correct URL"""
        mock_redis = MagicMock()
        mock_from_url.return_value = mock_redis

        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_get_settings.return_value = mock_settings

        result = get_redis()

        mock_from_url.assert_called_once_with(
            "redis://localhost:6379/0", decode_responses=True
        )
        assert result == mock_redis

    @patch("redis.asyncio.from_url")
    def test_get_redis_caching(self, mock_from_url):
        """Test get_redis caches connection (LRU cache)"""
        mock_redis = MagicMock()
        mock_from_url.return_value = mock_redis

        with patch("inference_core.core.config.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.redis_url = "redis://localhost:6379/0"
            mock_get_settings.return_value = mock_settings

            # Call get_redis multiple times
            result1 = get_redis()
            result2 = get_redis()
            result3 = get_redis()

            # Should only call from_url once due to caching
            mock_from_url.assert_called_once()
            assert result1 == result2 == result3 == mock_redis

    @patch("inference_core.core.redis_client.get_settings")
    @patch("redis.asyncio.from_url")
    def test_get_redis_with_custom_url(self, mock_from_url, mock_get_settings):
        """Test get_redis with custom Redis URL"""
        mock_redis = MagicMock()
        mock_from_url.return_value = mock_redis

        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://custom-host:6380/5"
        mock_get_settings.return_value = mock_settings

        result = get_redis()

        mock_from_url.assert_called_once_with(
            "redis://custom-host:6380/5", decode_responses=True
        )
        assert result == mock_redis


class TestEnsureRedisConnection:
    """Test ensure_redis_connection function"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from inference_core.core.config import get_settings

        get_settings.cache_clear()
        get_redis.cache_clear()
        get_sync_redis.cache_clear()

    @patch("inference_core.core.redis_client.get_redis")
    @pytest.mark.asyncio
    async def test_ensure_redis_connection_success(self, mock_get_redis):
        """Test ensure_redis_connection returns True when ping succeeds"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_get_redis.return_value = mock_redis

        result = await ensure_redis_connection()

        assert result is True
        mock_redis.ping.assert_called_once()

    @patch("inference_core.core.redis_client.get_redis")
    @pytest.mark.asyncio
    async def test_ensure_redis_connection_ping_false(self, mock_get_redis):
        """Test ensure_redis_connection returns False when ping returns False"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = False
        mock_get_redis.return_value = mock_redis

        result = await ensure_redis_connection()

        assert result is False
        mock_redis.ping.assert_called_once()

    @patch("inference_core.core.redis_client.get_redis")
    @pytest.mark.asyncio
    async def test_ensure_redis_connection_ping_none(self, mock_get_redis):
        """Test ensure_redis_connection returns False when ping returns None"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = None
        mock_get_redis.return_value = mock_redis

        result = await ensure_redis_connection()

        assert result is False
        mock_redis.ping.assert_called_once()

    @patch("inference_core.core.redis_client.get_redis")
    @pytest.mark.asyncio
    async def test_ensure_redis_connection_exception(self, mock_get_redis):
        """Test ensure_redis_connection returns False when exception occurs"""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Connection failed")
        mock_get_redis.return_value = mock_redis

        result = await ensure_redis_connection()

        assert result is False
        mock_redis.ping.assert_called_once()

    @patch("inference_core.core.redis_client.get_redis")
    @pytest.mark.asyncio
    async def test_ensure_redis_connection_redis_error(self, mock_get_redis):
        """Test ensure_redis_connection handles Redis-specific errors"""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = aioredis.ConnectionError("Redis unavailable")
        mock_get_redis.return_value = mock_redis

        result = await ensure_redis_connection()

        assert result is False
        mock_redis.ping.assert_called_once()

    @patch("inference_core.core.redis_client.get_redis")
    @pytest.mark.asyncio
    async def test_ensure_redis_connection_timeout_error(self, mock_get_redis):
        """Test ensure_redis_connection handles timeout errors"""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = aioredis.TimeoutError("Request timeout")
        mock_get_redis.return_value = mock_redis

        result = await ensure_redis_connection()

        assert result is False
        mock_redis.ping.assert_called_once()

    @patch("inference_core.core.redis_client.get_redis")
    @pytest.mark.asyncio
    async def test_ensure_redis_connection_async_timeout(self, mock_get_redis):
        """Test ensure_redis_connection handles asyncio timeout"""
        import asyncio

        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = asyncio.TimeoutError()
        mock_get_redis.return_value = mock_redis

        result = await ensure_redis_connection()

        assert result is False
        mock_redis.ping.assert_called_once()


class TestRedisClientIntegration:
    """Test Redis client integration scenarios"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from inference_core.core.config import get_settings

        get_settings.cache_clear()
        get_redis.cache_clear()
        get_sync_redis.cache_clear()

    @patch("redis.asyncio.from_url")
    @pytest.mark.asyncio
    async def test_redis_client_integration(self, mock_from_url):
        """Test complete Redis client integration flow"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_from_url.return_value = mock_redis

        with patch(
            "inference_core.core.redis_client.get_settings"
        ) as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.redis_url = "redis://localhost:6379/0"
            mock_get_settings.return_value = mock_settings

            # Test get_redis creates connection
            redis_client = get_redis()
            assert redis_client == mock_redis

            # Test ensure_redis_connection works
            is_connected = await ensure_redis_connection()
            assert is_connected is True

            # Verify the connection was created with correct parameters
            mock_from_url.assert_called_once_with(
                "redis://localhost:6379/0", decode_responses=True
            )

    @patch("redis.asyncio.from_url")
    @pytest.mark.asyncio
    async def test_redis_client_connection_failure(self, mock_from_url):
        """Test Redis client behavior when connection fails"""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = aioredis.ConnectionError("Cannot connect")
        mock_from_url.return_value = mock_redis

        with patch("inference_core.core.config.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.redis_url = "redis://nonexistent:6379/0"
            mock_get_settings.return_value = mock_settings

            # Test get_redis still returns client (doesn't validate connection)
            redis_client = get_redis()
            assert redis_client == mock_redis

            # Test ensure_redis_connection detects failure
            is_connected = await ensure_redis_connection()
            assert is_connected is False

    @patch("inference_core.core.redis_client.get_redis")
    @pytest.mark.asyncio
    async def test_multiple_health_checks(self, mock_get_redis):
        """Test multiple health checks use same Redis instance"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_get_redis.return_value = mock_redis

        # Perform multiple health checks
        result1 = await ensure_redis_connection()
        result2 = await ensure_redis_connection()
        result3 = await ensure_redis_connection()

        assert result1 is True
        assert result2 is True
        assert result3 is True

        # get_redis should be called for each health check
        # (since it's not cached at the ensure_redis_connection level)
        assert mock_get_redis.call_count == 3

        # ping should be called for each health check
        assert mock_redis.ping.call_count == 3


class TestRedisLifecycle:
    """Test explicit Redis client cleanup helpers."""

    def setup_method(self):
        """Reset cached clients before each cleanup test."""
        get_redis.cache_clear()
        get_sync_redis.cache_clear()

    def teardown_method(self):
        """Leave no cached clients behind for later tests."""
        get_redis.cache_clear()
        get_sync_redis.cache_clear()

    @patch("inference_core.core.redis_client.get_settings")
    @patch("redis.from_url")
    @patch("redis.asyncio.from_url")
    @pytest.mark.asyncio
    async def test_close_redis_clients_closes_cached_async_and_sync_clients(
        self,
        mock_async_from_url,
        mock_sync_from_url,
        mock_get_settings,
    ):
        """Cached async and sync clients are closed before caches are cleared."""
        async_pool = SimpleNamespace(disconnect=AsyncMock())
        async_client = SimpleNamespace(close=AsyncMock(), connection_pool=async_pool)
        sync_pool = SimpleNamespace(disconnect=MagicMock())
        sync_client = SimpleNamespace(close=MagicMock(), connection_pool=sync_pool)

        mock_settings = MagicMock(redis_url="redis://localhost:6379/0")
        mock_get_settings.return_value = mock_settings
        mock_async_from_url.return_value = async_client
        mock_sync_from_url.return_value = sync_client

        assert get_redis() is async_client
        assert get_sync_redis() is sync_client

        await close_redis_clients()

        async_client.close.assert_awaited_once()
        async_pool.disconnect.assert_awaited_once()
        sync_client.close.assert_called_once()
        sync_pool.disconnect.assert_called_once()
        assert get_redis.cache_info().currsize == 0
        assert get_sync_redis.cache_info().currsize == 0

    @patch("inference_core.core.redis_client.get_settings")
    @patch("redis.from_url")
    @patch("redis.asyncio.from_url")
    def test_reset_redis_clients_closes_clients_from_sync_hooks(
        self,
        mock_async_from_url,
        mock_sync_from_url,
        mock_get_settings,
    ):
        """Celery sync signal hooks can close both Redis client types."""
        async_client = SimpleNamespace(close=AsyncMock(), connection_pool=None)
        sync_client = SimpleNamespace(close=MagicMock(), connection_pool=None)

        mock_settings = MagicMock(redis_url="redis://localhost:6379/0")
        mock_get_settings.return_value = mock_settings
        mock_async_from_url.return_value = async_client
        mock_sync_from_url.return_value = sync_client

        get_redis()
        get_sync_redis()

        reset_redis_clients()

        async_client.close.assert_awaited_once()
        sync_client.close.assert_called_once()
        assert get_redis.cache_info().currsize == 0
        assert get_sync_redis.cache_info().currsize == 0

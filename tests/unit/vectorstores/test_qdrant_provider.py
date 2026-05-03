from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_core.vectorstores.qdrant_provider import QdrantProvider


@pytest.mark.asyncio
async def test_health_check_reuses_async_client_without_ready_probe():
    """Repeated health checks should use the managed Qdrant client only."""
    async_client = MagicMock()
    async_client.get_collections = AsyncMock(
        return_value=SimpleNamespace(
            collections=[SimpleNamespace(name="documents"), SimpleNamespace(name="faq")]
        )
    )

    with (
        patch(
            "inference_core.vectorstores.qdrant_provider.AsyncQdrantClient",
            return_value=async_client,
        ) as client_cls,
        patch(
            "inference_core.core.config.get_settings",
            return_value=SimpleNamespace(embedding_backend="remote"),
        ),
    ):
        provider = QdrantProvider(
            {
                "url": "http://qdrant:6333",
                "api_key": "secret",
                "embedding_model": "test-embedding-model",
                "dimension": 128,
                "distance": "dot",
            }
        )

        first_result = await provider.health_check()
        second_result = await provider.health_check()

    client_cls.assert_called_once_with(url="http://qdrant:6333", api_key="secret")
    assert async_client.get_collections.await_count == 2
    assert first_result == second_result
    assert first_result["status"] == "healthy"
    assert first_result["collections"] == ["documents", "faq"]
    assert first_result["embedding_backend"] == "remote"
    assert first_result["embedding_model"] == "test-embedding-model"
    assert first_result["dimension"] == 128
    assert first_result["distance_metric"] == "dot"
    assert "ready" not in first_result


@pytest.mark.asyncio
async def test_health_check_reports_unhealthy_when_qdrant_client_fails():
    """Provider health should fail closed when the managed client cannot respond."""
    async_client = MagicMock()
    async_client.get_collections = AsyncMock(side_effect=OSError("too many files"))

    with patch(
        "inference_core.vectorstores.qdrant_provider.AsyncQdrantClient",
        return_value=async_client,
    ):
        provider = QdrantProvider({"url": "http://qdrant:6333"})

        result = await provider.health_check()

    assert result["status"] == "unhealthy"
    assert result["backend"] == "qdrant"
    assert result["url"] == "http://qdrant:6333"
    assert "too many files" in result["error"]
    assert result["embedding_backend"] == "unknown"


@pytest.mark.asyncio
async def test_close_closes_managed_clients_and_resets_references():
    """Provider shutdown should release only the managed Qdrant clients."""
    provider = QdrantProvider({"url": "http://qdrant:6333"})
    async_client = MagicMock()
    async_client.close = AsyncMock()
    sync_client = MagicMock()
    provider._async_client = async_client
    provider._sync_client = sync_client

    await provider.close()

    async_client.close.assert_awaited_once()
    sync_client.close.assert_called_once()
    assert provider._async_client is None
    assert provider._sync_client is None


@pytest.mark.asyncio
async def test_close_is_safe_when_clients_were_never_created():
    """Provider shutdown should be idempotent for disabled or unused Qdrant setups."""
    provider = QdrantProvider({"url": "http://qdrant:6333"})

    await provider.close()

    assert provider._async_client is None
    assert provider._sync_client is None

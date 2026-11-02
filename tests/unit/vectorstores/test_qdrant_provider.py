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


async def _local_provider(monkeypatch):
    """A provider on Qdrant's local mode, with a fixed two-number embedding."""
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http import models

    provider = QdrantProvider({"dimension": 2, "distance": "cosine"})
    client = AsyncQdrantClient(":memory:")
    await client.create_collection(
        "docs",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    monkeypatch.setattr(provider, "_get_async_client", AsyncMock(return_value=client))
    monkeypatch.setattr(
        provider, "_embed_texts", lambda texts: [[1.0, float(len(t))] for t in texts]
    )
    return provider


@pytest.mark.asyncio
async def test_a_prebuilt_filter_is_used_as_it_is(monkeypatch):
    """``should`` and two conditions on one key, which a dict cannot say."""
    from qdrant_client.http import models

    provider = await _local_provider(monkeypatch)
    await provider.add_texts(
        ["one", "two", "three"],
        metadatas=[
            {"owner": "a", "tags": ["x", "y"]},
            {"owner": "a", "tags": ["y"]},
            {"owner": "b", "tags": ["x", "y"]},
        ],
        collection="docs",
    )
    both = models.Filter(
        must=[
            models.FieldCondition(key="owner", match=models.MatchValue(value="a")),
            models.FieldCondition(key="tags", match=models.MatchValue(value="x")),
            models.FieldCondition(key="tags", match=models.MatchValue(value="y")),
        ]
    )
    either = models.Filter(
        should=[
            models.FieldCondition(key="owner", match=models.MatchValue(value="b")),
            models.FieldCondition(key="tags", match=models.MatchValue(value="x")),
        ]
    )

    listed, total = await provider.list_documents("docs", filters=both)
    found = await provider.similarity_search(
        "one", k=10, collection="docs", filters=either
    )

    assert (total, [d.content for d in listed]) == (1, ["one"])
    assert sorted(d.content for d in found) == ["one", "three"]


@pytest.mark.asyncio
async def test_a_dict_filter_is_read_as_before(monkeypatch):
    provider = await _local_provider(monkeypatch)
    await provider.add_texts(
        ["one", "two"],
        metadatas=[{"owner": "a", "n": 1}, {"owner": "a", "n": 5}],
        collection="docs",
    )

    listed, total = await provider.list_documents(
        "docs", filters={"owner": "a", "n": {"gte": 2}}
    )

    assert (total, [d.content for d in listed]) == (1, ["two"])

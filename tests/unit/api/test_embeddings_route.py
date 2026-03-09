"""Unit tests for the /api/v1/embeddings/generate endpoint."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_core.services.embedding_service import clear_embedding_service_cache


@pytest.fixture(autouse=True)
def _reset_embedding_singleton():
    clear_embedding_service_cache()
    yield
    clear_embedding_service_cache()


class FakeEmbeddingService:
    """Minimal fake of EmbeddingService for endpoint tests."""

    def __init__(self, embeddings: list[list[float]] | None = None):
        self._embeddings = embeddings or [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings[: len(texts)]


class FailingEmbeddingService:
    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Worker unreachable")


@pytest.mark.anyio
async def test_generate_success(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import embeddings as embed_module

    svc = FakeEmbeddingService()
    monkeypatch.setattr(embed_module, "get_embedding_service", lambda: svc)

    async for client in async_test_client_factory(llm_api_access_mode="public"):
        response = await client.post(
            "/api/v1/embeddings/generate",
            json={"texts": ["hello", "world"]},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["dimension"] == 3
    assert len(data["embeddings"]) == 2
    assert data["backend"] == "local"  # default


@pytest.mark.anyio
async def test_generate_empty_texts_rejected(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import embeddings as embed_module

    svc = FakeEmbeddingService()
    monkeypatch.setattr(embed_module, "get_embedding_service", lambda: svc)

    async for client in async_test_client_factory(llm_api_access_mode="public"):
        response = await client.post(
            "/api/v1/embeddings/generate",
            json={"texts": []},
        )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_generate_error_returns_500(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import embeddings as embed_module

    svc = FailingEmbeddingService()
    monkeypatch.setattr(embed_module, "get_embedding_service", lambda: svc)

    async for client in async_test_client_factory(llm_api_access_mode="public"):
        response = await client.post(
            "/api/v1/embeddings/generate",
            json={"texts": ["test"]},
        )

    assert response.status_code == 500
    assert "Worker unreachable" in response.json()["detail"]

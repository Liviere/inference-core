import typing as t

import pytest
from httpx import AsyncClient


class FakeDoc:
    def __init__(
        self,
        id: str,
        content: str,
        metadata: dict | None = None,
        score: float | None = None,
    ):
        self.id = id
        self.content = content
        self.metadata = metadata or {}
        self.score = score


class FakeStats:
    def __init__(self, name: str, count: int, dimension: int, distance_metric: str):
        self.name = name
        self.count = count
        self.dimension = dimension
        self.distance_metric = distance_metric


class FakeProvider:
    def get_default_collection(self) -> str:
        return "default_documents"


class FakeVectorService:
    def __init__(self):
        self.provider = FakeProvider()
        self._available = True

    @property
    def is_available(self) -> bool:
        return self._available

    async def health_check(self):
        return {
            "status": "healthy",
            "backend": "memory",
            "message": "OK",
            "collections": [self.provider.get_default_collection()],
        }

    async def add_texts(self, texts, metadatas=None, ids=None, collection=None):
        return ids or [f"id_{i}" for i, _ in enumerate(texts)]

    async def similarity_search(
        self, query: str, k: int = 4, collection=None, filters=None, **kwargs
    ):
        return [
            FakeDoc(id="d1", content=f"match:{query}", metadata={"k": k}, score=0.9),
            FakeDoc(id="d2", content=f"match:{query}", metadata={"k": k}, score=0.8),
        ][:k]

    async def list_documents(
        self,
        collection=None,
        filters=None,
        limit: int = 50,
        offset: int = 0,
        order_by: str | None = None,
        order: str = "desc",
        include_scores: bool = False,
    ):
        docs = [
            FakeDoc(id=f"d{i}", content=f"doc{i}", metadata={"i": i})
            for i in range(offset, offset + min(limit, 3))
        ]
        return docs, 10

    async def get_collection_stats(self, collection: str):
        return FakeStats(
            name=collection, count=5, dimension=384, distance_metric="cosine"
        )

    async def delete_collection(self, collection: str) -> bool:
        return collection != "missing"

    async def ensure_collection(self, collection: str) -> bool:
        return collection == "new"


@pytest.mark.anyio
async def test_vector_health_ok(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import vector as vector_module

    svc = FakeVectorService()
    monkeypatch.setattr(vector_module, "get_vector_store_service", lambda: svc)

    async for client in async_test_client_factory(llm_api_access_mode="public"):
        resp = await client.get("/api/v1/vector/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["backend"] == "memory"
        assert "collections" in data


@pytest.mark.anyio
async def test_vector_health_handles_exception(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import vector as vector_module

    class FailingSvc(FakeVectorService):
        async def health_check(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(vector_module, "get_vector_store_service", lambda: FailingSvc())

    async for client in async_test_client_factory(llm_api_access_mode="public"):
        resp = await client.get("/api/v1/vector/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "Health check failed" in data["message"]


@pytest.mark.anyio
async def test_ingest_async_submits_task(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import vector as vector_module

    svc = FakeVectorService()
    monkeypatch.setattr(vector_module, "get_vector_store_service", lambda: svc)

    class FakeTask:
        def __init__(self):
            self.id = "abc123"

        @staticmethod
        def delay(**kwargs):
            return FakeTask()

    import inference_core.celery.tasks.vector_tasks as vt

    monkeypatch.setattr(vt, "ingest_documents_task", FakeTask)

    async for client in async_test_client_factory(llm_api_access_mode="public"):
        payload = {
            "texts": ["hello", "world"],
            "metadatas": [{"a": 1}, {"a": 2}],
            "ids": [1, 2],
            "collection": "mycol",
            "async_mode": True,
        }
        resp = await client.post("/api/v1/vector/ingest", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "abc123"
        assert data["collection"] == "mycol"
        assert data["estimated_count"] == 2


@pytest.mark.anyio
async def test_ingest_sync_success(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import vector as vector_module

    svc = FakeVectorService()
    monkeypatch.setattr(vector_module, "get_vector_store_service", lambda: svc)

    async for client in async_test_client_factory(llm_api_access_mode="public"):
        payload = {
            "texts": ["a", "b", "c"],
            "metadatas": None,
            "ids": None,
            "collection": None,
            "async_mode": False,
        }
        resp = await client.post("/api/v1/vector/ingest", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "synchronous"
        assert data["collection"] == "default_documents"
        assert data["estimated_count"] == 3


@pytest.mark.anyio
async def test_ingest_sync_limit_guard(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import vector as vector_module

    svc = FakeVectorService()
    monkeypatch.setattr(vector_module, "get_vector_store_service", lambda: svc)

    async for client in async_test_client_factory(llm_api_access_mode="public"):
        payload = {
            "texts": ["x"] * 101,
            "async_mode": False,
        }
        resp = await client.post("/api/v1/vector/ingest", json=payload)
        assert resp.status_code == 400
        assert "Synchronous processing is limited" in resp.json()["detail"]


@pytest.mark.anyio
async def test_query_success(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import vector as vector_module

    svc = FakeVectorService()
    monkeypatch.setattr(vector_module, "get_vector_store_service", lambda: svc)

    async for client in async_test_client_factory(llm_api_access_mode="public"):
        payload = {"query": "foo", "k": 2}
        resp = await client.post("/api/v1/vector/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert data["collection"] == "default_documents"
        assert data["documents"][0]["content"].startswith("match:")


@pytest.mark.anyio
async def test_query_error_bubbles_to_500(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import vector as vector_module

    class ErrSvc(FakeVectorService):
        async def similarity_search(self, *a, **kw):
            raise ValueError("bad query")

    monkeypatch.setattr(vector_module, "get_vector_store_service", lambda: ErrSvc())

    async for client in async_test_client_factory(llm_api_access_mode="public"):
        resp = await client.post("/api/v1/vector/query", json={"query": "ok", "k": 1})
        assert resp.status_code == 500
        assert "Document query failed" in resp.json()["detail"]


@pytest.mark.anyio
async def test_list_documents_ok(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import vector as vector_module

    svc = FakeVectorService()
    monkeypatch.setattr(vector_module, "get_vector_store_service", lambda: svc)

    async for client in async_test_client_factory(llm_api_access_mode="public"):
        payload = {"filters": {"tag": "t"}, "limit": 2, "offset": 0}
        resp = await client.post("/api/v1/vector/list", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] <= 2
        assert data["total"] == 10
        assert data["collection"] == "default_documents"


@pytest.mark.anyio
async def test_list_documents_value_error_returns_400(
    async_test_client_factory, monkeypatch
):
    from inference_core.api.v1.routes import vector as vector_module

    class BadSvc(FakeVectorService):
        async def list_documents(self, *a, **kw):
            raise ValueError("invalid params")

    monkeypatch.setattr(vector_module, "get_vector_store_service", lambda: BadSvc())

    async for client in async_test_client_factory(llm_api_access_mode="public"):
        resp = await client.post("/api/v1/vector/list", json={"limit": 1})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "invalid params"


@pytest.mark.anyio
async def test_collection_stats_ok(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import vector as vector_module

    svc = FakeVectorService()
    monkeypatch.setattr(vector_module, "get_vector_store_service", lambda: svc)

    async for client in async_test_client_factory(llm_api_access_mode="public"):
        resp = await client.get("/api/v1/vector/collections/demo/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "demo"
        assert data["count"] == 5
        assert data["dimension"] == 384
        assert data["distance_metric"] == "cosine"


@pytest.mark.anyio
async def test_delete_collection_ok_and_404(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import vector as vector_module

    svc = FakeVectorService()
    monkeypatch.setattr(vector_module, "get_vector_store_service", lambda: svc)

    async for client in async_test_client_factory(llm_api_access_mode="public"):
        resp = await client.delete("/api/v1/vector/collections/demo")
        assert resp.status_code == 200
        assert "deleted successfully" in resp.json()["message"]

        resp2 = await client.delete("/api/v1/vector/collections/missing")
        assert resp2.status_code == 404
        assert "not found" in resp2.json()["detail"]


@pytest.mark.anyio
async def test_ensure_collection_messages(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import vector as vector_module

    svc = FakeVectorService()
    monkeypatch.setattr(vector_module, "get_vector_store_service", lambda: svc)

    async for client in async_test_client_factory(llm_api_access_mode="public"):
        resp = await client.post("/api/v1/vector/collections/new/ensure")
        assert resp.status_code == 200
        assert "created successfully" in resp.json()["message"]

        resp2 = await client.post("/api/v1/vector/collections/existing/ensure")
        assert resp2.status_code == 200
        assert "already exists" in resp2.json()["message"]

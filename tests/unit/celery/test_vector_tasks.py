import types
from types import SimpleNamespace

import pytest


class FakeCurrentTask:
    def __init__(self):
        self.states = []

    def update_state(self, *, state, meta):
        self.states.append({"state": state, "meta": meta})


@pytest.fixture()
def fake_service_ok():
    class _Provider:
        def get_default_collection(self):
            return "default"

    class _Service:
        is_available = True
        provider = _Provider()

        async def add_texts(self, *, texts, metadatas=None, ids=None, collection=None):
            return [f"doc-{i}" for i, _ in enumerate(texts, start=1)]

        async def get_collection_stats(self, collection):
            return SimpleNamespace(count=2, dimension=384, distance_metric="cosine")

        async def health_check(self):
            return {"status": "healthy", "backend": "qdrant"}

        async def delete_collection(self, collection):
            return True

    return _Service()


def test_ingest_documents_task_success(monkeypatch, fake_service_ok):
    from inference_core.celery.tasks import vector_tasks as vt

    # Patch service factory and worker loop
    monkeypatch.setattr(
        vt, "get_vector_store_service", lambda: fake_service_ok, raising=True
    )

    def _fake_run_in_worker_loop(awaitable):
        # Simulate the return from the inner coroutine in ingest_documents_task
        return (
            ["doc-1", "doc-2"],
            "default",
            SimpleNamespace(count=2, dimension=384, distance_metric="cosine"),
        )

    monkeypatch.setattr(
        vt, "run_in_worker_loop", _fake_run_in_worker_loop, raising=True
    )

    # Prepare current_task and call underlying run function with fake self
    fake_current = FakeCurrentTask()
    monkeypatch.setattr(vt, "current_task", fake_current, raising=True)
    func = vt.ingest_documents_task.run.__func__
    out = func(
        SimpleNamespace(request=SimpleNamespace(id="test-task-id")),
        texts=["a", "b"],
        metadatas=None,
        ids=None,
        collection=None,
    )

    assert out["success"] is True
    assert out["count"] == 2
    assert out["collection"] == "default"
    assert out["total_in_collection"] == 2
    assert out["dimension"] == 384
    assert out["distance_metric"] == "cosine"


def test_ingest_documents_task_unavailable(monkeypatch, fake_service_ok):
    from inference_core.celery.tasks import vector_tasks as vt

    # Make service unavailable
    fake_service_ok.is_available = False
    monkeypatch.setattr(
        vt, "get_vector_store_service", lambda: fake_service_ok, raising=True
    )

    fake_current = FakeCurrentTask()
    monkeypatch.setattr(vt, "current_task", fake_current, raising=True)

    func = vt.ingest_documents_task.run.__func__
    with pytest.raises(RuntimeError):
        func(
            SimpleNamespace(request=SimpleNamespace(id="test-task-id")),
            texts=["x"],
            metadatas=None,
            ids=None,
            collection=None,
        )

    # Ensure failure state was recorded
    assert any(s["state"] == "FAILURE" for s in fake_current.states)


def test_health_check_task_ok(monkeypatch, fake_service_ok):
    from inference_core.celery.tasks import vector_tasks as vt

    monkeypatch.setattr(
        vt, "get_vector_store_service", lambda: fake_service_ok, raising=True
    )

    def _fake_run_in_worker_loop(awaitable):
        return {"status": "healthy", "backend": "qdrant"}

    monkeypatch.setattr(
        vt, "run_in_worker_loop", _fake_run_in_worker_loop, raising=True
    )

    func = vt.health_check_task.run.__func__
    out = func(SimpleNamespace(request=SimpleNamespace(id="tid")))
    assert out == {"status": "healthy", "backend": "qdrant"}


def test_health_check_task_exception(monkeypatch):
    from inference_core.celery.tasks import vector_tasks as vt

    def _raise():
        raise RuntimeError("boom")

    monkeypatch.setattr(vt, "get_vector_store_service", _raise, raising=True)

    func = vt.health_check_task.run.__func__
    out = func(SimpleNamespace(request=SimpleNamespace(id="tid")))
    assert out["status"] == "unhealthy"
    assert "error" in out


def test_cleanup_collection_task_success(monkeypatch, fake_service_ok):
    from inference_core.celery.tasks import vector_tasks as vt

    monkeypatch.setattr(
        vt, "get_vector_store_service", lambda: fake_service_ok, raising=True
    )

    def _fake_run_in_worker_loop(awaitable):
        return True

    monkeypatch.setattr(
        vt, "run_in_worker_loop", _fake_run_in_worker_loop, raising=True
    )

    func = vt.cleanup_collection_task.run.__func__
    out = func(SimpleNamespace(request=SimpleNamespace(id="tid")), collection="demo")
    assert out["success"] is True
    assert out["collection"] == "demo"
    assert out["deleted"] is True


def test_cleanup_collection_task_unavailable(monkeypatch, fake_service_ok):
    from inference_core.celery.tasks import vector_tasks as vt

    fake_service_ok.is_available = False
    monkeypatch.setattr(
        vt, "get_vector_store_service", lambda: fake_service_ok, raising=True
    )

    func = vt.cleanup_collection_task.run.__func__
    with pytest.raises(RuntimeError):
        func(SimpleNamespace(request=SimpleNamespace(id="tid")), collection="demo")

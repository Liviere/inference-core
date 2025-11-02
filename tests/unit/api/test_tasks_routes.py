import pytest
from httpx import ASGITransport, AsyncClient

from inference_core.core.dependecies import get_db
from inference_core.database.sql.connection import (
    Base,
    create_database_engine,
    get_non_singleton_session_maker,
)
from inference_core.main_factory import create_application
from inference_core.services.task_service import get_task_service


class FakeTaskService:
    async def get_task_status_async(self, task_id: str):
        return {
            "status": "SUCCESS",
            "result": {"ok": True},
            "info": {"progress": 100},
            "traceback": None,
            "successful": True,
            "failed": False,
        }

    async def get_task_result_async(self, task_id: str, timeout=None):
        return {"value": 42}

    async def cancel_task_async(self, task_id: str):
        return True

    async def get_active_tasks_async(self):
        return {"active": {"w1": ["t1"]}, "scheduled": {}, "reserved": {}}

    async def get_worker_stats_async(self):
        return {
            "stats": {"w1": {"pool": {"max-concurrency": 1}}},
            "ping": {"w1": {"ok": "pong"}},
            "registered": {"w1": ["task.a", "task.b"]},
        }


@pytest.mark.asyncio
async def test_task_status_and_result_and_cancel():
    # Build app with temporary DB and override TaskService
    engine = create_database_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = get_non_singleton_session_maker(engine=engine)

    async def override_get_db():
        async with session_maker() as session:
            yield session

    app = create_application()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_task_service] = lambda: FakeTaskService()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # status
        r = await client.get("/api/v1/tasks/abc/status")
        assert r.status_code == 200
        body = r.json()
        assert body["task_id"] == "abc"
        assert body["status"] == "SUCCESS"
        assert body["successful"] is True

        # result
        r = await client.get("/api/v1/tasks/abc/result")
        assert r.status_code == 200
        body = r.json()
        assert body["result"] == {"value": 42}
        assert body["success"] is True

        # cancel
        r = await client.delete("/api/v1/tasks/abc")
        assert r.status_code == 200
        assert r.json()["cancelled"] is True

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_active_and_worker_stats_and_health():
    engine = create_database_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = get_non_singleton_session_maker(engine=engine)

    async def override_get_db():
        async with session_maker() as session:
            yield session

    app = create_application()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_task_service] = lambda: FakeTaskService()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # active
        r = await client.get("/api/v1/tasks/active")
        assert r.status_code == 200
        body = r.json()
        assert "active" in body and "w1" in body["active"]

        # workers stats
        r = await client.get("/api/v1/tasks/workers/stats")
        assert r.status_code == 200
        body = r.json()
        assert "stats" in body and "ping" in body and "registered" in body

        # health
        r = await client.get("/api/v1/tasks/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("healthy", "degraded")
        assert body["celery_available"] is True

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

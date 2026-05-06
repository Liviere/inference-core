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
    active_calls = []
    stats_calls = []
    ping_calls = []
    status_ids = []
    result_ids = []
    cancel_ids = []

    async def get_task_status_async(self, task_id: str):
        self.status_ids.append(task_id)
        return {
            "status": "SUCCESS",
            "result": {"ok": True},
            "info": {"progress": 100},
            "traceback": None,
            "successful": True,
            "failed": False,
        }

    async def get_task_result_async(self, task_id: str, timeout=None):
        self.result_ids.append(task_id)
        return {"value": 42}

    async def cancel_task_async(self, task_id: str):
        self.cancel_ids.append(task_id)
        return True

    async def get_active_tasks_async(
        self, timeout=None, cache_ttl=0.0, failure_cache_ttl=0.0
    ):
        self.active_calls.append(
            {
                "timeout": timeout,
                "cache_ttl": cache_ttl,
                "failure_cache_ttl": failure_cache_ttl,
            }
        )
        return {"active": {"w1": ["t1"]}, "scheduled": {}, "reserved": {}}

    async def get_worker_stats_async(
        self, timeout=None, cache_ttl=0.0, failure_cache_ttl=0.0
    ):
        self.stats_calls.append(
            {
                "timeout": timeout,
                "cache_ttl": cache_ttl,
                "failure_cache_ttl": failure_cache_ttl,
            }
        )
        return {
            "stats": {"w1": {"pool": {"max-concurrency": 1}}},
            "ping": {"w1": {"ok": "pong"}},
            "registered": {"w1": ["task.a", "task.b"]},
        }

    async def get_worker_ping_async(
        self, timeout=None, cache_ttl=0.0, failure_cache_ttl=0.0
    ):
        self.ping_calls.append(
            {
                "timeout": timeout,
                "cache_ttl": cache_ttl,
                "failure_cache_ttl": failure_cache_ttl,
            }
        )
        return {"w1": {"ok": "pong"}}


@pytest.mark.asyncio
async def test_task_status_and_result_and_cancel():
    FakeTaskService.status_ids = []
    FakeTaskService.result_ids = []
    FakeTaskService.cancel_ids = []

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
async def test_task_routes_reject_control_character_ids():
    FakeTaskService.status_ids = []
    FakeTaskService.result_ids = []
    FakeTaskService.cancel_ids = []

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
        status_response = await client.get("/api/v1/tasks/bad%0Aid/status")
        assert status_response.status_code == 422

        result_response = await client.get("/api/v1/tasks/bad%0Aid/result")
        assert result_response.status_code == 422

        cancel_response = await client.delete("/api/v1/tasks/bad%0Aid")
        assert cancel_response.status_code == 422

    assert FakeTaskService.status_ids == []
    assert FakeTaskService.result_ids == []
    assert FakeTaskService.cancel_ids == []

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_active_and_worker_stats_and_health():
    FakeTaskService.active_calls = []
    FakeTaskService.stats_calls = []
    FakeTaskService.ping_calls = []
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
        assert FakeTaskService.active_calls == [
            {"timeout": 1.0, "cache_ttl": 2.0, "failure_cache_ttl": 30.0}
        ]

        # workers stats
        r = await client.get("/api/v1/tasks/workers/stats")
        assert r.status_code == 200
        body = r.json()
        assert "stats" in body and "ping" in body and "registered" in body
        assert FakeTaskService.stats_calls == [
            {"timeout": 1.0, "cache_ttl": 5.0, "failure_cache_ttl": 30.0}
        ]

        # health
        r = await client.get("/api/v1/tasks/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("healthy", "degraded")
        assert body["celery_available"] is True
        assert FakeTaskService.ping_calls == [
            {"timeout": 1.0, "cache_ttl": 5.0, "failure_cache_ttl": 30.0}
        ]

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

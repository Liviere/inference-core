import pytest
from httpx import ASGITransport, AsyncClient

from inference_core.core.dependecies import get_db
from inference_core.database.sql.connection import (
    Base,
    create_database_engine,
    db_manager,
    get_non_singleton_session_maker,
)
from inference_core.main_factory import create_application
from inference_core.services.task_service import get_task_service


class FakeTaskService:
    calls = []

    async def get_worker_ping_async(
        self, timeout=None, cache_ttl=0.0, failure_cache_ttl=0.0
    ):
        self.calls.append(
            {
                "timeout": timeout,
                "cache_ttl": cache_ttl,
                "failure_cache_ttl": failure_cache_ttl,
            }
        )
        return {"w1": {"ok": "pong"}, "w2": {"ok": "pong"}}


class FailingTaskService:
    async def get_worker_ping_async(
        self, timeout=None, cache_ttl=0.0, failure_cache_ttl=0.0
    ):
        raise RuntimeError("broker unavailable")


class EmptyTaskService:
    async def get_worker_ping_async(
        self, timeout=None, cache_ttl=0.0, failure_cache_ttl=0.0
    ):
        return {}


class FakeModelFactory:
    def get_available_models(self):
        return {"gpt-4o-mini": True, "claude-3-haiku": False}


class FakeVectorService:
    async def health_check(self):
        return {"status": "healthy", "backend": "disabled"}


@pytest.mark.asyncio
async def test_health_overall(monkeypatch):
    FakeTaskService.calls = []
    engine = create_database_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = get_non_singleton_session_maker(engine=engine)

    async def override_get_db():
        async with session_maker() as session:
            yield session

    # Force DB healthy
    async def _hc(_session):
        return True

    monkeypatch.setattr(db_manager, "health_check", _hc)
    # Vector store
    monkeypatch.setattr(
        "inference_core.api.v1.routes.health.get_vector_store_service",
        lambda: FakeVectorService(),
        raising=True,
    )
    monkeypatch.setattr(
        "inference_core.api.v1.routes.health.get_model_factory",
        lambda: FakeModelFactory(),
        raising=True,
    )

    app = create_application()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_task_service] = lambda: FakeTaskService()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/api/v1/health/")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("healthy", "degraded")
        assert data["components"]["database"]["status"] == "healthy"
        assert data["components"]["tasks"]["active_workers"] == 2
        assert FakeTaskService.calls == [
            {"timeout": 1.0, "cache_ttl": 5.0, "failure_cache_ttl": 30.0}
        ]

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_health_no_workers_degrades(monkeypatch):
    engine = create_database_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = get_non_singleton_session_maker(engine=engine)

    async def override_get_db():
        async with session_maker() as session:
            yield session

    async def _hc(_session):
        return True

    monkeypatch.setattr(db_manager, "health_check", _hc)
    monkeypatch.setattr(
        "inference_core.api.v1.routes.health.get_vector_store_service",
        lambda: FakeVectorService(),
        raising=True,
    )
    monkeypatch.setattr(
        "inference_core.api.v1.routes.health.get_model_factory",
        lambda: FakeModelFactory(),
        raising=True,
    )

    app = create_application()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_task_service] = lambda: EmptyTaskService()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/api/v1/health/")
        assert r.status_code == 200
        data = r.json()
        assert data["components"]["tasks"]["status"] == "unavailable"
        assert data["components"]["tasks"]["active_workers"] == 0
        assert data["components"]["tasks"]["celery_available"] is False

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_health_task_service_error(monkeypatch):
    engine = create_database_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = get_non_singleton_session_maker(engine=engine)

    async def override_get_db():
        async with session_maker() as session:
            yield session

    async def _hc(_session):
        return True

    monkeypatch.setattr(db_manager, "health_check", _hc)
    monkeypatch.setattr(
        "inference_core.api.v1.routes.health.get_vector_store_service",
        lambda: FakeVectorService(),
        raising=True,
    )
    monkeypatch.setattr(
        "inference_core.api.v1.routes.health.get_model_factory",
        lambda: FakeModelFactory(),
        raising=True,
    )

    app = create_application()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_task_service] = lambda: FailingTaskService()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/api/v1/health/")
        assert r.status_code == 200
        data = r.json()
        assert data["components"]["tasks"]["status"] == "unavailable"
        assert data["components"]["tasks"]["active_workers"] == 0
        assert "broker unavailable" in data["components"]["tasks"]["message"]

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_health_database_endpoint(monkeypatch):
    engine = create_database_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = get_non_singleton_session_maker(engine=engine)

    async def override_get_db():
        async with session_maker() as session:
            yield session

    async def _db_info(_session):
        return {"status": "healthy", "driver": "sqlite", "dsn": "sqlite://"}

    monkeypatch.setattr(db_manager, "get_database_info", _db_info)

    app = create_application()
    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/api/v1/health/database")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["details"]["driver"] == "sqlite"

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_health_ping():
    app = create_application()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/api/v1/health/ping")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

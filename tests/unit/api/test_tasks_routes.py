import uuid

import pytest
from fastapi import Depends
from httpx import ASGITransport, AsyncClient

from inference_core.core.config import Settings
from inference_core.core.dependecies import get_current_user, get_db
from inference_core.core.security import (
    TokenData,
    create_access_token,
    get_current_user_token,
)
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


# --- Access control: tasks_api_access_mode and task_access_checker ---

USER_ID = str(uuid.uuid4())
OTHER_USER_ID = str(uuid.uuid4())
ADMIN_ID = str(uuid.uuid4())
INACTIVE_ADMIN_ID = str(uuid.uuid4())

USERS = {
    USER_ID: {"is_active": True, "is_superuser": False},
    OTHER_USER_ID: {"is_active": True, "is_superuser": False},
    ADMIN_ID: {"is_active": True, "is_superuser": True},
    INACTIVE_ADMIN_ID: {"is_active": False, "is_superuser": True},
}

OWNED_TASK = "task-of-user"

ADMIN_REQUESTS = [
    ("DELETE", "/api/v1/tasks/abc"),
    ("GET", "/api/v1/tasks/active"),
    ("GET", "/api/v1/tasks/workers/stats"),
    ("GET", "/api/v1/tasks/health"),
]
READ_REQUESTS = [
    ("GET", "/api/v1/tasks/abc/status"),
    ("GET", "/api/v1/tasks/abc/result"),
]


async def _fake_current_user(
    token_data: TokenData = Depends(get_current_user_token),
) -> dict:
    """The real token check, with users from ``USERS`` instead of the DB."""
    return {"id": token_data.user_id, **USERS[token_data.user_id]}


def _auth(user_id: str) -> dict:
    return {"Authorization": f"Bearer {create_access_token({'sub': user_id})}"}


class _Client:
    def __init__(self, task_access_checker=None, **overrides):
        self.task_access_checker = task_access_checker
        self.overrides = overrides

    async def __aenter__(self) -> AsyncClient:
        self.engine = create_database_engine()
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        session_maker = get_non_singleton_session_maker(engine=self.engine)

        async def override_get_db():
            async with session_maker() as session:
                yield session

        app = create_application(
            custom_settings=Settings(**self.overrides),
            task_access_checker=self.task_access_checker,
        )
        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_task_service] = lambda: FakeTaskService()
        app.dependency_overrides[get_current_user] = _fake_current_user
        self.client = AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        )
        return await self.client.__aenter__()

    async def __aexit__(self, *exc):
        await self.client.__aexit__(*exc)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await self.engine.dispose()


def test_default_mode_is_public():
    assert Settings.model_fields["tasks_api_access_mode"].default == "public"


@pytest.mark.asyncio
async def test_public_mode_without_checker_is_open():
    async with _Client(tasks_api_access_mode="public") as client:
        for method, url in READ_REQUESTS + ADMIN_REQUESTS:
            response = await client.request(method, url)
            assert response.status_code == 200, (method, url)


@pytest.mark.asyncio
async def test_user_mode_requires_a_token():
    async with _Client(tasks_api_access_mode="user") as client:
        for method, url in READ_REQUESTS + ADMIN_REQUESTS:
            response = await client.request(method, url)
            assert response.status_code == 401, (method, url)


@pytest.mark.asyncio
async def test_user_mode_lets_a_user_read_but_not_administer():
    FakeTaskService.cancel_ids = []
    async with _Client(tasks_api_access_mode="user") as client:
        for method, url in READ_REQUESTS:
            response = await client.request(method, url, headers=_auth(USER_ID))
            assert response.status_code == 200, (method, url)
        for method, url in ADMIN_REQUESTS:
            response = await client.request(method, url, headers=_auth(USER_ID))
            assert response.status_code == 403, (method, url)
    assert FakeTaskService.cancel_ids == []


@pytest.mark.asyncio
async def test_admin_routes_take_an_active_superuser():
    async with _Client(tasks_api_access_mode="user") as client:
        for method, url in ADMIN_REQUESTS:
            response = await client.request(method, url, headers=_auth(ADMIN_ID))
            assert response.status_code == 200, (method, url)
            response = await client.request(
                method, url, headers=_auth(INACTIVE_ADMIN_ID)
            )
            assert response.status_code == 400, (method, url)


@pytest.mark.asyncio
async def test_superuser_mode_closes_status_and_result_to_users():
    async with _Client(tasks_api_access_mode="superuser") as client:
        for method, url in READ_REQUESTS:
            response = await client.request(method, url, headers=_auth(USER_ID))
            assert response.status_code == 403, (method, url)
            response = await client.request(method, url, headers=_auth(ADMIN_ID))
            assert response.status_code == 200, (method, url)


def _owner_checker(calls: list):
    async def _check(task_id: str, user):
        calls.append((task_id, user["id"] if user else None))
        return user is not None and task_id == OWNED_TASK and user["id"] == USER_ID

    return _check


@pytest.mark.asyncio
async def test_checker_hides_other_users_tasks():
    FakeTaskService.status_ids = []
    FakeTaskService.result_ids = []
    calls: list = []
    async with _Client(
        task_access_checker=_owner_checker(calls), tasks_api_access_mode="user"
    ) as client:
        own_status = await client.get(
            f"/api/v1/tasks/{OWNED_TASK}/status", headers=_auth(USER_ID)
        )
        own_result = await client.get(
            f"/api/v1/tasks/{OWNED_TASK}/result", headers=_auth(USER_ID)
        )
        assert own_status.status_code == 200
        assert own_result.status_code == 200

        foreign_status = await client.get(
            f"/api/v1/tasks/{OWNED_TASK}/status", headers=_auth(OTHER_USER_ID)
        )
        foreign_result = await client.get(
            f"/api/v1/tasks/{OWNED_TASK}/result", headers=_auth(OTHER_USER_ID)
        )
        unknown = await client.get(
            "/api/v1/tasks/no-such-task/status", headers=_auth(USER_ID)
        )
        # A refusal looks like a task that does not exist.
        assert foreign_status.status_code == 404
        assert foreign_result.status_code == 404
        assert unknown.status_code == 404
        assert foreign_status.json() == unknown.json()

    # The task was read only for its owner.
    assert FakeTaskService.status_ids == [OWNED_TASK]
    assert FakeTaskService.result_ids == [OWNED_TASK]
    assert (OWNED_TASK, OTHER_USER_ID) in calls


@pytest.mark.asyncio
async def test_sync_checker_is_accepted():
    async with _Client(
        task_access_checker=lambda task_id, user: task_id == OWNED_TASK,
        tasks_api_access_mode="user",
    ) as client:
        allowed = await client.get(
            f"/api/v1/tasks/{OWNED_TASK}/status", headers=_auth(USER_ID)
        )
        refused = await client.get("/api/v1/tasks/other/status", headers=_auth(USER_ID))
    assert allowed.status_code == 200
    assert refused.status_code == 404


@pytest.mark.asyncio
async def test_checker_in_public_mode_sees_anonymous_callers():
    calls: list = []
    async with _Client(
        task_access_checker=_owner_checker(calls), tasks_api_access_mode="public"
    ) as client:
        response = await client.get(f"/api/v1/tasks/{OWNED_TASK}/status")
        # Admin routes stay as open as the mode says.
        active = await client.get("/api/v1/tasks/active")
    assert response.status_code == 404
    assert calls == [(OWNED_TASK, None)]
    assert active.status_code == 200

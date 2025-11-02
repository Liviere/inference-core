import uuid
from datetime import datetime

import pytest

from inference_core.core.dependecies import (
    get_current_active_user as dep_current_active_user,
)


@pytest.mark.anyio
async def test_create_batch_job_ok(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import batch as batch_module
    from inference_core.database.sql.models.batch import BatchJobStatus

    # Fake llm_config
    class FakeModelCfg:
        def __init__(self, provider: str):
            self.provider = provider

    class FakeBatchModelCfg:
        def __init__(self, mode: str):
            self.mode = mode

    class FakeBatchConfig:
        def is_provider_enabled(self, provider: str) -> bool:
            return True

        def get_model_config(self, provider: str, model: str):
            return FakeBatchModelCfg(mode="chat")

    class FakeLLMConfig:
        def __init__(self):
            self.providers = {"openai": {}}
            self.batch_config = FakeBatchConfig()

        def get_model_config(self, model_name: str):
            return FakeModelCfg(provider="openai")

    monkeypatch.setattr(batch_module, "llm_config", FakeLLMConfig())

    # Fake service
    class FakeBatchService:
        async def create_batch_job(self, job_data, created_by):
            return type(
                "Job",
                (),
                {
                    "id": uuid.uuid4(),
                    "status": BatchJobStatus.CREATED,
                    "provider": job_data.provider,
                    "model": job_data.model,
                    "mode": job_data.mode,
                    "request_count": job_data.request_count,
                    "success_count": 0,
                    "error_count": 0,
                    "submitted_at": None,
                    "completed_at": None,
                    "error_summary": None,
                    "result_uri": None,
                    "config_json": job_data.config_json,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "created_by": created_by,
                    "updated_by": created_by,
                },
            )()

        async def create_batch_items(self, job_id, items_data, created_by):
            return None

    # Dependency overrides: current user and service
    fake_user = {"id": "00000000-0000-0000-0000-000000000001"}

    async def _fake_user():
        return fake_user

    overrides = {
        dep_current_active_user: _fake_user,
        batch_module.get_batch_service: (lambda db=None: FakeBatchService()),
    }

    # Build app
    async for client in async_test_client_factory(
        llm_api_access_mode="public", dependency_overrides=overrides
    ):
        payload = {
            "provider": "openai",
            "model": "gpt-mini",
            "items": [
                {
                    "custom_id": "a",
                    "input": {"messages": [{"role": "user", "content": "hi"}]},
                }
            ],
        }
        resp = await client.post("/api/v1/llm/batch/", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["item_count"] == 1
        assert data["status"] == "created"


@pytest.mark.anyio
async def test_create_batch_job_unknown_provider(
    async_test_client_factory, monkeypatch
):
    from inference_core.api.v1.routes import batch as batch_module

    # llm_config with empty providers makes provider unknown
    class FakeLLMConfig:
        providers = {}
        batch_config = type(
            "BC",
            (),
            {
                "is_provider_enabled": lambda *a, **k: True,
                "get_model_config": lambda *a, **k: type("M", (), {"mode": "chat"})(),
            },
        )()

        def get_model_config(self, model_name: str):
            return None

    monkeypatch.setattr(batch_module, "llm_config", FakeLLMConfig())

    fake_user = {"id": "00000000-0000-0000-0000-000000000001"}

    async def _fake_user():
        return fake_user

    async for client in async_test_client_factory(
        llm_api_access_mode="public",
        dependency_overrides={dep_current_active_user: _fake_user},
    ):
        payload = {
            "provider": "nope",
            "model": "gpt-mini",
            "items": [{"custom_id": "a", "input": {"text": "hi"}}],
        }
        resp = await client.post("/api/v1/llm/batch/", json=payload)
        assert resp.status_code == 400
        assert "Unknown provider" in resp.json()["detail"]


@pytest.mark.anyio
async def test_get_batch_job_ok_and_404(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import batch as batch_module
    from inference_core.database.sql.models.batch import BatchJobStatus

    job_id = uuid.uuid4()
    user_id = uuid.UUID("00000000-0000-0000-0000-000000000001")

    class FakeBatchService:
        def __init__(self, exists=True):
            self.exists = exists

        async def get_batch_job(self, jid):
            if not self.exists:
                return None
            return type(
                "Job",
                (),
                {
                    "id": job_id,
                    "provider": "openai",
                    "provider_batch_id": "prov-1234567-abc",
                    "model": "gpt-mini",
                    "mode": "chat",
                    "status": BatchJobStatus.SUBMITTED,
                    "request_count": 2,
                    "success_count": 1,
                    "error_count": 0,
                    "error_summary": None,
                    "result_uri": None,
                    "submitted_at": datetime.utcnow(),
                    "completed_at": None,
                    "config_json": None,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "created_by": user_id,
                    "updated_by": user_id,
                },
            )()

        async def get_batch_events(self, jid):
            return []

    fake_user = {"id": str(user_id)}

    async def _fake_user():
        return fake_user

    # we'll pass overrides via client factory

    # 404 case
    async for client in async_test_client_factory(
        llm_api_access_mode="public",
        dependency_overrides={
            dep_current_active_user: _fake_user,
            batch_module.get_batch_service: (
                lambda db=None: FakeBatchService(exists=False)
            ),
        },
    ):
        resp = await client.get(f"/api/v1/llm/batch/{job_id}")
        assert resp.status_code == 404

    # OK case
    async for client in async_test_client_factory(
        llm_api_access_mode="public",
        dependency_overrides={
            dep_current_active_user: _fake_user,
            batch_module.get_batch_service: (
                lambda db=None: FakeBatchService(exists=True)
            ),
        },
    ):
        resp = await client.get(f"/api/v1/llm/batch/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == str(job_id)
        assert data["provider"] == "openai"


@pytest.mark.anyio
async def test_get_batch_items_pagination(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import batch as batch_module
    from inference_core.database.sql.models.batch import BatchItemStatus, BatchJobStatus

    job_id = uuid.uuid4()
    user_id = uuid.UUID("00000000-0000-0000-0000-000000000001")

    class FakeItem:
        def __init__(self, idx):
            self.id = uuid.uuid4()
            self.batch_job_id = job_id
            self.sequence_index = idx
            self.custom_external_id = f"c{idx}"
            self.input_payload = {"x": idx}
            self.output_payload = None
            self.status = BatchItemStatus.QUEUED
            self.error_detail = None
            self.created_at = datetime.utcnow()
            self.updated_at = datetime.utcnow()

    class FakeBatchService:
        async def get_batch_job(self, jid):
            return type(
                "Job",
                (),
                {
                    "id": job_id,
                    "created_by": user_id,
                    "status": BatchJobStatus.CREATED,
                },
            )()

        async def get_batch_items(self, jid, status=None):
            return [FakeItem(i) for i in range(5)]

    fake_user = {"id": str(user_id)}

    async def _fake_user():
        return fake_user

    async for client in async_test_client_factory(
        llm_api_access_mode="public",
        dependency_overrides={
            dep_current_active_user: _fake_user,
            batch_module.get_batch_service: (lambda db=None: FakeBatchService()),
        },
    ):
        resp = await client.get(f"/api/v1/llm/batch/{job_id}/items?limit=2&offset=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert data["limit"] == 2
        assert data["offset"] == 2
        assert data["has_more"] is True
        assert len(data["items"]) == 2


@pytest.mark.anyio
async def test_cancel_batch_job_paths(async_test_client_factory, monkeypatch):
    from inference_core.api.v1.routes import batch as batch_module
    from inference_core.database.sql.models.batch import BatchJobStatus

    job_id = uuid.uuid4()
    user_id = uuid.UUID("00000000-0000-0000-0000-000000000001")

    class FakeJob:
        def __init__(self, status, provider_batch_id=None):
            self.id = job_id
            self.created_by = user_id
            self.status = status
            self.provider = "openai"
            self.provider_batch_id = provider_batch_id

    class FakeProvider:
        def __init__(self, cancel_result=True):
            self._res = cancel_result

        def cancel(self, provider_batch_id):
            return self._res

    class FakeRegistry:
        def __init__(self, cancel_result=True):
            self._res = cancel_result

        def create_provider(self, provider):
            return FakeProvider(cancel_result=self._res)

    class FakeBatchService:
        def __init__(self, job_status, provider_batch_id=None):
            self.job = FakeJob(job_status, provider_batch_id)

        async def get_batch_job(self, jid):
            return self.job

        async def update_batch_job(self, job_id, update_data, updated_by=None):
            self.job.status = update_data.status
            return self.job

    fake_user = {"id": str(user_id)}

    async def _fake_user():
        return fake_user

    # we'll pass overrides via client factory

    # Already terminal -> returns message and cancelled False
    async for client in async_test_client_factory(
        llm_api_access_mode="public",
        dependency_overrides={
            dep_current_active_user: _fake_user,
            batch_module.get_batch_service: (
                lambda db=None: FakeBatchService(BatchJobStatus.COMPLETED)
            ),
        },
    ):
        resp = await client.post(f"/api/v1/llm/batch/{job_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["cancelled"] is False

    # Provider cancel success
    # we'll set get_batch_service via client factory overrides below
    # Override FastAPI dependency get_global_registry to return our FakeRegistry(cancel_result=True)
    async for client in async_test_client_factory(
        llm_api_access_mode="public",
        dependency_overrides={
            dep_current_active_user: _fake_user,
            batch_module.get_batch_service: (
                lambda db=None: FakeBatchService(
                    BatchJobStatus.SUBMITTED, provider_batch_id="prov-1"
                )
            ),
            batch_module.get_global_registry: (
                lambda: FakeRegistry(cancel_result=True)
            ),
        },
    ):
        resp = await client.post(f"/api/v1/llm/batch/{job_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["cancelled"] is True
        assert "provider cancellation succeeded" in resp.json()["message"]

    # Provider cancel failure
    async for client in async_test_client_factory(
        llm_api_access_mode="public",
        dependency_overrides={
            dep_current_active_user: _fake_user,
            batch_module.get_batch_service: (
                lambda db=None: FakeBatchService(
                    BatchJobStatus.SUBMITTED, provider_batch_id="prov-2"
                )
            ),
            batch_module.get_global_registry: (
                lambda: FakeRegistry(cancel_result=False)
            ),
        },
    ):
        resp = await client.post(f"/api/v1/llm/batch/{job_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["cancelled"] is True
        assert "local cancellation only" in resp.json()["message"]

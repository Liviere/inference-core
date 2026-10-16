import asyncio
import uuid
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient

from inference_core.database.sql.models.batch import BatchEventType, BatchJobStatus
from inference_core.llm.batch.dto import (
    PreparedSubmission,
    ProviderResultRow,
    ProviderStatus,
    ProviderSubmitResult,
)
from inference_core.llm.batch.providers.base import BaseBatchProvider
from tests.fixtures.batch_llm_config import (  # noqa: F401 - fixture import
    BATCH_CHAT_MODEL,
    batch_llm_config,
)


class FakeBatchProvider(BaseBatchProvider):
    """Batch provider that completes every job at once, without network calls.

    WHY: the timeline test drives the real submit/poll/fetch tasks, which
    would otherwise send the job to the provider's API with whatever key the
    environment holds.
    """

    PROVIDER_NAME = "openai"
    PROVIDER_BATCH_ID = "fake-batch-1"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.item_ids: list[str] = []

    def supports_model(self, model: str, mode: str) -> bool:
        return True

    def prepare_payloads(
        self,
        batch_items: list[dict],
        model: str,
        mode: str,
        config: dict | None = None,
    ) -> PreparedSubmission:
        return PreparedSubmission(
            batch_job_id=uuid.uuid4(),
            provider=self.PROVIDER_NAME,
            model=model,
            mode=mode,
            items=[{"custom_id": item["id"]} for item in batch_items],
            config=config,
        )

    def submit(self, prepared_submission: PreparedSubmission) -> ProviderSubmitResult:
        return ProviderSubmitResult(
            provider_batch_id=self.PROVIDER_BATCH_ID,
            status="validating",
            submitted_at=datetime.now(timezone.utc),
            item_count=len(prepared_submission.items),
        )

    def poll_status(self, provider_batch_id: str) -> ProviderStatus:
        return ProviderStatus(
            provider_batch_id=provider_batch_id,
            status="completed",
            normalized_status=BatchJobStatus.COMPLETED.value,
        )

    def fetch_results(self, provider_batch_id: str) -> list[ProviderResultRow]:
        return [
            ProviderResultRow(custom_id=item_id, output_text="ok", is_success=True)
            for item_id in self.item_ids
        ]

    def cancel(self, provider_batch_id: str) -> bool:
        return True


class FakeBatchRegistry:
    """Hands out one shared fake provider for every provider name."""

    def __init__(self, provider: FakeBatchProvider):
        self.provider = provider

    def create_provider(
        self, provider_name: str, config: dict | None = None
    ) -> FakeBatchProvider:
        return self.provider


async def _get_token(client: AsyncClient) -> str:
    unique = str(uuid.uuid4()).replace("-", "")[:8]
    username = f"evtuser{unique}"
    email = f"evt{unique}@example.com"
    reg = await client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "email": email,
            "password": "SecurePass123!",
            "first_name": "Evt",
            "last_name": "User",
        },
    )
    assert reg.status_code == 201, reg.text
    login = await client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": "SecurePass123!"},
    )
    assert login.status_code == 200
    return login.json()["access_token"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("batch_llm_config")
async def test_batch_events_semantic_and_order(
    async_test_client: AsyncClient, monkeypatch
):
    from inference_core.celery.tasks import batch_tasks

    provider = FakeBatchProvider()
    monkeypatch.setattr(
        batch_tasks, "get_global_registry", lambda: FakeBatchRegistry(provider)
    )
    # batch_poll queues the fetch on the broker; the test runs it directly.
    queued_fetches: list[str] = []
    monkeypatch.setattr(batch_tasks.batch_fetch, "delay", queued_fetches.append)

    token = await _get_token(async_test_client)
    headers = {"Authorization": f"Bearer {token}"}
    create_payload = {
        "provider": "openai",
        "model": BATCH_CHAT_MODEL,
        "items": [
            {"input": {"messages": [{"role": "user", "content": "hi"}]}},
            {"input": {"messages": [{"role": "user", "content": "there"}]}},
        ],
        "params": {"mode": "chat"},
    }
    resp = await async_test_client.post(
        "/api/v1/llm/batch/", json=create_payload, headers=headers
    )
    assert resp.status_code == 201, resp.text
    job_id = resp.json()["job_id"]

    items = await async_test_client.get(
        f"/api/v1/llm/batch/{job_id}/items", headers=headers
    )
    assert items.status_code == 200
    provider.item_ids = [item["id"] for item in items.json()["items"]]

    # The tasks run their own event loop, so they cannot share the test's
    # thread; each runs directly (no worker) in a separate thread.
    await asyncio.to_thread(batch_tasks.batch_submit, job_id)
    await asyncio.to_thread(batch_tasks.batch_poll)
    assert queued_fetches == [job_id]
    await asyncio.to_thread(batch_tasks.batch_fetch, job_id)

    detail = await async_test_client.get(f"/api/v1/llm/batch/{job_id}", headers=headers)
    assert detail.status_code == 200
    data = detail.json()
    assert data["status"] == BatchJobStatus.COMPLETED.value
    events = data["events"]
    assert events, "Expected at least one event"
    timestamps = [e["event_timestamp"] for e in events]
    assert timestamps == sorted(timestamps)
    types = [e["event_type"] for e in events]
    assert types[0] == BatchEventType.STATUS_CHANGE.value
    assert BatchEventType.SUBMITTED.value in types
    assert BatchEventType.FETCH_COMPLETED.value in types
    assert types.index(BatchEventType.SUBMITTED.value) < types.index(
        BatchEventType.FETCH_COMPLETED.value
    )

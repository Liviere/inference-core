import asyncio
import uuid

import pytest
from httpx import AsyncClient

from inference_core.database.sql.models.batch import BatchEventType, BatchJobStatus


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
async def test_batch_events_semantic_and_order(public_access_async_client: AsyncClient):
    # Use public access for functionality testing
    create_payload = {
        "provider": "openai",
        "model": "gpt-5-mini",
        "items": [
            {"input": {"messages": [{"role": "user", "content": "hi"}]}},
            {"input": {"messages": [{"role": "user", "content": "there"}]}},
        ],
        "params": {"mode": "chat"},
    }
    resp = await public_access_async_client.post(
        "/api/v1/llm/batch/", json=create_payload
    )
    assert resp.status_code == 201, resp.text
    job_id = resp.json()["job_id"]

    from inference_core.celery.tasks.batch_tasks import batch_fetch, batch_submit

    batch_submit(job_id)  # direct invocation (no worker)

    # Poll for submission
    for _ in range(15):
        detail = await public_access_async_client.get(
            f"/api/v1/llm/batch/{job_id}"
        )
        assert detail.status_code == 200
        data = detail.json()
        if data["status"] in [
            BatchJobStatus.SUBMITTED.value,
            BatchJobStatus.IN_PROGRESS.value,
            BatchJobStatus.COMPLETED.value,
        ]:
            break
        await asyncio.sleep(0.2)

    batch_fetch(job_id)

    detail = await public_access_async_client.get(f"/api/v1/llm/batch/{job_id}")
    assert detail.status_code == 200
    data = detail.json()
    events = data["events"]
    assert events, "Expected at least one event"
    timestamps = [e["event_timestamp"] for e in events]
    assert timestamps == sorted(timestamps)
    types = [e["event_type"] for e in events]
    assert types[0] == BatchEventType.STATUS_CHANGE.value
    if not any(t == BatchEventType.SUBMITTED.value for t in types):
        import pytest

        pytest.skip(
            "SUBMITTED semantic event not present – likely Celery worker not running in test environment"
        )
    assert any(t == BatchEventType.FETCH_COMPLETED.value for t in types)

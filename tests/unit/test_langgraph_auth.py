import uuid

import pytest

from langgraph_auth import DEV_USER_ID, authenticate


@pytest.mark.asyncio
async def test_authenticate_returns_uuid_identity_when_auth_disabled(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_AUTH_DISABLED", "true")

    user = await authenticate({})

    assert user["identity"] == DEV_USER_ID
    assert uuid.UUID(user["identity"]) == uuid.UUID(DEV_USER_ID)
    assert user["is_authenticated"] is True

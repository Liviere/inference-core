"""
Refresh with ``auth_refresh_require_active``: a deleted or inactive user's
refresh session is revoked and refused; with the option off, refresh does not
read the user at all.
"""

import pytest
from sqlalchemy import delete, update

from inference_core.core.dependecies import get_db
from inference_core.database.sql.models.user import User
from inference_core.services import refresh_session_store as rss


class _Sessions:
    """In-memory stand-in for the Redis session store."""

    def __init__(self):
        self.live: set[str] = set()
        self.revoked: list[str] = []


@pytest.fixture
def sessions(monkeypatch):
    state = _Sessions()

    async def _exists(self, jti):  # type: ignore[no-untyped-def]
        return jti in state.live

    async def _add(self, jti, user_id, exp):  # type: ignore[no-untyped-def]
        state.live.add(jti)

    async def _revoke(self, jti):  # type: ignore[no-untyped-def]
        state.live.discard(jti)
        state.revoked.append(jti)

    monkeypatch.setattr(rss.RefreshSessionStore, "exists", _exists)
    monkeypatch.setattr(rss.RefreshSessionStore, "add", _add)
    monkeypatch.setattr(rss.RefreshSessionStore, "revoke", _revoke)
    return state


async def _signed_in(client, name: str) -> str:
    reg = await client.post(
        "/api/v1/auth/register",
        json={"username": name, "email": f"{name}@example.com", "password": "Password123"},
    )
    assert reg.status_code == 201
    login = await client.post(
        "/api/v1/auth/login", json={"username": name, "password": "Password123"}
    )
    assert login.status_code == 200
    return login.cookies["refresh_token"]


async def _db(client):
    app = client._transport.app
    gen = app.dependency_overrides[get_db]()
    return gen, await gen.__anext__()


async def _change_user(client, name: str, *, remove: bool) -> None:
    gen, db = await _db(client)
    try:
        if remove:
            await db.execute(delete(User).where(User.username == name))
        else:
            await db.execute(
                update(User).where(User.username == name).values(is_active=False)
            )
        await db.commit()
    finally:
        await gen.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("remove", [False, True], ids=["inactive", "deleted"])
async def test_refresh_is_refused_for_a_gone_or_inactive_user(
    async_test_client_factory, sessions, remove
):
    async for client in async_test_client_factory(
        auth_refresh_require_active=True, auth_register_default_active=True
    ):
        cookie = await _signed_in(client, "carol")
        await _change_user(client, "carol", remove=remove)

        refused = await client.post(
            "/api/v1/auth/refresh", cookies={"refresh_token": cookie}
        )

        assert refused.status_code == 401
        assert refused.json()["detail"] == "Refresh token revoked"
        assert len(sessions.revoked) == 1
        assert not sessions.live


@pytest.mark.asyncio
async def test_refresh_still_works_for_an_active_user(
    async_test_client_factory, sessions
):
    async for client in async_test_client_factory(
        auth_refresh_require_active=True, auth_register_default_active=True
    ):
        cookie = await _signed_in(client, "dave")

        ref = await client.post("/api/v1/auth/refresh", cookies={"refresh_token": cookie})

        assert ref.status_code == 200
        assert ref.cookies["refresh_token"] != cookie


@pytest.mark.asyncio
async def test_with_the_option_off_refresh_does_not_read_the_user(
    async_test_client_factory, sessions
):
    async for client in async_test_client_factory(auth_register_default_active=True):
        cookie = await _signed_in(client, "erin")
        await _change_user(client, "erin", remove=False)

        ref = await client.post("/api/v1/auth/refresh", cookies={"refresh_token": cookie})

        assert ref.status_code == 200

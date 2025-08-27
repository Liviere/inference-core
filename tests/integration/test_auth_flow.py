"""
Integration tests for authentication flow: register, login, refresh, logout.
"""

import pytest


@pytest.mark.asyncio
async def test_register_login_me_requires_access_token(async_test_client):
    # Register
    reg = await async_test_client.post(
        "/api/v1/auth/register",
        json={
            "username": "alice",
            "email": "alice@example.com",
            "password": "Password123",
            "first_name": "Alice",
            "last_name": "A",
        },
    )
    assert reg.status_code == 201

    # Login
    login = await async_test_client.post(
        "/api/v1/auth/login",
        json={"username": "alice", "password": "Password123"},
    )
    assert login.status_code == 200
    tokens = login.json()
    assert "access_token" in tokens
    assert "refresh_token" not in tokens  # Should not be in JSON response anymore
    assert tokens["token_type"] == "bearer"
    
    # Verify refresh token is set as cookie
    cookies = login.cookies
    assert "refresh_token" in cookies

    # Calling /me with refresh token should fail (type enforcement) 
    # Extract refresh token from cookie for this test
    refresh_token_cookie = cookies["refresh_token"]
    r = await async_test_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {refresh_token_cookie}"},
    )
    assert r.status_code == 401

    # Calling /me with access token should succeed
    r_ok = await async_test_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert r_ok.status_code == 200
    body = r_ok.json()
    assert body["username"] == "alice"


@pytest.mark.asyncio
async def test_refresh_and_logout_flow(async_test_client, monkeypatch):
    # Register and login
    await async_test_client.post(
        "/api/v1/auth/register",
        json={
            "username": "bob",
            "email": "bob@example.com",
            "password": "Password123",
        },
    )
    login = await async_test_client.post(
        "/api/v1/auth/login",
        json={"username": "bob", "password": "Password123"},
    )
    assert login.status_code == 200
    
    # Verify login response structure (new cookie-based flow)
    tokens = login.json()
    assert "access_token" in tokens
    assert "refresh_token" not in tokens  # Should not be in response anymore
    assert tokens["token_type"] == "bearer"
    
    # Verify refresh token is set as cookie
    cookies = login.cookies
    assert "refresh_token" in cookies
    refresh_token_cookie = cookies["refresh_token"]

    # Monkeypatch RefreshSessionStore to avoid requiring Redis in tests
    from inference_core.services import refresh_session_store as rss

    async def _true_exists(self, jti: str) -> bool:  # type: ignore
        return True

    async def _noop(self, *_args, **_kwargs):  # type: ignore
        return None

    monkeypatch.setattr(rss.RefreshSessionStore, "exists", _true_exists)
    monkeypatch.setattr(rss.RefreshSessionStore, "add", _noop)
    monkeypatch.setattr(rss.RefreshSessionStore, "revoke", _noop)

    # Test refresh without cookie should fail (using a fresh client without cookies)
    from httpx import AsyncClient, ASGITransport
    app = async_test_client._transport.app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as fresh_client:
        ref_no_cookie = await fresh_client.post("/api/v1/auth/refresh")
        assert ref_no_cookie.status_code == 401
        assert "Refresh token not found in cookie" in ref_no_cookie.json()["detail"]

    # Refresh using cookie (no JSON body)
    ref = await async_test_client.post(
        "/api/v1/auth/refresh",
        cookies={"refresh_token": refresh_token_cookie}
    )
    assert ref.status_code == 200
    new_tokens = ref.json()
    assert "access_token" in new_tokens
    assert "refresh_token" not in new_tokens  # Should not be in response
    assert new_tokens["token_type"] == "bearer"
    
    # Verify new refresh token is set as cookie
    new_refresh_cookies = ref.cookies
    assert "refresh_token" in new_refresh_cookies
    new_refresh_token_cookie = new_refresh_cookies["refresh_token"]
    
    # Verify token rotation (new token should be different)
    assert new_refresh_token_cookie != refresh_token_cookie

    # Logout using cookie (no JSON body)
    lo = await async_test_client.post(
        "/api/v1/auth/logout",
        cookies={"refresh_token": new_refresh_token_cookie}
    )
    assert lo.status_code == 200
    assert lo.json()["success"] is True
    
    # Test logout without cookie should still succeed (best-effort)
    lo_no_cookie = await async_test_client.post("/api/v1/auth/logout")
    assert lo_no_cookie.status_code == 200
    assert lo_no_cookie.json()["success"] is True


@pytest.mark.asyncio
async def test_update_profile(async_test_client):
    # Register and login
    await async_test_client.post(
        "/api/v1/auth/register",
        json={
            "username": "carol",
            "email": "carol@example.com",
            "password": "Password123",
            "first_name": "Carol",
            "last_name": "C",
        },
    )
    login = await async_test_client.post(
        "/api/v1/auth/login",
        json={"username": "carol", "password": "Password123"},
    )
    access = login.json()["access_token"]

    # Update profile
    upd = await async_test_client.put(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {access}"},
        json={
            "first_name": "Karolina",
            "last_name": "Nowak",
            "email": "carol2@example.com",
        },
    )
    assert upd.status_code == 200
    body = upd.json()
    assert body["first_name"] == "Karolina"
    assert body["last_name"] == "Nowak"
    assert body["email"] == "carol2@example.com"


@pytest.mark.asyncio
async def test_change_password_flow(async_test_client):
    # Register and login
    await async_test_client.post(
        "/api/v1/auth/register",
        json={
            "username": "dave",
            "email": "dave@example.com",
            "password": "Password123",
        },
    )
    login = await async_test_client.post(
        "/api/v1/auth/login",
        json={"username": "dave", "password": "Password123"},
    )
    access = login.json()["access_token"]

    # Change password
    ch = await async_test_client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {access}"},
        json={"current_password": "Password123", "new_password": "NewPass123"},
    )
    assert ch.status_code == 200

    # Old password should fail
    old_login = await async_test_client.post(
        "/api/v1/auth/login",
        json={"username": "dave", "password": "Password123"},
    )
    assert old_login.status_code == 401

    # New password works
    new_login = await async_test_client.post(
        "/api/v1/auth/login",
        json={"username": "dave", "password": "NewPass123"},
    )
    assert new_login.status_code == 200


@pytest.mark.asyncio
async def test_forgot_password_success(async_test_client):
    # Register
    await async_test_client.post(
        "/api/v1/auth/register",
        json={
            "username": "eric",
            "email": "eric@example.com",
            "password": "Password123",
        },
    )

    # Unknown email still returns success (no user enumeration)
    res_unknown = await async_test_client.post(
        "/api/v1/auth/forgot-password",
        json={"email": "unknown@example.com"},
    )
    assert res_unknown.status_code == 200
    assert res_unknown.json()["success"] is True

    # Known email returns success
    res_known = await async_test_client.post(
        "/api/v1/auth/forgot-password",
        json={"email": "eric@example.com"},
    )
    assert res_known.status_code == 200
    assert res_known.json()["success"] is True


@pytest.mark.asyncio
async def test_reset_password_with_valid_token(async_test_client):
    # Register
    await async_test_client.post(
        "/api/v1/auth/register",
        json={
            "username": "frank",
            "email": "frank@example.com",
            "password": "Password123",
        },
    )

    # Create reset token using the same utility as the service
    from inference_core.core.security import security_manager

    token = security_manager.generate_password_reset_token("frank@example.com")

    # Reset password via endpoint
    res = await async_test_client.post(
        "/api/v1/auth/reset-password",
        json={"token": token, "new_password": "BrandNew123"},
    )
    assert res.status_code == 200
    assert res.json()["success"] is True

    # Login with new password should work
    login = await async_test_client.post(
        "/api/v1/auth/login",
        json={"username": "frank", "password": "BrandNew123"},
    )
    assert login.status_code == 200

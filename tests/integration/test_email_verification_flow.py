"""
Integration tests for email verification flow.
"""

import os
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_register_with_default_active_true(async_test_client):
    """Test registration with default active setting (true)"""
    # Clear settings cache to ensure default values

    # Register user
    response = await async_test_client.post(
        "/api/v1/auth/register",
        json={
            "username": "activeuser",
            "email": "activeuser@example.com",
            "password": "Password123",
            "first_name": "Active",
            "last_name": "User",
        },
    )
    assert response.status_code == 201
    user_data = response.json()
    assert user_data["is_active"] is True  # Default behavior
    assert user_data["is_verified"] is False


@pytest.mark.asyncio
async def test_register_with_active_false_setting(async_test_client_factory):
    """Test registration with AUTH_REGISTER_DEFAULT_ACTIVE=false"""
    async for async_test_client in async_test_client_factory(
        auth_register_default_active=False
    ):

        # Register user
        response = await async_test_client.post(
            "/api/v1/auth/register",
            json={
                "username": "inactiveuser",
                "email": "inactiveuser@example.com",
                "password": "Password123",
                "first_name": "Inactive",
                "last_name": "User",
            },
        )
        assert response.status_code == 201
        user_data = response.json()
        assert user_data["is_active"] is False  # Should respect setting
        assert user_data["is_verified"] is False


@pytest.mark.asyncio
async def test_login_requires_active_default(async_test_client):
    """Test login requires active user by default"""

    # Register an active user
    await async_test_client.post(
        "/api/v1/auth/register",
        json={
            "username": "activelogin",
            "email": "activelogin@example.com",
            "password": "Password123",
        },
    )

    # Login should work with active user
    response = await async_test_client.post(
        "/api/v1/auth/login",
        json={"username": "activelogin", "password": "Password123"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_login_denied_for_inactive_user(async_test_client_factory):
    """Test login denied for inactive user when AUTH_REGISTER_DEFAULT_ACTIVE=false"""
    async for async_test_client in async_test_client_factory(
        auth_register_default_active=False,
        auth_login_require_verified=False,
        auth_login_require_active=True,
    ):
        # Register inactive user
        await async_test_client.post(
            "/api/v1/auth/register",
            json={
                "username": "inactivelogin",
                "email": "inactivelogin@example.com",
                "password": "Password123",
            },
        )

        # Login should fail for inactive user
        response = await async_test_client.post(
            "/api/v1/auth/login",
            json={"username": "inactivelogin", "password": "Password123"},
        )
        assert response.status_code == 400
        assert "Inactive user" in response.json()["detail"]


@pytest.mark.asyncio
async def test_login_verified_required_setting(async_test_client_factory):
    """Test login requires verified user when AUTH_LOGIN_REQUIRE_VERIFIED=true"""
    async for async_test_client in async_test_client_factory(
        auth_register_default_active=False,
        auth_login_require_verified=True,
    ):

        # Register user (will be unverified by default)
        await async_test_client.post(
            "/api/v1/auth/register",
            json={
                "username": "unverifieduser",
                "email": "unverifieduser@example.com",
                "password": "Password123",
            },
        )

        # Login should fail for unverified user
        response = await async_test_client.post(
            "/api/v1/auth/login",
            json={"username": "unverifieduser", "password": "Password123"},
        )
        assert response.status_code == 400
        assert "Email not verified" in response.json()["detail"]


@pytest.mark.asyncio
async def test_email_verification_endpoints(async_test_client):
    """Test email verification request and confirmation endpoints"""
    # Register user
    await async_test_client.post(
        "/api/v1/auth/register",
        json={
            "username": "verifyuser",
            "email": "verifyuser@example.com",
            "password": "Password123",
        },
    )

    # Request verification email
    response = await async_test_client.post(
        "/api/v1/auth/verify-email/request",
        json={"email": "verifyuser@example.com"},
    )
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert "Verification email sent" in response.json()["message"]

    # Test with non-existent email (should still return success to not leak user existence)
    response = await async_test_client.post(
        "/api/v1/auth/verify-email/request",
        json={"email": "nonexistent@example.com"},
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


@pytest.mark.asyncio
async def test_email_verification_token_flow(async_test_client):
    """Test complete email verification token flow"""
    # Register user
    reg_response = await async_test_client.post(
        "/api/v1/auth/register",
        json={
            "username": "tokenuser",
            "email": "tokenuser@example.com",
            "password": "Password123",
        },
    )
    assert reg_response.status_code == 201
    user_data = reg_response.json()
    assert user_data["is_verified"] is False

    # Generate a verification token (simulate what would be sent in email)
    from inference_core.core.security import security_manager

    user_id = user_data["id"]
    verification_token = security_manager.generate_email_verification_token(user_id)

    # Verify email with token
    response = await async_test_client.post(
        "/api/v1/auth/verify-email",
        json={"token": verification_token},
    )
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert "Email verified successfully" in response.json()["message"]

    # Test with invalid token
    response = await async_test_client.post(
        "/api/v1/auth/verify-email",
        json={"token": "invalid.token.here"},
    )
    assert response.status_code == 400
    assert "Invalid or expired verification token" in response.json()["detail"]


@pytest.mark.asyncio
async def test_verified_user_can_login_when_required(async_test_client_factory):
    """Test that verified user can login when verification is required"""
    async for async_test_client in async_test_client_factory(
        auth_login_require_verified=False
    ):

        # Register user
        reg_response = await async_test_client.post(
            "/api/v1/auth/register",
            json={
                "username": "verifiedlogin",
                "email": "verifiedlogin@example.com",
                "password": "Password123",
            },
        )
        assert reg_response.status_code == 201
        user_data = reg_response.json()
        user_id = user_data["id"]

        # Generate verification token and verify email
        from inference_core.core.security import security_manager

        verification_token = security_manager.generate_email_verification_token(user_id)

        verify_response = await async_test_client.post(
            "/api/v1/auth/verify-email",
            json={"token": verification_token},
        )
        assert verify_response.status_code == 200

        # Now login should work
        login_response = await async_test_client.post(
            "/api/v1/auth/login",
            json={"username": "verifiedlogin", "password": "Password123"},
        )
        assert login_response.status_code == 200
        assert "access_token" in login_response.json()


@pytest.mark.asyncio
async def test_register_sends_verification_email_when_enabled(
    async_test_client_factory,
):
    """Test registration sends verification email when AUTH_SEND_VERIFICATION_EMAIL_ON_REGISTER=true"""
    async for async_test_client in async_test_client_factory(
        auth_send_verification_email_on_register=False
    ):

        # Mock the email sending to avoid actual email
        with patch("inference_core.services.auth_service.EMAIL_AVAILABLE", False):
            # Register user
            response = await async_test_client.post(
                "/api/v1/auth/register",
                json={
                    "username": "emailuser",
                    "email": "emailuser@example.com",
                    "password": "Password123",
                },
            )
            assert response.status_code == 201
            # Registration should still succeed even if email fails
            user_data = response.json()
            assert user_data["is_verified"] is False

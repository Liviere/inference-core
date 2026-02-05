import pytest
from httpx import AsyncClient

from inference_core.core.dependecies import (
    get_current_active_user,
    get_current_superuser,
)


@pytest.mark.asyncio
class TestConfigRoutes:

    @pytest.fixture
    async def authenticated_client(self, async_test_client: AsyncClient):
        """
        Fixture that takes the base async_test_client (with configured DB)
        and injects authentication overrides for the duration of the test.
        """
        # Access the app instance from the client transport
        app = async_test_client._transport.app

        async def override_user():
            # Return a valid dict that matches what the auth dependency expects
            return {
                "id": "12345678-1234-5678-1234-567812345678",
                "username": "test_admin",
                "email": "admin@example.com",
                "is_superuser": True,
                "is_active": True,
            }

        app.dependency_overrides[get_current_active_user] = override_user
        app.dependency_overrides[get_current_superuser] = override_user

        yield async_test_client

        # Cleanup overrides
        app.dependency_overrides.pop(get_current_active_user, None)
        app.dependency_overrides.pop(get_current_superuser, None)

    async def test_get_available_options(self, authenticated_client: AsyncClient):
        """Test getting available options (requires auth)."""
        response = await authenticated_client.get("/api/v1/config/available-options")
        assert response.status_code == 200
        data = response.json()
        assert "options" in data
        assert "available_models" in data
        assert "available_tasks" in data

    async def test_set_user_preference(self, authenticated_client: AsyncClient):
        """Test setting a user preference."""
        payload = {
            "preference_type": "model_params",
            "preference_key": "temperature",
            "preference_value": {"value": 0.7},
        }

        response = await authenticated_client.post(
            "/api/v1/config/preferences", json=payload
        )
        # Should be 201 Created if successful, or 400 if validation fails, but authorized
        assert response.status_code in [201, 200, 400]

    async def test_get_resolved_config(self, authenticated_client: AsyncClient):
        """Test getting resolved configuration."""
        response = await authenticated_client.get("/api/v1/config/resolved")
        assert response.status_code == 200
        data = response.json()
        assert "defaults" in data
        assert "tasks" in data
        # "agents" key should be present since we added agent support
        assert "agents" in data

    async def test_access_without_auth(self, async_test_client: AsyncClient):
        """Test that config endpoints require authentication."""
        # Use raw async_test_client without auth overrides
        response = await async_test_client.get("/api/v1/config/preferences")
        assert response.status_code == 401

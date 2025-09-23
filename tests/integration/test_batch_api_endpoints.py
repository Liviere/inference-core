"""
Test Batch API Endpoints

Integration tests for the batch processing API endpoints.
"""

import json
from uuid import UUID

import pytest
from fastapi import status
from httpx import AsyncClient

from inference_core.database.sql.models.batch import BatchItemStatus, BatchJobStatus


@pytest.mark.asyncio
@pytest.mark.integration
class TestBatchAPIEndpoints:
    """Test batch API endpoints"""

    async def get_auth_token(self, public_access_async_client: AsyncClient) -> str:
        """Helper to get an auth token for testing"""
        import uuid

        # Use unique username for each test (alphanumeric only)
        unique_id = str(uuid.uuid4()).replace("-", "")[:8]
        username = f"testuser{unique_id}"
        email = f"test{unique_id}@example.com"

        # Register a test user
        reg_response = await public_access_async_client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "email": email,
                "password": "SecurePass123!",
                "first_name": "Test",
                "last_name": "User",
            },
        )
        assert reg_response.status_code == status.HTTP_201_CREATED

        # Login to get token
        login_response = await public_access_async_client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": "SecurePass123!"},
        )
        assert login_response.status_code == status.HTTP_200_OK

        tokens = login_response.json()
        return tokens["access_token"]

    async def test_create_batch_job_unauthorized(self, async_test_client: AsyncClient):
        """Test creating batch job without authentication"""
        request_data = {
            "provider": "openai",
            "model": "gpt-5-mini",
            "items": [
                {
                    "input": {"messages": [{"role": "user", "content": "Hello"}]},
                    "custom_id": "test-1",
                }
            ],
        }

        response = await async_test_client.post("/api/v1/llm/batch/", json=request_data)

        # FastAPI returns 403 when no credentials provided to dependency that requires auth
        assert response.status_code == status.HTTP_403_FORBIDDEN

    async def test_create_batch_job_success(self, public_access_async_client: AsyncClient):
        """Test creating batch job successfully with public access mode"""
        # In public mode, no auth needed for LLM endpoints
        request_data = {
            "provider": "openai",
            "model": "gpt-5-mini",
            "items": [
                {
                    "input": {"messages": [{"role": "user", "content": "Hello"}]},
                    "custom_id": "test-1",
                },
                {
                    "input": {"messages": [{"role": "user", "content": "World"}]},
                    "custom_id": "test-2",
                },
            ],
            "params": {"mode": "chat", "temperature": 0.7},
        }

        response = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json=request_data,
        )

        assert response.status_code == status.HTTP_201_CREATED

        data = response.json()
        assert "job_id" in data
        assert data["status"] == "created"
        assert data["message"] == "Batch job created successfully"
        assert data["item_count"] == 2

        # Validate UUID format
        job_id = UUID(data["job_id"])
        assert isinstance(job_id, UUID)

        return data["job_id"]

    async def test_create_batch_job_validation_errors(
        self, public_access_async_client: AsyncClient
    ):
        """Test batch job creation validation errors"""
        auth_token = await self.get_auth_token(async_test_client)

        # Test empty items
        response = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={"provider": "openai", "model": "gpt-4o-mini", "items": []},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test missing required fields
        response = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "items": [
                    {"input": {"messages": [{"role": "user", "content": "Hello"}]}}
                ],
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_get_batch_job_success(self, public_access_async_client: AsyncClient):
        """Test getting batch job details"""
        # Use single user for create + read
        auth_token = await self.get_auth_token(async_test_client)
        create_payload = {
            "provider": "openai",
            "model": "gpt-5-mini",
            "items": [
                {
                    "input": {"messages": [{"role": "user", "content": "Hello"}]},
                    "custom_id": "test-1",
                },
                {
                    "input": {"messages": [{"role": "user", "content": "World"}]},
                    "custom_id": "test-2",
                },
            ],
            "params": {"mode": "chat", "temperature": 0.7},
        }
        create_resp = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json=create_payload,
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert create_resp.status_code == status.HTTP_201_CREATED
        job_id = create_resp.json()["job_id"]

        # Then get it
        response = await public_access_async_client.get(
            f"/api/v1/llm/batch/{job_id}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["id"] == job_id
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-5-mini"
        assert data["status"] == "created"
        assert data["request_count"] == 2
        assert data["success_count"] == 0
        assert data["error_count"] == 0
        assert data["completion_rate"] == 0.0
        assert data["pending_count"] == 2
        assert "events" in data
        assert len(data["events"]) >= 1  # Should have at least creation event

    async def test_get_batch_job_not_found(self, public_access_async_client: AsyncClient):
        """Test getting non-existent batch job"""
        auth_token = await self.get_auth_token(async_test_client)
        fake_uuid = "12345678-1234-5678-9012-123456789012"

        response = await public_access_async_client.get(
            f"/api/v1/llm/batch/{fake_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_get_batch_job_unauthorized(self, async_test_client: AsyncClient):
        """Test getting batch job without authentication"""
        fake_uuid = "12345678-1234-5678-9012-123456789012"

        response = await async_test_client.get(f"/api/v1/llm/batch/{fake_uuid}")

        assert response.status_code == status.HTTP_403_FORBIDDEN

    async def test_get_batch_items_success(self, public_access_async_client: AsyncClient):
        """Test getting batch items"""
        auth_token = await self.get_auth_token(async_test_client)
        create_resp = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "model": "gpt-5-mini",
                "items": [
                    {
                        "input": {"messages": [{"role": "user", "content": "Hello"}]},
                        "custom_id": "test-1",
                    },
                    {
                        "input": {"messages": [{"role": "user", "content": "World"}]},
                        "custom_id": "test-2",
                    },
                ],
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert create_resp.status_code == status.HTTP_201_CREATED
        job_id = create_resp.json()["job_id"]

        # Then get its items
        response = await public_access_async_client.get(
            f"/api/v1/llm/batch/{job_id}/items",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert "has_more" in data

        assert data["total"] == 2
        assert len(data["items"]) == 2
        assert data["limit"] == 100
        assert data["offset"] == 0
        assert data["has_more"] is False

        # Check item structure
        item = data["items"][0]
        assert "id" in item
        assert "sequence_index" in item
        assert "custom_external_id" in item
        assert "status" in item
        assert item["status"] == "queued"
        assert "input_payload" in item
        assert "is_completed" in item
        assert "is_successful" in item

    async def test_get_batch_items_with_status_filter(
        self, public_access_async_client: AsyncClient
    ):
        """Test getting batch items with status filter"""
        auth_token = await self.get_auth_token(async_test_client)
        create_resp = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "model": "gpt-5-mini",
                "items": [
                    {
                        "input": {"messages": [{"role": "user", "content": "Hello"}]},
                        "custom_id": "test-1",
                    },
                    {
                        "input": {"messages": [{"role": "user", "content": "World"}]},
                        "custom_id": "test-2",
                    },
                ],
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert create_resp.status_code == status.HTTP_201_CREATED
        job_id = create_resp.json()["job_id"]

        # Get items with queued status
        response = await public_access_async_client.get(
            f"/api/v1/llm/batch/{job_id}/items?item_status=queued",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 2

        # Get items with completed status (should be empty)
        response = await public_access_async_client.get(
            f"/api/v1/llm/batch/{job_id}/items?item_status=completed",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 0
        assert len(data["items"]) == 0

    async def test_get_batch_items_pagination(self, public_access_async_client: AsyncClient):
        """Test batch items pagination"""
        auth_token = await self.get_auth_token(async_test_client)
        create_resp = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "model": "gpt-5-mini",
                "items": [
                    {
                        "input": {"messages": [{"role": "user", "content": "Hello"}]},
                        "custom_id": "test-1",
                    },
                    {
                        "input": {"messages": [{"role": "user", "content": "World"}]},
                        "custom_id": "test-2",
                    },
                ],
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert create_resp.status_code == status.HTTP_201_CREATED
        job_id = create_resp.json()["job_id"]

        # Get first page with limit 1
        response = await public_access_async_client.get(
            f"/api/v1/llm/batch/{job_id}/items?limit=1&offset=0",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 2
        assert len(data["items"]) == 1
        assert data["limit"] == 1
        assert data["offset"] == 0
        assert data["has_more"] is True

        # Get second page
        response = await public_access_async_client.get(
            f"/api/v1/llm/batch/{job_id}/items?limit=1&offset=1",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 2
        assert len(data["items"]) == 1
        assert data["limit"] == 1
        assert data["offset"] == 1
        assert data["has_more"] is False

    async def test_cancel_batch_job_success(self, public_access_async_client: AsyncClient):
        """Test cancelling a batch job"""
        auth_token = await self.get_auth_token(async_test_client)
        create_resp = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "model": "gpt-5-mini",
                "items": [
                    {
                        "input": {"messages": [{"role": "user", "content": "Hello"}]},
                        "custom_id": "test-1",
                    },
                    {
                        "input": {"messages": [{"role": "user", "content": "World"}]},
                        "custom_id": "test-2",
                    },
                ],
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert create_resp.status_code == status.HTTP_201_CREATED
        job_id = create_resp.json()["job_id"]

        # Then cancel it
        response = await public_access_async_client.post(
            f"/api/v1/llm/batch/{job_id}/cancel",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "cancelled"
        assert data["cancelled"] is True
        assert "message" in data

        # Verify job is actually cancelled
        response = await public_access_async_client.get(
            f"/api/v1/llm/batch/{job_id}",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        job_data = response.json()
        assert job_data["status"] == "cancelled"

    async def test_cancel_batch_job_already_cancelled(
        self, public_access_async_client: AsyncClient
    ):
        """Test cancelling an already cancelled job"""
        auth_token = await self.get_auth_token(async_test_client)
        create_resp = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "model": "gpt-5-mini",
                "items": [
                    {
                        "input": {"messages": [{"role": "user", "content": "Hello"}]},
                        "custom_id": "test-1",
                    },
                    {
                        "input": {"messages": [{"role": "user", "content": "World"}]},
                        "custom_id": "test-2",
                    },
                ],
            },
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert create_resp.status_code == status.HTTP_201_CREATED
        job_id = create_resp.json()["job_id"]

        # Cancel it first time
        response = await public_access_async_client.post(
            f"/api/v1/llm/batch/{job_id}/cancel",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == status.HTTP_200_OK

        # Try to cancel again
        response = await public_access_async_client.post(
            f"/api/v1/llm/batch/{job_id}/cancel",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["cancelled"] is False
        assert "already cancelled" in data["message"]

    async def test_cancel_batch_job_not_found(self, public_access_async_client: AsyncClient):
        """Test cancelling non-existent batch job"""
        auth_token = await self.get_auth_token(async_test_client)
        fake_uuid = "12345678-1234-5678-9012-123456789012"

        response = await public_access_async_client.post(
            f"/api/v1/llm/batch/{fake_uuid}/cancel",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_cancel_batch_job_unauthorized(self, async_test_client: AsyncClient):
        """Test cancelling batch job without authentication"""
        fake_uuid = "12345678-1234-5678-9012-123456789012"

        response = await async_test_client.post(f"/api/v1/llm/batch/{fake_uuid}/cancel")

        assert response.status_code == status.HTTP_403_FORBIDDEN

    async def test_get_batch_job_other_user_not_found(
        self, public_access_async_client: AsyncClient
    ):
        """User B should not see User A's job (returns 404 to avoid info leak)"""
        # User A creates job
        token_user_a = await self.get_auth_token(async_test_client)
        create_resp = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "model": "gpt-5-mini",
                "items": [
                    {
                        "input": {"messages": [{"role": "user", "content": "Hello"}]},
                        "custom_id": "uA-1",
                    },
                    {
                        "input": {"messages": [{"role": "user", "content": "World"}]},
                        "custom_id": "uA-2",
                    },
                ],
            },
            headers={"Authorization": f"Bearer {token_user_a}"},
        )
        assert create_resp.status_code == status.HTTP_201_CREATED
        job_id = create_resp.json()["job_id"]

        # User B attempts to fetch
        token_user_b = await self.get_auth_token(async_test_client)
        resp = await public_access_async_client.get(
            f"/api/v1/llm/batch/{job_id}",
            headers={"Authorization": f"Bearer {token_user_b}"},
        )
        assert resp.status_code == status.HTTP_404_NOT_FOUND

    async def test_get_batch_items_other_user_not_found(
        self, public_access_async_client: AsyncClient
    ):
        """User B should not list User A's items (404)"""
        token_user_a = await self.get_auth_token(async_test_client)
        create_resp = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "model": "gpt-5-mini",
                "items": [
                    {
                        "input": {"messages": [{"role": "user", "content": "Hello"}]},
                        "custom_id": "uA-1",
                    },
                    {
                        "input": {"messages": [{"role": "user", "content": "World"}]},
                        "custom_id": "uA-2",
                    },
                ],
            },
            headers={"Authorization": f"Bearer {token_user_a}"},
        )
        assert create_resp.status_code == status.HTTP_201_CREATED
        job_id = create_resp.json()["job_id"]

        token_user_b = await self.get_auth_token(async_test_client)
        resp = await public_access_async_client.get(
            f"/api/v1/llm/batch/{job_id}/items",
            headers={"Authorization": f"Bearer {token_user_b}"},
        )
        assert resp.status_code == status.HTTP_404_NOT_FOUND

    async def test_cancel_batch_job_other_user_not_found(
        self, public_access_async_client: AsyncClient
    ):
        """User B should not cancel User A's job (404)"""
        token_user_a = await self.get_auth_token(async_test_client)
        create_resp = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "model": "gpt-5-mini",
                "items": [
                    {
                        "input": {"messages": [{"role": "user", "content": "Hello"}]},
                        "custom_id": "uA-1",
                    },
                    {
                        "input": {"messages": [{"role": "user", "content": "World"}]},
                        "custom_id": "uA-2",
                    },
                ],
            },
            headers={"Authorization": f"Bearer {token_user_a}"},
        )
        assert create_resp.status_code == status.HTTP_201_CREATED
        job_id = create_resp.json()["job_id"]

        token_user_b = await self.get_auth_token(async_test_client)
        resp = await public_access_async_client.post(
            f"/api/v1/llm/batch/{job_id}/cancel",
            headers={"Authorization": f"Bearer {token_user_b}"},
        )
        assert resp.status_code == status.HTTP_404_NOT_FOUND

    async def test_batch_job_input_size_limits(self, public_access_async_client: AsyncClient):
        """Test batch job input size limits"""
        auth_token = await self.get_auth_token(async_test_client)

        # Test too many items (over 1000)
        large_items = []
        for i in range(1001):
            large_items.append(
                {
                    "input": {
                        "messages": [{"role": "user", "content": f"Message {i}"}]
                    },
                    "custom_id": f"test-{i}",
                }
            )

        response = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={"provider": "openai", "model": "gpt-4o-mini", "items": large_items},
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_batch_endpoints_require_authentication(
        self, public_access_async_client: AsyncClient
    ):
        """Test that all batch endpoints require authentication"""
        fake_uuid = "12345678-1234-5678-9012-123456789012"

        # Test create endpoint
        response = await public_access_async_client.post("/api/v1/llm/batch/", json={})
        assert response.status_code == status.HTTP_403_FORBIDDEN

        # Test get job endpoint
        response = await public_access_async_client.get(f"/api/v1/llm/batch/{fake_uuid}")
        assert response.status_code == status.HTTP_403_FORBIDDEN

        # Test get items endpoint
        response = await public_access_async_client.get(f"/api/v1/llm/batch/{fake_uuid}/items")
        assert response.status_code == status.HTTP_403_FORBIDDEN

        # Test cancel endpoint
        response = await public_access_async_client.post(f"/api/v1/llm/batch/{fake_uuid}/cancel")
        assert response.status_code == status.HTTP_403_FORBIDDEN

    # --- Provider / Model Validation Tests ---
    async def test_create_batch_job_unknown_provider(
        self, public_access_async_client: AsyncClient
    ):
        token = await self.get_auth_token(async_test_client)
        resp = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "nope",
                "model": "gpt-5-mini",
                "items": [{"input": {"messages": [{"role": "user", "content": "Hi"}]}}],
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert "Unknown provider" in resp.text

    async def test_create_batch_job_unknown_model(self, public_access_async_client: AsyncClient):
        token = await self.get_auth_token(async_test_client)
        resp = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "model": "made-up-model-xyz",
                "items": [{"input": {"messages": [{"role": "user", "content": "Hi"}]}}],
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert "Unknown model" in resp.text

    async def test_create_batch_job_model_provider_mismatch(
        self, public_access_async_client: AsyncClient
    ):
        token = await self.get_auth_token(async_test_client)
        resp = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "model": "claude-3-5-haiku-latest",
                "items": [{"input": {"messages": [{"role": "user", "content": "Hi"}]}}],
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert "belongs to provider" in resp.text

    async def test_create_batch_job_provider_not_batch_enabled(
        self, public_access_async_client: AsyncClient
    ):
        token = await self.get_auth_token(async_test_client)
        resp = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "custom_openai_compatible",
                "model": "deepseek-ai/DeepSeek-V3-0324",
                "items": [{"input": {"messages": [{"role": "user", "content": "Hi"}]}}],
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert "not enabled for batch" in resp.text

    async def test_create_batch_job_model_not_batch_enabled(
        self, public_access_async_client: AsyncClient
    ):
        token = await self.get_auth_token(async_test_client)
        resp = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "model": "gpt-5",
                "items": [{"input": {"messages": [{"role": "user", "content": "Hi"}]}}],
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert "not configured for batch" in resp.text

    async def test_create_batch_job_mode_mismatch(self, public_access_async_client: AsyncClient):
        token = await self.get_auth_token(async_test_client)
        resp = await public_access_async_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "model": "gpt-5-mini",
                "items": [{"input": {"messages": [{"role": "user", "content": "Hi"}]}}],
                "params": {"mode": "embedding"},
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert "Mode 'embedding' not allowed" in resp.text

"""
Test Batch API Endpoints

Integration tests for the batch processing API endpoints.
"""

import json
import pytest
from uuid import UUID
from fastapi import status
from httpx import AsyncClient

from app.database.sql.models.batch import BatchJobStatus, BatchItemStatus


@pytest.mark.asyncio
@pytest.mark.integration
class TestBatchAPIEndpoints:
    """Test batch API endpoints"""

    async def get_auth_token(self, async_test_client: AsyncClient) -> str:
        """Helper to get an auth token for testing"""
        import uuid
        
        # Use unique username for each test (alphanumeric only)
        unique_id = str(uuid.uuid4()).replace('-', '')[:8]
        username = f"testuser{unique_id}"
        email = f"test{unique_id}@example.com"
        
        # Register a test user
        reg_response = await async_test_client.post(
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
        login_response = await async_test_client.post(
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
            "model": "gpt-4o-mini",
            "items": [
                {
                    "input": {"messages": [{"role": "user", "content": "Hello"}]},
                    "custom_id": "test-1"
                }
            ]
        }
        
        response = await async_test_client.post(
            "/api/v1/llm/batch/",
            json=request_data
        )
        
        # FastAPI returns 403 when no credentials provided to dependency that requires auth
        assert response.status_code == status.HTTP_403_FORBIDDEN

    async def test_create_batch_job_success(self, async_test_client: AsyncClient):
        """Test creating batch job successfully"""
        auth_token = await self.get_auth_token(async_test_client)
        
        request_data = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "items": [
                {
                    "input": {"messages": [{"role": "user", "content": "Hello"}]},
                    "custom_id": "test-1"
                },
                {
                    "input": {"messages": [{"role": "user", "content": "World"}]},
                    "custom_id": "test-2"
                }
            ],
            "params": {"mode": "chat", "temperature": 0.7}
        }
        
        response = await async_test_client.post(
            "/api/v1/llm/batch/",
            json=request_data,
            headers={"Authorization": f"Bearer {auth_token}"}
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

    async def test_create_batch_job_validation_errors(self, async_test_client: AsyncClient):
        """Test batch job creation validation errors"""
        auth_token = await self.get_auth_token(async_test_client)
        
        # Test empty items
        response = await async_test_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "items": []
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test missing required fields
        response = await async_test_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "items": [{"input": {"messages": [{"role": "user", "content": "Hello"}]}}]
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_get_batch_job_success(self, async_test_client: AsyncClient):
        """Test getting batch job details"""
        # First create a job
        job_id = await self.test_create_batch_job_success(async_test_client)
        auth_token = await self.get_auth_token(async_test_client)
        
        # Then get it
        response = await async_test_client.get(
            f"/api/v1/llm/batch/{job_id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["id"] == job_id
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4o-mini"
        assert data["status"] == "created"
        assert data["request_count"] == 2
        assert data["success_count"] == 0
        assert data["error_count"] == 0
        assert data["completion_rate"] == 0.0
        assert data["pending_count"] == 2
        assert "events" in data
        assert len(data["events"]) >= 1  # Should have at least creation event

    async def test_get_batch_job_not_found(self, async_test_client: AsyncClient):
        """Test getting non-existent batch job"""
        auth_token = await self.get_auth_token(async_test_client)
        fake_uuid = "12345678-1234-5678-9012-123456789012"
        
        response = await async_test_client.get(
            f"/api/v1/llm/batch/{fake_uuid}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_get_batch_job_unauthorized(self, async_test_client: AsyncClient):
        """Test getting batch job without authentication"""
        fake_uuid = "12345678-1234-5678-9012-123456789012"
        
        response = await async_test_client.get(f"/api/v1/llm/batch/{fake_uuid}")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN

    async def test_get_batch_items_success(self, async_test_client: AsyncClient):
        """Test getting batch items"""
        # First create a job
        job_id = await self.test_create_batch_job_success(async_test_client)
        auth_token = await self.get_auth_token(async_test_client)
        
        # Then get its items
        response = await async_test_client.get(
            f"/api/v1/llm/batch/{job_id}/items",
            headers={"Authorization": f"Bearer {auth_token}"}
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

    async def test_get_batch_items_with_status_filter(self, async_test_client: AsyncClient):
        """Test getting batch items with status filter"""
        # First create a job
        job_id = await self.test_create_batch_job_success(async_test_client)
        auth_token = await self.get_auth_token(async_test_client)
        
        # Get items with queued status
        response = await async_test_client.get(
            f"/api/v1/llm/batch/{job_id}/items?status=queued",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 2
        
        # Get items with completed status (should be empty)
        response = await async_test_client.get(
            f"/api/v1/llm/batch/{job_id}/items?status=completed",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 0
        assert len(data["items"]) == 0

    async def test_get_batch_items_pagination(self, async_test_client: AsyncClient):
        """Test batch items pagination"""
        # First create a job
        job_id = await self.test_create_batch_job_success(async_test_client)
        auth_token = await self.get_auth_token(async_test_client)
        
        # Get first page with limit 1
        response = await async_test_client.get(
            f"/api/v1/llm/batch/{job_id}/items?limit=1&offset=0",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 2
        assert len(data["items"]) == 1
        assert data["limit"] == 1
        assert data["offset"] == 0
        assert data["has_more"] is True
        
        # Get second page
        response = await async_test_client.get(
            f"/api/v1/llm/batch/{job_id}/items?limit=1&offset=1",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 2
        assert len(data["items"]) == 1
        assert data["limit"] == 1
        assert data["offset"] == 1
        assert data["has_more"] is False

    async def test_cancel_batch_job_success(self, async_test_client: AsyncClient):
        """Test cancelling a batch job"""
        # First create a job
        job_id = await self.test_create_batch_job_success(async_test_client)
        auth_token = await self.get_auth_token(async_test_client)
        
        # Then cancel it
        response = await async_test_client.post(
            f"/api/v1/llm/batch/{job_id}/cancel",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "cancelled"
        assert data["cancelled"] is True
        assert "message" in data
        
        # Verify job is actually cancelled
        response = await async_test_client.get(
            f"/api/v1/llm/batch/{job_id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        job_data = response.json()
        assert job_data["status"] == "cancelled"

    async def test_cancel_batch_job_already_cancelled(self, async_test_client: AsyncClient):
        """Test cancelling an already cancelled job"""
        # First create and cancel a job
        job_id = await self.test_create_batch_job_success(async_test_client)
        auth_token = await self.get_auth_token(async_test_client)
        
        # Cancel it first time
        response = await async_test_client.post(
            f"/api/v1/llm/batch/{job_id}/cancel",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == status.HTTP_200_OK
        
        # Try to cancel again
        response = await async_test_client.post(
            f"/api/v1/llm/batch/{job_id}/cancel",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["cancelled"] is False
        assert "already cancelled" in data["message"]

    async def test_cancel_batch_job_not_found(self, async_test_client: AsyncClient):
        """Test cancelling non-existent batch job"""
        auth_token = await self.get_auth_token(async_test_client)
        fake_uuid = "12345678-1234-5678-9012-123456789012"
        
        response = await async_test_client.post(
            f"/api/v1/llm/batch/{fake_uuid}/cancel",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_cancel_batch_job_unauthorized(self, async_test_client: AsyncClient):
        """Test cancelling batch job without authentication"""
        fake_uuid = "12345678-1234-5678-9012-123456789012"
        
        response = await async_test_client.post(f"/api/v1/llm/batch/{fake_uuid}/cancel")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN

    async def test_batch_job_input_size_limits(self, async_test_client: AsyncClient):
        """Test batch job input size limits"""
        auth_token = await self.get_auth_token(async_test_client)
        
        # Test too many items (over 1000)
        large_items = []
        for i in range(1001):
            large_items.append({
                "input": {"messages": [{"role": "user", "content": f"Message {i}"}]},
                "custom_id": f"test-{i}"
            })
        
        response = await async_test_client.post(
            "/api/v1/llm/batch/",
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "items": large_items
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_batch_endpoints_require_authentication(self, async_test_client: AsyncClient):
        """Test that all batch endpoints require authentication"""
        fake_uuid = "12345678-1234-5678-9012-123456789012"
        
        # Test create endpoint
        response = await async_test_client.post("/api/v1/llm/batch/", json={})
        assert response.status_code == status.HTTP_403_FORBIDDEN
        
        # Test get job endpoint
        response = await async_test_client.get(f"/api/v1/llm/batch/{fake_uuid}")
        assert response.status_code == status.HTTP_403_FORBIDDEN
        
        # Test get items endpoint
        response = await async_test_client.get(f"/api/v1/llm/batch/{fake_uuid}/items")
        assert response.status_code == status.HTTP_403_FORBIDDEN
        
        # Test cancel endpoint
        response = await async_test_client.post(f"/api/v1/llm/batch/{fake_uuid}/cancel")
        assert response.status_code == status.HTTP_403_FORBIDDEN
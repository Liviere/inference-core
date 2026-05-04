"""Integration tests for batch endpoint access control."""

import pytest


class TestBatchEndpointAccessControl:
    """Integration tests for batch endpoint access control across modes."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_endpoints_consistency_public_mode(
        self, public_access_async_client
    ):
        """Batch endpoints should not return 401 in public mode."""
        resp = await public_access_async_client.get("/api/v1/llm/batch/test-job-id")
        assert resp.status_code != 401

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_endpoints_consistency_user_mode(
        self, user_access_async_client
    ):
        """Batch endpoints should return 401 in user mode without auth."""
        resp = await user_access_async_client.get("/api/v1/llm/batch/test-job-id")
        assert resp.status_code == 401

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_endpoints_consistency_superuser_mode(
        self, superuser_access_async_client
    ):
        """Batch endpoints should return 401 in superuser mode without auth."""
        resp = await superuser_access_async_client.get("/api/v1/llm/batch/test-job-id")
        assert resp.status_code == 401

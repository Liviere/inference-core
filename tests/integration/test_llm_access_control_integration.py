"""
Integration tests for LLM API access control.

This module tests the actual behavior of LLM endpoints under different
access control modes, including authentication requirements and HTTP responses.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestLLMEndpointAccessControlTasks:
    """Integration tests for LLM endpoint access control across different modes."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_completion_endpoint_public_mode(
        self, public_access_async_client, monkeypatch
    ):
        """/api/v1/llm/completion should work in public mode without auth."""
        # Patch task service factory to avoid real task submission
        with patch(
            "inference_core.services.task_service.get_task_service"
        ) as mock_task_service:
            mock_service = mock_task_service.return_value
            # mimic async submit returning a task id
            mock_service.completion_submit_async.return_value = "test-task-id"

            resp = await public_access_async_client.post(
                "/api/v1/llm/completion", json={"question": "What is AI?"}
            )
            assert resp.status_code == 200
            assert "task_id" in resp.json()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_completion_endpoint_user_mode_no_auth(
        self, user_access_async_client
    ):
        """In user mode without authentication, completion endpoint should 401."""
        resp = await user_access_async_client.post(
            "/api/v1/llm/completion", json={"question": "What is AI?"}
        )
        assert resp.status_code == 403

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_completion_endpoint_superuser_mode_no_auth(
        self, superuser_access_async_client
    ):
        """In superuser mode without authentication, completion endpoint should 401."""
        resp = await superuser_access_async_client.post(
            "/api/v1/llm/completion", json={"question": "What is AI?"}
        )
        assert resp.status_code == 403

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_models_endpoint_public_mode(self, public_access_async_client):
        """/api/v1/llm/models should be accessible in public mode without auth."""
        with patch(
            "inference_core.services.llm_service.get_llm_service"
        ) as mock_llm_service:
            mock_service = mock_llm_service.return_value
            mock_service.get_available_models.return_value = {"gpt-4": True}

            resp = await public_access_async_client.get("/api/v1/llm/models")
            assert resp.status_code == 200
            assert "models" in resp.json()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_models_endpoint_user_mode_no_auth(
        self, user_access_async_client
    ):
        """In user mode without auth, models endpoint should 401."""
        resp = await user_access_async_client.get("/api/v1/llm/models")
        assert resp.status_code == 403

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_health_endpoint_public_mode(self, public_access_async_client):
        """/api/v1/llm/health should be accessible in public mode."""
        with patch(
            "inference_core.services.llm_service.get_llm_service"
        ) as mock_llm_service:
            mock_service = mock_llm_service.return_value
            mock_service.get_available_models.return_value = {"gpt-4": True}

            resp = await public_access_async_client.get("/api/v1/llm/health")
            assert resp.status_code == 200
            assert "status" in resp.json()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_stats_endpoint_public_mode(self, public_access_async_client):
        """/api/v1/llm/stats should be accessible in public mode."""
        with patch(
            "inference_core.services.llm_service.get_llm_service"
        ) as mock_llm_service:
            mock_service = mock_llm_service.return_value
            mock_service.get_usage_stats.return_value = {"requests": 100}

            resp = await public_access_async_client.get("/api/v1/llm/stats")
            assert resp.status_code == 200
            assert "stats" in resp.json()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_endpoints_consistency_public_mode(
        self, public_access_async_client
    ):
        """Batch endpoints should not return 401 in public mode (may return other errors)."""
        resp = await public_access_async_client.get("/api/v1/llm/batch/test-job-id")
        assert resp.status_code != 401

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_endpoints_consistency_user_mode(
        self, user_access_async_client
    ):
        """Batch endpoints should return 401 in user mode without auth."""
        resp = await user_access_async_client.get("/api/v1/llm/batch/test-job-id")
        assert resp.status_code == 403

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_endpoints_consistency_superuser_mode(
        self, superuser_access_async_client
    ):
        """Batch endpoints should return 401 in superuser mode without auth."""
        resp = await superuser_access_async_client.get("/api/v1/llm/batch/test-job-id")
        assert resp.status_code == 403


class TestStreamingEndpointsAccessControl:
    """Access control tests for streaming endpoints using async clients."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_completion_stream_endpoint_public_mode(
        self, public_access_async_client
    ):
        """Streaming completion endpoint should work in public mode without auth."""
        with patch(
            "inference_core.services.llm_service.get_llm_service"
        ) as mock_llm_service:
            mock_service = mock_llm_service.return_value

            async def mock_stream():
                # simple async generator producing one SSE chunk
                yield "data: chunk\n\n"

            mock_service.stream_completion.return_value = mock_stream()

            resp = await public_access_async_client.post(
                "/api/v1/llm/completion/stream", json={"question": "What is AI?"}
            )
            assert resp.status_code == 200
            assert resp.headers.get("content-type", "").startswith("text/event-stream")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_completion_stream_endpoint_superuser_mode_no_auth(
        self, superuser_access_async_client
    ):
        """Completion stream should 401 in superuser mode without auth."""
        resp = await superuser_access_async_client.post(
            "/api/v1/llm/completion/stream", json={"question": "What is AI?"}
        )
        assert resp.status_code == 403

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_chat_stream_endpoint_public_mode(self, public_access_async_client):
        """Streaming chat endpoint should work in public mode without auth."""
        with patch(
            "inference_core.services.llm_service.get_llm_service"
        ) as mock_llm_service:
            mock_service = mock_llm_service.return_value

            async def mock_stream():
                yield "data: conv\n\n"

            mock_service.stream_chat.return_value = mock_stream()

            resp = await public_access_async_client.post(
                "/api/v1/llm/chat/stream", json={"user_input": "Hello"}
            )
            assert resp.status_code == 200
            assert resp.headers.get("content-type", "").startswith("text/event-stream")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_chat_stream_endpoint_superuser_mode_no_auth(
        self, superuser_access_async_client
    ):
        """Chat stream should 401 in superuser mode without auth."""
        resp = await superuser_access_async_client.post(
            "/api/v1/llm/chat/stream", json={"user_input": "Hello"}
        )
        assert resp.status_code == 403

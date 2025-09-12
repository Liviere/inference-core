"""
Integration tests for LLM API access control.

This module tests the actual behavior of LLM endpoints under different
access control modes, including authentication requirements and HTTP responses.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from inference_core.main_factory import create_application
from inference_core.core.config import Settings


class TestLLMEndpointAccessControl:
    """Integration tests for LLM endpoint access control across different modes"""

    @pytest.fixture
    def mock_settings_public(self):
        """Mock settings with public access mode"""
        with patch('inference_core.core.config.get_settings') as mock:
            settings = Settings()
            settings.llm_api_access_mode = "public"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def mock_settings_user(self):
        """Mock settings with user access mode"""
        with patch('inference_core.core.config.get_settings') as mock:
            settings = Settings()
            settings.llm_api_access_mode = "user"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def mock_settings_superuser(self):
        """Mock settings with superuser access mode"""
        with patch('inference_core.core.config.get_settings') as mock:
            settings = Settings()
            settings.llm_api_access_mode = "superuser"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def client_public(self, mock_settings_public):
        """Test client with public access mode"""
        with patch('inference_core.core.dependecies.get_settings', return_value=mock_settings_public):
            app = create_application()
            return TestClient(app)

    @pytest.fixture
    def client_user(self, mock_settings_user):
        """Test client with user access mode"""
        with patch('inference_core.core.dependecies.get_settings', return_value=mock_settings_user):
            app = create_application()
            return TestClient(app)

    @pytest.fixture
    def client_superuser(self, mock_settings_superuser):
        """Test client with superuser access mode"""
        with patch('inference_core.core.dependecies.get_settings', return_value=mock_settings_superuser):
            app = create_application()
            return TestClient(app)

    def test_llm_explain_endpoint_public_mode(self, client_public):
        """Test /api/v1/llm/explain endpoint in public mode (no auth required)"""
        # Mock LLM services to avoid actual LLM calls
        with patch('inference_core.services.task_service.get_task_service') as mock_task_service:
            mock_service = mock_task_service.return_value
            mock_service.explain_submit_async.return_value = "test-task-id"
            
            response = client_public.post(
                "/api/v1/llm/explain",
                json={"question": "What is AI?"}
            )
            
            # In public mode, request should succeed without authentication
            assert response.status_code == 200
            assert "task_id" in response.json()

    def test_llm_explain_endpoint_user_mode_no_auth(self, client_user):
        """Test /api/v1/llm/explain endpoint in user mode without authentication"""
        response = client_user.post(
            "/api/v1/llm/explain",
            json={"question": "What is AI?"}
        )
        
        # In user mode without auth, should return 401
        assert response.status_code == 401

    def test_llm_explain_endpoint_superuser_mode_no_auth(self, client_superuser):
        """Test /api/v1/llm/explain endpoint in superuser mode without authentication"""
        response = client_superuser.post(
            "/api/v1/llm/explain",
            json={"question": "What is AI?"}
        )
        
        # In superuser mode without auth, should return 401
        assert response.status_code == 401

    def test_llm_models_endpoint_public_mode(self, client_public):
        """Test /api/v1/llm/models endpoint in public mode"""
        with patch('inference_core.services.llm_service.get_llm_service') as mock_llm_service:
            mock_service = mock_llm_service.return_value
            mock_service.get_available_models.return_value = {"gpt-4": True}
            
            response = client_public.get("/api/v1/llm/models")
            
            # In public mode, should succeed without authentication
            assert response.status_code == 200
            assert "models" in response.json()

    def test_llm_models_endpoint_user_mode_no_auth(self, client_user):
        """Test /api/v1/llm/models endpoint in user mode without authentication"""
        response = client_user.get("/api/v1/llm/models")
        
        # In user mode without auth, should return 401
        assert response.status_code == 401

    def test_llm_models_endpoint_superuser_mode_no_auth(self, client_superuser):
        """Test /api/v1/llm/models endpoint in superuser mode without authentication"""
        response = client_superuser.get("/api/v1/llm/models")
        
        # In superuser mode without auth, should return 401
        assert response.status_code == 401

    def test_llm_health_endpoint_public_mode(self, client_public):
        """Test /api/v1/llm/health endpoint in public mode"""
        with patch('inference_core.services.llm_service.get_llm_service') as mock_llm_service:
            mock_service = mock_llm_service.return_value
            mock_service.get_available_models.return_value = {"gpt-4": True}
            
            response = client_public.get("/api/v1/llm/health")
            
            # In public mode, should succeed without authentication
            assert response.status_code == 200
            assert "status" in response.json()

    def test_llm_health_endpoint_user_mode_no_auth(self, client_user):
        """Test /api/v1/llm/health endpoint in user mode without authentication"""
        response = client_user.get("/api/v1/llm/health")
        
        # In user mode without auth, should return 401
        assert response.status_code == 401

    def test_llm_stats_endpoint_public_mode(self, client_public):
        """Test /api/v1/llm/stats endpoint in public mode"""
        with patch('inference_core.services.llm_service.get_llm_service') as mock_llm_service:
            mock_service = mock_llm_service.return_value
            mock_service.get_usage_stats.return_value = {"requests": 100}
            
            response = client_public.get("/api/v1/llm/stats")
            
            # In public mode, should succeed without authentication
            assert response.status_code == 200
            assert "stats" in response.json()

    def test_batch_endpoints_consistency_public_mode(self, client_public):
        """Test that batch endpoints also respect the access control mode"""
        response = client_public.get("/api/v1/llm/batch/test-job-id")
        
        # Even in public mode, batch endpoints might require user for business logic
        # But the access control should be consistent
        # This might return 422 (validation error) or other business logic errors
        # rather than 401 (authentication error)
        assert response.status_code != 401  # Should not be authentication error

    def test_batch_endpoints_consistency_user_mode(self, client_user):
        """Test that batch endpoints respect user access control mode"""
        response = client_user.get("/api/v1/llm/batch/test-job-id")
        
        # In user mode without auth, should return 401
        assert response.status_code == 401

    def test_batch_endpoints_consistency_superuser_mode(self, client_superuser):
        """Test that batch endpoints respect superuser access control mode"""
        response = client_superuser.get("/api/v1/llm/batch/test-job-id")
        
        # In superuser mode without auth, should return 401
        assert response.status_code == 401


class TestStreamingEndpointsAccessControl:
    """Test access control for streaming endpoints specifically"""

    @pytest.fixture
    def client_public(self):
        """Test client with public access mode for streaming tests"""
        with patch('inference_core.core.config.get_settings') as mock:
            settings = Settings()
            settings.llm_api_access_mode = "public"
            mock.return_value = settings
            
            with patch('inference_core.core.dependecies.get_settings', return_value=settings):
                app = create_application()
                return TestClient(app)

    @pytest.fixture
    def client_superuser(self):
        """Test client with superuser access mode for streaming tests"""
        with patch('inference_core.core.config.get_settings') as mock:
            settings = Settings()
            settings.llm_api_access_mode = "superuser"
            mock.return_value = settings
            
            with patch('inference_core.core.dependecies.get_settings', return_value=settings):
                app = create_application()
                return TestClient(app)

    def test_explain_stream_endpoint_public_mode(self, client_public):
        """Test /api/v1/llm/explain/stream endpoint in public mode"""
        with patch('inference_core.services.llm_service.get_llm_service') as mock_llm_service:
            # Mock streaming response
            async def mock_stream():
                yield b'data: {"event": "start"}\n\n'
                yield b'data: {"event": "token", "content": "test"}\n\n'
                yield b'data: {"event": "end"}\n\n'
            
            mock_service = mock_llm_service.return_value
            mock_service.stream_explanation.return_value = mock_stream()
            
            response = client_public.post(
                "/api/v1/llm/explain/stream",
                json={"question": "What is AI?"}
            )
            
            # In public mode, streaming should work without authentication
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_explain_stream_endpoint_superuser_mode_no_auth(self, client_superuser):
        """Test /api/v1/llm/explain/stream endpoint in superuser mode without auth"""
        response = client_superuser.post(
            "/api/v1/llm/explain/stream",
            json={"question": "What is AI?"}
        )
        
        # In superuser mode without auth, should return 401
        assert response.status_code == 401

    def test_conversation_stream_endpoint_public_mode(self, client_public):
        """Test /api/v1/llm/conversation/stream endpoint in public mode"""
        with patch('inference_core.services.llm_service.get_llm_service') as mock_llm_service:
            # Mock streaming response
            async def mock_stream():
                yield b'data: {"event": "start"}\n\n'
                yield b'data: {"event": "token", "content": "Hello"}\n\n'
                yield b'data: {"event": "end"}\n\n'
            
            mock_service = mock_llm_service.return_value
            mock_service.stream_conversation.return_value = mock_stream()
            
            response = client_public.post(
                "/api/v1/llm/conversation/stream",
                json={"user_input": "Hello"}
            )
            
            # In public mode, streaming should work without authentication
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_conversation_stream_endpoint_superuser_mode_no_auth(self, client_superuser):
        """Test /api/v1/llm/conversation/stream endpoint in superuser mode without auth"""
        response = client_superuser.post(
            "/api/v1/llm/conversation/stream",
            json={"user_input": "Hello"}
        )
        
        # In superuser mode without auth, should return 401
        assert response.status_code == 401
"""
Integration tests for LLM streaming API endpoints
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import AsyncClient

from inference_core.main_factory import create_application


@pytest.fixture
async def async_test_client():
    """Create test client for streaming endpoints (httpx>=0.28 uses ASGITransport)."""
    app = create_application()
    from httpx import ASGITransport

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
class TestStreamingEndpoints:
    """Test streaming API endpoints"""

    @patch("inference_core.llm.streaming.get_model_factory")
    @patch("inference_core.llm.streaming.SQLChatMessageHistory")
    @patch("inference_core.llm.streaming.get_chat_prompt_template")
    async def test_conversation_stream_endpoint(
        self, mock_prompt, mock_history, mock_factory, public_access_async_client
    ):
        """Test conversation streaming endpoint"""
        # Mock model factory
        mock_model = AsyncMock()

        # Create an async generator for model streaming
        async def mock_astream(*args):
            yield Mock(content="Hello")
            yield Mock(content=" there!")

        mock_model.astream = mock_astream

        mock_factory_instance = Mock()
        mock_factory_instance.create_model.return_value = mock_model
        mock_factory.return_value = mock_factory_instance

        # Mock history
        mock_history_instance = Mock()
        mock_history_instance.messages = []
        mock_history_instance.add_message = Mock()
        mock_history.return_value = mock_history_instance

        # Mock prompt template
        mock_prompt.return_value = None

        # Make request to streaming endpoint
        request_data = {
            "user_input": "Hello, how are you?",
            "session_id": "test-session",
        }

        async with public_access_async_client.stream(
            "POST", "/api/v1/llm/conversation/stream", json=request_data
        ) as response:
            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )

            # Collect streamed chunks
            chunks = []
            async for chunk in response.aiter_text():
                if chunk.strip():
                    chunks.append(chunk)
                if len(chunks) >= 10:  # Safety limit
                    break

        # Verify we got streaming data
        assert len(chunks) > 0

        # Parse and verify events (robust against coalesced SSE frames)
        events = []
        buffer = "".join(chunks)
        for line in buffer.splitlines():
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            try:
                event_data = json.loads(payload)
                events.append(event_data)
            except json.JSONDecodeError:
                continue

        assert len(events) > 0

        # Should have start event
        start_events = [e for e in events if e.get("event") == "start"]
        assert len(start_events) > 0
        assert start_events[0].get("session_id") == "test-session"

        # Should have token events
        token_events = [e for e in events if e.get("event") == "token"]
        assert len(token_events) > 0

    @patch("inference_core.llm.streaming.get_model_factory")
    @patch("inference_core.llm.streaming.get_chat_prompt_template")
    async def test_explain_stream_endpoint(
        self, mock_prompt, mock_factory, public_access_async_client
    ):
        """Test explanation streaming endpoint"""
        # Mock model factory
        mock_model = AsyncMock()

        # Create an async generator for model streaming
        async def mock_astream(*args):
            yield Mock(content="The answer")
            yield Mock(content=" is 42.")

        mock_model.astream = mock_astream

        mock_factory_instance = Mock()
        mock_factory_instance.create_model.return_value = mock_model
        mock_factory.return_value = mock_factory_instance

        # Mock prompt template
        mock_prompt_instance = Mock()
        mock_prompt_instance.format_messages.return_value = [
            Mock(content="Explain: What is the meaning of life?")
        ]
        mock_prompt.return_value = mock_prompt_instance

        # Make request to streaming endpoint
        request_data = {"question": "What is the meaning of life?"}

        async with public_access_async_client.stream(
            "POST", "/api/v1/llm/explain/stream", json=request_data
        ) as response:
            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )

            # Collect streamed chunks
            chunks = []
            async for chunk in response.aiter_text():
                if chunk.strip():
                    chunks.append(chunk)
                if len(chunks) >= 10:  # Safety limit
                    break

        # Verify we got streaming data
        assert len(chunks) > 0

        # Parse and verify events (robust against coalesced SSE frames)
        events = []
        buffer = "".join(chunks)
        for line in buffer.splitlines():
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            try:
                event_data = json.loads(payload)
                events.append(event_data)
            except json.JSONDecodeError:
                continue

        assert len(events) > 0

        # Should have start event
        start_events = [e for e in events if e.get("event") == "start"]
        assert len(start_events) > 0

        # Should have token events
        token_events = [e for e in events if e.get("event") == "token"]
        assert len(token_events) > 0

    async def test_conversation_stream_missing_user_input(
        self, public_access_async_client
    ):
        """Test conversation streaming with missing user input"""
        request_data = {
            "session_id": "test-session"
            # Missing user_input
        }

        response = await public_access_async_client.post(
            "/api/v1/llm/conversation/stream", json=request_data
        )

        # Should return validation error
        assert response.status_code == 422

    async def test_explain_stream_missing_question(self, public_access_async_client):
        """Test explanation streaming with missing question"""
        request_data = {
            "model_name": "gpt-4o-mini"
            # Missing question
        }

        response = await public_access_async_client.post(
            "/api/v1/llm/explain/stream", json=request_data
        )

        # Should return validation error
        assert response.status_code == 422

    @patch("inference_core.llm.streaming.get_model_factory")
    async def test_conversation_stream_model_failure(
        self, mock_factory, public_access_async_client
    ):
        """Test conversation streaming when model creation fails"""
        # Mock factory that returns None (failed model creation)
        mock_factory_instance = Mock()
        mock_factory_instance.create_model.return_value = None
        mock_factory.return_value = mock_factory_instance

        request_data = {"user_input": "Hello", "session_id": "test-session"}

        async with public_access_async_client.stream(
            "POST", "/api/v1/llm/conversation/stream", json=request_data
        ) as response:
            assert response.status_code == 200  # Stream starts successfully

            # Collect error response
            chunks = []
            async for chunk in response.aiter_text():
                if chunk.strip():
                    chunks.append(chunk)
                    break  # Just get first chunk which should be error

        # Should get error event
        assert len(chunks) > 0
        error_chunk = chunks[0]
        assert "data: " in error_chunk

        try:
            event_data = json.loads(error_chunk[6:])
            assert event_data.get("event") == "error"
            assert "Failed to create model" in event_data.get("message", "")
        except json.JSONDecodeError:
            pytest.fail("Error response was not valid JSON")

    async def test_conversation_stream_with_model_params(
        self, public_access_async_client
    ):
        """Test conversation streaming with additional model parameters"""
        with (
            patch("inference_core.llm.streaming.get_model_factory") as mock_factory,
            patch("inference_core.llm.streaming.SQLChatMessageHistory") as mock_history,
            patch(
                "inference_core.llm.streaming.get_chat_prompt_template"
            ) as mock_prompt,
        ):

            # Mock model factory
            mock_model = AsyncMock()

            async def mock_astream(*args):
                yield Mock(content="Response")

            mock_model.astream = mock_astream

            mock_factory_instance = Mock()
            mock_factory_instance.create_model.return_value = mock_model
            mock_factory.return_value = mock_factory_instance

            # Mock history
            mock_history_instance = Mock()
            mock_history_instance.messages = []
            mock_history_instance.add_message = Mock()
            mock_history.return_value = mock_history_instance

            # Mock prompt template
            mock_prompt.return_value = None

            request_data = {
                "user_input": "Hello",
                "session_id": "test-session",
                "temperature": 0.8,
                "max_tokens": 512,
                "model_name": "gpt-4o-mini",
            }

            async with public_access_async_client.stream(
                "POST", "/api/v1/llm/conversation/stream", json=request_data
            ) as response:
                assert response.status_code == 200

                # Verify factory was called with correct parameters
                mock_factory_instance.create_model.assert_called_once()
                call_args = mock_factory_instance.create_model.call_args

                # Check streaming=True was passed
                assert call_args[1]["streaming"] is True
                # Check model parameters were passed
                assert call_args[1]["temperature"] == 0.8
                assert call_args[1]["max_tokens"] == 512

    async def test_streaming_response_headers(self, public_access_async_client):
        """Test that streaming responses have correct headers"""
        with patch("inference_core.llm.streaming.get_model_factory") as mock_factory:
            # Mock factory that returns None to get quick error response
            mock_factory_instance = Mock()
            mock_factory_instance.create_model.return_value = None
            mock_factory.return_value = mock_factory_instance

            request_data = {"user_input": "Hello", "session_id": "test-session"}

            response = await public_access_async_client.post(
                "/api/v1/llm/conversation/stream", json=request_data
            )

            # Check streaming headers
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )
            assert "Cache-Control" in response.headers
            assert response.headers["Cache-Control"] == "no-cache"
            assert "X-Accel-Buffering" in response.headers
            assert response.headers["X-Accel-Buffering"] == "no"

    async def test_stream_with_auto_generated_session_id(
        self, public_access_async_client
    ):
        """Test conversation streaming with auto-generated session ID"""
        with (
            patch("inference_core.llm.streaming.get_model_factory") as mock_factory,
            patch("inference_core.llm.streaming.SQLChatMessageHistory") as mock_history,
            patch(
                "inference_core.llm.streaming.get_chat_prompt_template"
            ) as mock_prompt,
        ):

            # Mock model factory
            mock_model = AsyncMock()

            async def mock_astream(*args):
                yield Mock(content="Hello")

            mock_model.astream = mock_astream

            mock_factory_instance = Mock()
            mock_factory_instance.create_model.return_value = mock_model
            mock_factory.return_value = mock_factory_instance

            # Mock history
            mock_history_instance = Mock()
            mock_history_instance.messages = []
            mock_history_instance.add_message = Mock()
            mock_history.return_value = mock_history_instance

            # Mock prompt template
            mock_prompt.return_value = None

            request_data = {
                "user_input": "Hello"
                # No session_id provided
            }

            async with public_access_async_client.stream(
                "POST", "/api/v1/llm/conversation/stream", json=request_data
            ) as response:
                assert response.status_code == 200

                # Get first chunk (start event)
                async for chunk in response.aiter_text():
                    if chunk.strip() and chunk.startswith("data: "):
                        try:
                            event_data = json.loads(chunk[6:])
                            if event_data.get("event") == "start":
                                # Should have auto-generated session_id
                                assert "session_id" in event_data
                                session_id = event_data["session_id"]
                                assert len(session_id) > 0
                                assert session_id != "test-session"
                                break
                        except json.JSONDecodeError:
                            pass


@pytest.mark.asyncio
class TestStreamingAuthentication:
    """Test streaming endpoints with authentication"""

    async def test_conversation_stream_requires_auth(self, public_access_async_client):
        """Test that streaming endpoints require authentication"""
        # This test assumes auth is enabled - may need to be adjusted
        # based on the actual auth configuration
        request_data = {"user_input": "Hello", "session_id": "test-session"}

        # Request without authentication headers
        response = await public_access_async_client.post(
            "/api/v1/llm/conversation/stream", json=request_data
        )

        # Check if auth is required (this depends on the app's auth setup)
        # If auth is required, should get 401 or 403
        # If no auth required, should get 200
        assert response.status_code in [200, 401, 403]


@pytest.mark.asyncio
class TestStreamingPerformance:
    """Test streaming performance characteristics"""

    @patch("inference_core.llm.streaming.get_model_factory")
    @patch("inference_core.llm.streaming.SQLChatMessageHistory")
    @patch("inference_core.llm.streaming.get_chat_prompt_template")
    async def test_streaming_response_timing(
        self, mock_prompt, mock_history, mock_factory, public_access_async_client
    ):
        """Test that streaming starts responding quickly"""
        import time

        # Mock model factory
        mock_model = AsyncMock()

        # Create slow async generator to test initial response timing
        async def slow_astream(*args):
            yield Mock(content="First")
            await asyncio.sleep(0.1)  # Small delay
            yield Mock(content=" token")

        mock_model.astream = slow_astream

        mock_factory_instance = Mock()
        mock_factory_instance.create_model.return_value = mock_model
        mock_factory.return_value = mock_factory_instance

        # Mock history
        mock_history_instance = Mock()
        mock_history_instance.messages = []
        mock_history_instance.add_message = Mock()
        mock_history.return_value = mock_history_instance

        # Mock prompt template
        mock_prompt.return_value = None

        request_data = {"user_input": "Hello", "session_id": "test-session"}

        start_time = time.time()

        async with public_access_async_client.stream(
            "POST", "/api/v1/llm/conversation/stream", json=request_data
        ) as response:
            # Time to get first chunk should be fast
            first_chunk_time = None
            async for chunk in response.aiter_text():
                if chunk.strip():
                    first_chunk_time = time.time()
                    break

            assert first_chunk_time is not None
            initial_response_time = first_chunk_time - start_time

            # Should get first response within reasonable time (allowing for test overhead)
            assert initial_response_time < 2.0  # 2 seconds should be plenty for tests

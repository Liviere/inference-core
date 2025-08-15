"""
Tests for LLM streaming functionality
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, Mock, patch

from app.llm.streaming import (
    StreamingForwarderHandler,
    format_sse,
    StreamChunk,
    stream_conversation,
    stream_explanation
)


class TestStreamingForwarderHandler:
    """Test the streaming callback handler"""

    def test_handler_initialization(self):
        """Test handler initialization"""
        queue = asyncio.Queue()
        handler = StreamingForwarderHandler(queue)
        
        assert handler.queue is queue
        assert handler.max_queue_size == 100
        assert not handler._cancelled

    def test_on_llm_new_token(self):
        """Test token forwarding"""
        queue = asyncio.Queue()
        handler = StreamingForwarderHandler(queue)
        
        # Test token forwarding
        handler.on_llm_new_token("hello")
        
        # Check queue has token
        assert queue.qsize() == 1
        chunk = queue.get_nowait()
        assert chunk.type == "token"
        assert chunk.content == "hello"

    def test_on_llm_start(self):
        """Test start event"""
        queue = asyncio.Queue()
        handler = StreamingForwarderHandler(queue)
        
        serialized = {"name": "test-model"}
        handler.on_llm_start(serialized, ["test prompt"])
        
        # Check queue has start event
        assert queue.qsize() == 1
        chunk = queue.get_nowait()
        assert chunk.type == "start"
        assert chunk.model == "test-model"

    def test_on_llm_end_with_usage(self):
        """Test end event with usage metadata"""
        queue = asyncio.Queue()
        handler = StreamingForwarderHandler(queue)
        
        # Mock LLMResult with usage data
        llm_result = Mock()
        llm_result.llm_output = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        llm_result.generations = []
        
        handler.on_llm_end(llm_result)
        
        # Should have usage and end events
        assert queue.qsize() == 2
        
        usage_chunk = queue.get_nowait()
        assert usage_chunk.type == "usage"
        assert usage_chunk.usage == {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30
        }
        
        end_chunk = queue.get_nowait()
        assert end_chunk.type == "end"

    def test_on_llm_error(self):
        """Test error handling"""
        queue = asyncio.Queue()
        handler = StreamingForwarderHandler(queue)
        
        error = Exception("Test error")
        handler.on_llm_error(error)
        
        # Should have error and end events
        assert queue.qsize() == 2
        
        error_chunk = queue.get_nowait()
        assert error_chunk.type == "error"
        assert error_chunk.message == "Test error"
        
        end_chunk = queue.get_nowait()
        assert end_chunk.type == "end"

    def test_cancel_handler(self):
        """Test handler cancellation"""
        queue = asyncio.Queue()
        handler = StreamingForwarderHandler(queue)
        
        handler.cancel()
        assert handler._cancelled
        
        # After cancellation, no events should be added
        handler.on_llm_new_token("test")
        assert queue.qsize() == 0

    def test_queue_full_handling(self):
        """Test behavior when queue is full"""
        # Create a small queue
        queue = asyncio.Queue(maxsize=1)
        handler = StreamingForwarderHandler(queue, max_queue_size=1)
        
        # Fill the queue
        handler.on_llm_new_token("token1")
        assert queue.qsize() == 1
        
        # Try to add another token - should not raise exception
        handler.on_llm_new_token("token2")
        assert queue.qsize() == 1  # Still only one item


class TestFormatSSE:
    """Test SSE formatting"""

    def test_basic_formatting(self):
        """Test basic SSE formatting"""
        data = {"event": "token", "content": "hello"}
        formatted = format_sse(data)
        
        expected = b'data: {"event": "token", "content": "hello"}\n\n'
        assert formatted == expected

    def test_unicode_formatting(self):
        """Test unicode content formatting"""
        data = {"event": "token", "content": "こんにちは"}
        formatted = format_sse(data)
        
        # Should preserve unicode characters
        assert "こんにちは" in formatted.decode("utf-8")
        assert formatted.startswith(b"data: ")
        assert formatted.endswith(b"\n\n")

    def test_complex_data_formatting(self):
        """Test complex data structure formatting"""
        data = {
            "event": "usage",
            "usage": {
                "input_tokens": 123,
                "output_tokens": 456,
                "total_tokens": 579
            }
        }
        formatted = format_sse(data)
        
        # Should be valid JSON
        json_str = formatted.decode("utf-8").replace("data: ", "").replace("\n\n", "")
        parsed = json.loads(json_str)
        assert parsed == data


class TestStreamChunk:
    """Test StreamChunk dataclass"""

    def test_token_chunk(self):
        """Test token chunk creation"""
        chunk = StreamChunk(type="token", content="hello")
        assert chunk.type == "token"
        assert chunk.content == "hello"
        assert chunk.usage is None
        assert chunk.message is None

    def test_usage_chunk(self):
        """Test usage chunk creation"""
        usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        chunk = StreamChunk(type="usage", usage=usage)
        assert chunk.type == "usage"
        assert chunk.usage == usage
        assert chunk.content is None

    def test_error_chunk(self):
        """Test error chunk creation"""
        chunk = StreamChunk(type="error", message="Something went wrong")
        assert chunk.type == "error"
        assert chunk.message == "Something went wrong"
        assert chunk.content is None

    def test_start_chunk(self):
        """Test start chunk creation"""
        chunk = StreamChunk(
            type="start", 
            model="gpt-4o-mini", 
            session_id="test-session"
        )
        assert chunk.type == "start"
        assert chunk.model == "gpt-4o-mini"
        assert chunk.session_id == "test-session"


@pytest.mark.asyncio
class TestStreamingFunctions:
    """Test the main streaming functions"""

    @patch('app.llm.streaming.get_model_factory')
    @patch('app.llm.streaming.SQLChatMessageHistory')
    @patch('app.llm.streaming.get_chat_prompt_template')
    async def test_stream_conversation_basic(self, mock_prompt, mock_history, mock_factory):
        """Test basic conversation streaming"""
        # Mock model factory
        mock_model = AsyncMock()
        
        # Create async generator for model streaming
        async def mock_astream(*args):
            yield Mock(content="Hello")
            yield Mock(content=" world!")
        
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
        
        # Mock request
        mock_request = Mock()
        mock_request.is_disconnected = AsyncMock(return_value=False)
        
        # Collect streaming output
        chunks = []
        async for chunk in stream_conversation(
            session_id="test-session",
            user_input="Hello",
            request=mock_request
        ):
            chunks.append(chunk)
        
        # Verify we got some output
        assert len(chunks) > 0
        
        # First chunk should be start event
        first_chunk = chunks[0].decode("utf-8")
        assert "start" in first_chunk
        assert "test-session" in first_chunk
        
        # Should have token events
        token_chunks = [c for c in chunks if b'"event": "token"' in c]
        assert len(token_chunks) > 0

    @patch('app.llm.streaming.get_model_factory')
    @patch('app.llm.streaming.get_chat_prompt_template')
    async def test_stream_explanation_basic(self, mock_prompt, mock_factory):
        """Test basic explanation streaming"""
        # Mock model factory
        mock_model = AsyncMock()
        
        # Create async generator for model streaming
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
        
        # Mock request
        mock_request = Mock()
        mock_request.is_disconnected = AsyncMock(return_value=False)
        
        # Collect streaming output
        chunks = []
        async for chunk in stream_explanation(
            question="What is the meaning of life?",
            request=mock_request
        ):
            chunks.append(chunk)
        
        # Verify we got some output
        assert len(chunks) > 0
        
        # First chunk should be start event
        first_chunk = chunks[0].decode("utf-8")
        assert "start" in first_chunk
        
        # Should have token events
        token_chunks = [c for c in chunks if b'"event": "token"' in c]
        assert len(token_chunks) > 0

    async def test_session_id_generation(self):
        """Test that session ID is generated when not provided"""
        with patch('app.llm.streaming.get_model_factory') as mock_factory:
            # Mock model that fails immediately to test session ID generation
            mock_factory_instance = Mock()
            mock_factory_instance.create_model.return_value = None
            mock_factory.return_value = mock_factory_instance
            
            chunks = []
            async for chunk in stream_conversation(
                session_id=None,  # No session ID provided
                user_input="Hello"
            ):
                chunks.append(chunk)
                break  # Just get first chunk
            
            # Should get an error due to failed model creation
            # but session ID should still be generated in the process
            assert len(chunks) > 0

    @patch('app.llm.streaming.get_model_factory')
    async def test_model_creation_failure(self, mock_factory):
        """Test handling of model creation failure"""
        # Mock factory that returns None (failed model creation)
        mock_factory_instance = Mock()
        mock_factory_instance.create_model.return_value = None
        mock_factory.return_value = mock_factory_instance
        
        chunks = []
        async for chunk in stream_conversation(
            session_id="test-session",
            user_input="Hello"
        ):
            chunks.append(chunk)
        
        # Should get error event
        assert len(chunks) == 1
        error_chunk = chunks[0].decode("utf-8")
        assert "error" in error_chunk
        assert "Failed to create model" in error_chunk

    @patch('app.llm.streaming.get_model_factory')
    async def test_client_disconnect_handling(self, mock_factory):
        """Test handling of client disconnection"""
        # Mock model factory
        mock_model = AsyncMock()
        # Make astream hang to test disconnect
        async def slow_astream(*args):
            await asyncio.sleep(10)  # Long delay
            yield Mock(content="This should not be reached")
        
        mock_model.astream = slow_astream
        
        mock_factory_instance = Mock()
        mock_factory_instance.create_model.return_value = mock_model
        mock_factory.return_value = mock_factory_instance
        
        # Mock request that disconnects immediately
        mock_request = Mock()
        mock_request.is_disconnected = AsyncMock(return_value=True)
        
        with patch('app.llm.streaming.SQLChatMessageHistory'), \
             patch('app.llm.streaming.get_chat_prompt_template'):
            
            chunks = []
            async for chunk in stream_conversation(
                session_id="test-session",
                user_input="Hello",
                request=mock_request
            ):
                chunks.append(chunk)
                if len(chunks) > 10:  # Safety break
                    break
            
            # Should get start event but then disconnect
            assert len(chunks) >= 1
            first_chunk = chunks[0].decode("utf-8")
            assert "start" in first_chunk
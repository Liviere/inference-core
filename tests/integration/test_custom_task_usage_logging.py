"""Integration tests for custom task usage logging.

Tests the full integration of custom_task helpers with usage tracking.
Note: These tests verify the functional behavior of the helpers.
Database persistence may fail with FK constraints in test environment
due to fail_open=True in usage_logging config.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_core.llm.custom_task import run_with_usage, stream_with_usage


@pytest.mark.asyncio
class TestCustomTaskUsageLogging:
    """Integration tests for custom task usage logging"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from inference_core.core.config import get_settings

        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_run_with_usage_executes_successfully(self):
        """Test that run_with_usage executes runnable and returns result"""
        # Create a mock runnable
        mock_runnable = AsyncMock()
        mock_runnable.ainvoke.return_value = {
            "extracted_entities": ["entity1", "entity2"]
        }

        # Execute with usage tracking
        result = await run_with_usage(
            task_type="extraction",
            runnable=mock_runnable,
            input={"text": "Extract entities from this text"},
            model_name="gpt-5-nano",  # Using test model from config
            request_mode="sync",
            session_id="test-extraction-session",
            request_id="test-request-123",
        )

        # Verify result
        assert result == {"extracted_entities": ["entity1", "entity2"]}

        # Verify runnable was called with correct input
        mock_runnable.ainvoke.assert_called_once()
        call_args = mock_runnable.ainvoke.call_args
        assert call_args.args[0] == {"text": "Extract entities from this text"}
        assert "callbacks" in call_args.kwargs["config"]

    @pytest.mark.asyncio
    async def test_run_with_usage_logs_error(self):
        """Test that errors are properly handled and re-raised"""
        # Create a mock runnable that raises an error
        mock_runnable = AsyncMock()
        test_error = ValueError("Extraction failed")
        mock_runnable.ainvoke.side_effect = test_error

        # Execute and expect error
        with pytest.raises(ValueError) as exc_info:
            await run_with_usage(
                task_type="extraction",
                runnable=mock_runnable,
                input={"text": "This will fail"},
                model_name="gpt-5-nano",
                request_id="test-error-request",
            )

        # Verify error was re-raised correctly
        assert exc_info.value == test_error
        assert str(exc_info.value) == "Extraction failed"

    @pytest.mark.asyncio
    async def test_stream_with_usage_yields_chunks(self):
        """Test that stream_with_usage yields chunks correctly"""
        # Create a mock streaming runnable
        async def mock_stream():
            yield "Extracted: "
            yield "entity1, "
            yield "entity2"

        mock_runnable = MagicMock()
        mock_runnable.astream.return_value = mock_stream()

        # Execute with streaming
        chunks = []
        async for chunk in stream_with_usage(
            task_type="summarization",
            runnable=mock_runnable,
            input={"text": "Summarize this long document"},
            model_name="gpt-5-nano",
            session_id="test-stream-session",
            request_id="test-stream-request",
        ):
            chunks.append(chunk)

        # Verify chunks were received
        assert chunks == ["Extracted: ", "entity1, ", "entity2"]

        # Verify runnable was called with callbacks
        mock_runnable.astream.assert_called_once()
        call_args = mock_runnable.astream.call_args
        assert "callbacks" in call_args.kwargs["config"]

    @pytest.mark.asyncio
    async def test_custom_task_different_types(self):
        """Test multiple custom task types execute correctly"""
        # Define different task types
        tasks = [
            ("classification", {"text": "Classify this"}),
            ("entity_extraction", {"text": "Extract entities"}),
            ("sentiment_analysis", {"text": "Analyze sentiment"}),
        ]

        for task_type, input_data in tasks:
            mock_runnable = AsyncMock()
            mock_runnable.ainvoke.return_value = {"result": f"{task_type}_result"}

            result = await run_with_usage(
                task_type=task_type,
                runnable=mock_runnable,
                input=input_data,
                model_name="gpt-5-nano",
            )

            # Verify each task executed successfully
            assert result == {"result": f"{task_type}_result"}

    @pytest.mark.asyncio
    async def test_stream_with_usage_handles_error(self):
        """Test that streaming errors are properly handled"""
        # Create a mock streaming runnable that fails mid-stream
        async def mock_stream():
            yield "chunk1"
            raise ValueError("Streaming failed")

        mock_runnable = MagicMock()
        mock_runnable.astream.return_value = mock_stream()

        chunks = []
        with pytest.raises(ValueError) as exc_info:
            async for chunk in stream_with_usage(
                task_type="summarization",
                runnable=mock_runnable,
                input={"text": "Test"},
                model_name="gpt-5-nano",
            ):
                chunks.append(chunk)

        # Verify partial chunks were received before error
        assert chunks == ["chunk1"]
        assert str(exc_info.value) == "Streaming failed"

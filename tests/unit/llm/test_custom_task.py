"""Unit tests for inference_core.llm.custom_task module.

Tests the generic usage/cost logging abstraction for custom LLM tasks.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_core.llm.custom_task import run_with_usage, stream_with_usage


class TestRunWithUsage:
    """Test run_with_usage function for sync execution with usage logging"""

    @pytest.mark.asyncio
    async def test_run_with_usage_success(self):
        """Test successful execution with usage logging"""
        # Mock configuration
        mock_config = MagicMock()
        mock_model_cfg = MagicMock()
        mock_model_cfg.provider = "openai"
        mock_model_cfg.pricing = MagicMock()
        mock_config.models = {"test-model": mock_model_cfg}
        mock_config.usage_logging = MagicMock(enabled=True)

        # Mock usage logger and session
        mock_usage_logger = MagicMock()
        mock_session = MagicMock()
        mock_session.accumulated_usage = {"input_tokens": 10, "output_tokens": 20}
        mock_session.finalize = AsyncMock()
        mock_usage_logger.start_session.return_value = mock_session

        # Mock runnable
        mock_runnable = AsyncMock()
        mock_runnable.ainvoke.return_value = {"result": "test extraction"}

        with (
            patch(
                "inference_core.llm.custom_task.get_llm_config", return_value=mock_config
            ),
            patch(
                "inference_core.llm.custom_task.UsageLogger",
                return_value=mock_usage_logger,
            ),
            patch("inference_core.llm.custom_task.LLMUsageCallbackHandler") as mock_cb,
        ):
            result = await run_with_usage(
                task_type="extraction",
                runnable=mock_runnable,
                input={"text": "Extract entities"},
                model_name="test-model",
                request_mode="sync",
                session_id="test-session",
                user_id="12345678-1234-5678-1234-567812345678",
                request_id="test-request",
            )

            # Verify result
            assert result == {"result": "test extraction"}

            # Verify usage logger was started
            mock_usage_logger.start_session.assert_called_once()
            call_args = mock_usage_logger.start_session.call_args
            assert call_args.kwargs["task_type"] == "extraction"
            assert call_args.kwargs["request_mode"] == "sync"
            assert call_args.kwargs["model_name"] == "test-model"
            assert call_args.kwargs["provider"] == "openai"
            assert call_args.kwargs["session_id"] == "test-session"
            assert call_args.kwargs["request_id"] == "test-request"
            assert isinstance(call_args.kwargs["user_id"], uuid.UUID)

            # Verify callback was created
            mock_cb.assert_called_once()

            # Verify runnable was called with callbacks
            mock_runnable.ainvoke.assert_called_once()
            call_args = mock_runnable.ainvoke.call_args
            assert call_args.args[0] == {"text": "Extract entities"}
            assert "callbacks" in call_args.kwargs["config"]

            # Verify session was finalized with success
            mock_session.finalize.assert_called_once()
            finalize_args = mock_session.finalize.call_args
            assert finalize_args.kwargs["success"] is True
            assert finalize_args.kwargs["streamed"] is False
            assert finalize_args.kwargs["partial"] is False

    @pytest.mark.asyncio
    async def test_run_with_usage_error_handling(self):
        """Test error handling and session finalization on failure"""
        # Mock configuration
        mock_config = MagicMock()
        mock_model_cfg = MagicMock()
        mock_model_cfg.provider = "openai"
        mock_model_cfg.pricing = None  # Test without pricing
        mock_config.models = {"test-model": mock_model_cfg}
        mock_config.usage_logging = MagicMock(enabled=True)

        # Mock usage logger and session
        mock_usage_logger = MagicMock()
        mock_session = MagicMock()
        mock_session.accumulated_usage = {}
        mock_session.finalize = AsyncMock()
        mock_usage_logger.start_session.return_value = mock_session

        # Mock runnable that raises error
        mock_runnable = AsyncMock()
        test_error = ValueError("Test error")
        mock_runnable.ainvoke.side_effect = test_error

        with (
            patch(
                "inference_core.llm.custom_task.get_llm_config", return_value=mock_config
            ),
            patch(
                "inference_core.llm.custom_task.UsageLogger",
                return_value=mock_usage_logger,
            ),
            patch("inference_core.llm.custom_task.LLMUsageCallbackHandler"),
        ):
            with pytest.raises(ValueError) as exc_info:
                await run_with_usage(
                    task_type="extraction",
                    runnable=mock_runnable,
                    input={"text": "Extract entities"},
                    model_name="test-model",
                )

            # Verify error was raised
            assert exc_info.value == test_error

            # Verify session was finalized with error
            mock_session.finalize.assert_called_once()
            finalize_args = mock_session.finalize.call_args
            assert finalize_args.kwargs["success"] is False
            assert finalize_args.kwargs["error"] == test_error
            assert finalize_args.kwargs["streamed"] is False

    @pytest.mark.asyncio
    async def test_run_with_usage_no_pricing_config(self):
        """Test execution when model has no pricing configuration"""
        # Mock configuration
        mock_config = MagicMock()
        mock_model_cfg = MagicMock()
        mock_model_cfg.provider = "custom"
        mock_model_cfg.pricing = None  # No pricing
        mock_config.models = {"test-model": mock_model_cfg}
        mock_config.usage_logging = MagicMock(enabled=True)

        # Mock usage logger and session
        mock_usage_logger = MagicMock()
        mock_session = MagicMock()
        mock_session.accumulated_usage = {}
        mock_session.finalize = AsyncMock()
        mock_usage_logger.start_session.return_value = mock_session

        # Mock runnable
        mock_runnable = AsyncMock()
        mock_runnable.ainvoke.return_value = {"result": "success"}

        with (
            patch(
                "inference_core.llm.custom_task.get_llm_config", return_value=mock_config
            ),
            patch(
                "inference_core.llm.custom_task.UsageLogger",
                return_value=mock_usage_logger,
            ),
            patch("inference_core.llm.custom_task.LLMUsageCallbackHandler") as mock_cb,
        ):
            result = await run_with_usage(
                task_type="summarization",
                runnable=mock_runnable,
                input={"text": "Summarize this"},
                model_name="test-model",
            )

            # Verify execution succeeded
            assert result == {"result": "success"}

            # Verify callback was created with None pricing
            mock_cb.assert_called_once()
            call_args = mock_cb.call_args
            assert call_args.kwargs["pricing_config"] is None

    @pytest.mark.asyncio
    async def test_run_with_usage_unknown_model(self):
        """Test execution when model is not in config"""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.models = {}  # Empty models dict
        mock_config.usage_logging = MagicMock(enabled=True)

        # Mock usage logger and session
        mock_usage_logger = MagicMock()
        mock_session = MagicMock()
        mock_session.accumulated_usage = {}
        mock_session.finalize = AsyncMock()
        mock_usage_logger.start_session.return_value = mock_session

        # Mock runnable
        mock_runnable = AsyncMock()
        mock_runnable.ainvoke.return_value = {"result": "success"}

        with (
            patch(
                "inference_core.llm.custom_task.get_llm_config", return_value=mock_config
            ),
            patch(
                "inference_core.llm.custom_task.UsageLogger",
                return_value=mock_usage_logger,
            ),
            patch("inference_core.llm.custom_task.LLMUsageCallbackHandler"),
        ):
            result = await run_with_usage(
                task_type="classification",
                runnable=mock_runnable,
                input={"text": "Classify this"},
                model_name="unknown-model",
            )

            # Verify execution succeeded with "unknown" provider
            assert result == {"result": "success"}

            # Verify session was started with "unknown" provider
            call_args = mock_usage_logger.start_session.call_args
            assert call_args.kwargs["provider"] == "unknown"

    @pytest.mark.asyncio
    async def test_run_with_usage_extra_callbacks(self):
        """Test that extra callbacks are included"""
        # Mock configuration
        mock_config = MagicMock()
        mock_model_cfg = MagicMock()
        mock_model_cfg.provider = "openai"
        mock_model_cfg.pricing = None
        mock_config.models = {"test-model": mock_model_cfg}
        mock_config.usage_logging = MagicMock(enabled=True)

        # Mock usage logger and session
        mock_usage_logger = MagicMock()
        mock_session = MagicMock()
        mock_session.accumulated_usage = {}
        mock_session.finalize = AsyncMock()
        mock_usage_logger.start_session.return_value = mock_session

        # Mock runnable
        mock_runnable = AsyncMock()
        mock_runnable.ainvoke.return_value = {"result": "success"}

        # Create extra callbacks
        extra_cb1 = MagicMock()
        extra_cb2 = MagicMock()

        with (
            patch(
                "inference_core.llm.custom_task.get_llm_config", return_value=mock_config
            ),
            patch(
                "inference_core.llm.custom_task.UsageLogger",
                return_value=mock_usage_logger,
            ),
            patch("inference_core.llm.custom_task.LLMUsageCallbackHandler") as mock_cb,
        ):
            await run_with_usage(
                task_type="extraction",
                runnable=mock_runnable,
                input={"text": "Test"},
                model_name="test-model",
                extra_callbacks=[extra_cb1, extra_cb2],
            )

            # Verify runnable was called with all callbacks
            call_args = mock_runnable.ainvoke.call_args
            callbacks = call_args.kwargs["config"]["callbacks"]
            assert len(callbacks) == 3  # usage callback + 2 extra
            assert callbacks[1] == extra_cb1
            assert callbacks[2] == extra_cb2

    @pytest.mark.asyncio
    async def test_run_with_usage_logging_disabled(self):
        """Test that usage callback is not added when logging is disabled"""
        # Mock configuration with logging disabled
        mock_config = MagicMock()
        mock_model_cfg = MagicMock()
        mock_model_cfg.provider = "openai"
        mock_model_cfg.pricing = None
        mock_config.models = {"test-model": mock_model_cfg}
        mock_config.usage_logging = MagicMock(enabled=False)

        # Mock usage logger and session
        mock_usage_logger = MagicMock()
        mock_session = MagicMock()
        mock_session.accumulated_usage = {}
        mock_session.finalize = AsyncMock()
        mock_usage_logger.start_session.return_value = mock_session

        # Mock runnable
        mock_runnable = AsyncMock()
        mock_runnable.ainvoke.return_value = {"result": "success"}

        with (
            patch(
                "inference_core.llm.custom_task.get_llm_config", return_value=mock_config
            ),
            patch(
                "inference_core.llm.custom_task.UsageLogger",
                return_value=mock_usage_logger,
            ),
            patch("inference_core.llm.custom_task.LLMUsageCallbackHandler") as mock_cb,
        ):
            await run_with_usage(
                task_type="extraction",
                runnable=mock_runnable,
                input={"text": "Test"},
                model_name="test-model",
            )

            # Verify usage callback was NOT created
            mock_cb.assert_not_called()

            # Verify runnable was called with empty callbacks list
            call_args = mock_runnable.ainvoke.call_args
            callbacks = call_args.kwargs["config"]["callbacks"]
            assert len(callbacks) == 0

    @pytest.mark.asyncio
    async def test_run_with_usage_invalid_user_id(self):
        """Test handling of invalid user_id format"""
        # Mock configuration
        mock_config = MagicMock()
        mock_model_cfg = MagicMock()
        mock_model_cfg.provider = "openai"
        mock_model_cfg.pricing = None
        mock_config.models = {"test-model": mock_model_cfg}
        mock_config.usage_logging = MagicMock(enabled=False)

        # Mock usage logger and session
        mock_usage_logger = MagicMock()
        mock_session = MagicMock()
        mock_session.accumulated_usage = {}
        mock_session.finalize = AsyncMock()
        mock_usage_logger.start_session.return_value = mock_session

        # Mock runnable
        mock_runnable = AsyncMock()
        mock_runnable.ainvoke.return_value = {"result": "success"}

        with (
            patch(
                "inference_core.llm.custom_task.get_llm_config", return_value=mock_config
            ),
            patch(
                "inference_core.llm.custom_task.UsageLogger",
                return_value=mock_usage_logger,
            ),
        ):
            await run_with_usage(
                task_type="extraction",
                runnable=mock_runnable,
                input={"text": "Test"},
                model_name="test-model",
                user_id="invalid-uuid-format",  # Invalid UUID
            )

            # Verify session was started with None user_id
            call_args = mock_usage_logger.start_session.call_args
            assert call_args.kwargs["user_id"] is None


class TestStreamWithUsage:
    """Test stream_with_usage function for streaming execution with usage logging"""

    @pytest.mark.asyncio
    async def test_stream_with_usage_success(self):
        """Test successful streaming execution with usage logging"""
        # Mock configuration
        mock_config = MagicMock()
        mock_model_cfg = MagicMock()
        mock_model_cfg.provider = "openai"
        mock_model_cfg.pricing = MagicMock()
        mock_config.models = {"test-model": mock_model_cfg}
        mock_config.usage_logging = MagicMock(enabled=True)

        # Mock usage logger and session
        mock_usage_logger = MagicMock()
        mock_session = MagicMock()
        mock_session.accumulated_usage = {"input_tokens": 10, "output_tokens": 20}
        mock_session.finalize = AsyncMock()
        mock_usage_logger.start_session.return_value = mock_session

        # Mock runnable with streaming
        async def mock_stream():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        mock_runnable = MagicMock()
        mock_runnable.astream.return_value = mock_stream()

        with (
            patch(
                "inference_core.llm.custom_task.get_llm_config", return_value=mock_config
            ),
            patch(
                "inference_core.llm.custom_task.UsageLogger",
                return_value=mock_usage_logger,
            ),
            patch("inference_core.llm.custom_task.LLMUsageCallbackHandler"),
        ):
            chunks = []
            async for chunk in stream_with_usage(
                task_type="extraction",
                runnable=mock_runnable,
                input={"text": "Extract entities"},
                model_name="test-model",
                session_id="test-session",
            ):
                chunks.append(chunk)

            # Verify chunks were yielded
            assert chunks == ["chunk1", "chunk2", "chunk3"]

            # Verify session was finalized with success
            mock_session.finalize.assert_called_once()
            finalize_args = mock_session.finalize.call_args
            assert finalize_args.kwargs["success"] is True
            assert finalize_args.kwargs["streamed"] is True
            assert finalize_args.kwargs["partial"] is False

    @pytest.mark.asyncio
    async def test_stream_with_usage_error_handling(self):
        """Test error handling and session finalization on streaming failure"""
        # Mock configuration
        mock_config = MagicMock()
        mock_model_cfg = MagicMock()
        mock_model_cfg.provider = "openai"
        mock_model_cfg.pricing = None
        mock_config.models = {"test-model": mock_model_cfg}
        mock_config.usage_logging = MagicMock(enabled=True)

        # Mock usage logger and session
        mock_usage_logger = MagicMock()
        mock_session = MagicMock()
        mock_session.accumulated_usage = {}
        mock_session.finalize = AsyncMock()
        mock_usage_logger.start_session.return_value = mock_session

        # Mock runnable that raises error during streaming
        async def mock_stream():
            yield "chunk1"
            raise ValueError("Streaming error")

        mock_runnable = MagicMock()
        mock_runnable.astream.return_value = mock_stream()

        with (
            patch(
                "inference_core.llm.custom_task.get_llm_config", return_value=mock_config
            ),
            patch(
                "inference_core.llm.custom_task.UsageLogger",
                return_value=mock_usage_logger,
            ),
            patch("inference_core.llm.custom_task.LLMUsageCallbackHandler"),
        ):
            chunks = []
            with pytest.raises(ValueError) as exc_info:
                async for chunk in stream_with_usage(
                    task_type="extraction",
                    runnable=mock_runnable,
                    input={"text": "Extract entities"},
                    model_name="test-model",
                ):
                    chunks.append(chunk)

            # Verify partial chunks were received before error
            assert chunks == ["chunk1"]

            # Verify error message
            assert str(exc_info.value) == "Streaming error"

            # Verify session was finalized with error and partial=True
            mock_session.finalize.assert_called_once()
            finalize_args = mock_session.finalize.call_args
            assert finalize_args.kwargs["success"] is False
            assert finalize_args.kwargs["streamed"] is True
            assert finalize_args.kwargs["partial"] is True

    @pytest.mark.asyncio
    async def test_stream_with_usage_extra_callbacks(self):
        """Test that extra callbacks are included in streaming mode"""
        # Mock configuration
        mock_config = MagicMock()
        mock_model_cfg = MagicMock()
        mock_model_cfg.provider = "openai"
        mock_model_cfg.pricing = None
        mock_config.models = {"test-model": mock_model_cfg}
        mock_config.usage_logging = MagicMock(enabled=True)

        # Mock usage logger and session
        mock_usage_logger = MagicMock()
        mock_session = MagicMock()
        mock_session.accumulated_usage = {}
        mock_session.finalize = AsyncMock()
        mock_usage_logger.start_session.return_value = mock_session

        # Mock runnable
        async def mock_stream():
            yield "chunk1"

        mock_runnable = MagicMock()
        mock_runnable.astream.return_value = mock_stream()

        # Create extra callbacks
        extra_cb = MagicMock()

        with (
            patch(
                "inference_core.llm.custom_task.get_llm_config", return_value=mock_config
            ),
            patch(
                "inference_core.llm.custom_task.UsageLogger",
                return_value=mock_usage_logger,
            ),
            patch("inference_core.llm.custom_task.LLMUsageCallbackHandler"),
        ):
            async for _ in stream_with_usage(
                task_type="extraction",
                runnable=mock_runnable,
                input={"text": "Test"},
                model_name="test-model",
                extra_callbacks=[extra_cb],
            ):
                pass

            # Verify runnable was called with callbacks
            mock_runnable.astream.assert_called_once()
            call_args = mock_runnable.astream.call_args
            callbacks = call_args.kwargs["config"]["callbacks"]
            assert len(callbacks) == 2  # usage callback + 1 extra
            assert callbacks[1] == extra_cb

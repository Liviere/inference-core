"""
Integration tests for LLM usage logging functionality
"""

import os
import uuid
from contextlib import asynccontextmanager
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import delete, select

from inference_core.database.sql.models.llm_request_log import LLMRequestLog
from inference_core.database.sql.models.pricing_snapshot import LLMPricingSnapshot
from inference_core.services.llm_service import LLMService
from inference_core.services.llm_usage_service import get_llm_usage_service


@pytest.mark.asyncio
class TestLLMUsageLogging:
    """Integration tests for LLM usage logging"""

    def setup_method(self, method):
        """Setup test environment"""
        os.environ["ENVIRONMENT"] = "testing"
        from inference_core.core.config import get_settings

        get_settings.cache_clear()

    @patch("inference_core.services.llm_service.create_explanation_chain")
    @patch("inference_core.services.llm_service.get_model_factory")
    @pytest.mark.asyncio
    async def test_usage_log_creation_on_explain_success(
        self, mock_get_model_factory, mock_create_chain, async_session_with_engine
    ):
        """Test that explain operations create usage logs"""
        service = LLMService()
        # Ensure usage logging is enabled for this test
        service.config.usage_logging.enabled = True
        session, _ = async_session_with_engine

        # Patch usage logging to use the test session/engine to avoid cross-loop issues
        @asynccontextmanager
        async def _get_async_session_override():
            try:
                yield session
            finally:
                pass

        # Mock model factory
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory

        # Mock chain
        mock_chain = AsyncMock()
        mock_chain.model_name = "gpt-5-nano"
        mock_chain.generate_story.return_value = "This is a test explanation."
        mock_create_chain.return_value = mock_chain

        # Call the explain method with patched session
        with patch(
            "inference_core.llm.usage_logging.get_async_session",
            new=_get_async_session_override,
        ):
            result = await service.explain(
                question="What is Python?", model_name="gpt-5-nano"
            )

        assert result.result["answer"] == "This is a test explanation."
        assert result.metadata.model_name == "gpt-5-nano"

        # Check that a usage log was created

        query = (
            select(LLMRequestLog)
            .where(
                LLMRequestLog.task_type == "explain",
                LLMRequestLog.model_name == "gpt-5-nano",
            )
            .order_by(LLMRequestLog.created_at.desc())
            .limit(1)
        )

        result = await session.execute(query)
        log_entry = result.scalar_one_or_none()

        assert log_entry is not None
        assert log_entry.task_type == "explain"
        assert log_entry.request_mode == "sync"
        assert log_entry.model_name == "gpt-5-nano"
        assert isinstance(log_entry.provider, str) and len(log_entry.provider) > 0
        assert log_entry.success is True
        assert log_entry.error_type is None
        assert log_entry.streamed is False
        assert log_entry.partial is False

        await session.execute(delete(LLMRequestLog))
        await session.commit()

    # async def test_usage_log_creation_on_explain_error(self):
    #     """Test that explain errors create usage logs with error information"""
    #     service = LLMService()

    #     # Patch model factory & chain at service layer (consistent with success test)
    #     with (
    #         patch(
    #             "inference_core.services.llm_service.get_model_factory"
    #         ) as mock_get_model_factory,
    #         patch(
    #             "inference_core.services.llm_service.create_explanation_chain"
    #         ) as mock_create_chain,
    #     ):
    #         mock_factory = MagicMock()
    #         mock_get_model_factory.return_value = mock_factory

    #         mock_chain = AsyncMock()
    #         mock_chain.model_name = "gpt-5-nano"
    #         mock_chain.generate_story.side_effect = ValueError("Test error")
    #         mock_create_chain.return_value = mock_chain

    #         # Call the explain method and expect it to raise
    #         with pytest.raises(ValueError, match="Test error"):
    #             await service.explain(
    #                 question="What is Python?", model_name="gpt-5-nano"
    #             )

    #     # Check that a usage log was created with error information
    #     async with get_async_session() as session:
    #         from sqlalchemy import select

    #         query = (
    #             select(LLMRequestLog)
    #             .where(
    #                 LLMRequestLog.task_type == "explain", LLMRequestLog.success == False
    #             )
    #             .order_by(LLMRequestLog.created_at.desc())
    #             .limit(1)
    #         )

    #         result = await session.execute(query)
    #         log_entry = result.scalar_one_or_none()

    #         assert log_entry is not None
    #         assert log_entry.success is False
    #         assert log_entry.error_type == "ValueError"
    #         assert "Test error" in log_entry.error_message

    @patch("inference_core.services.llm_service.create_conversation_chain")
    @patch("inference_core.services.llm_service.get_model_factory")
    async def test_usage_log_creation_on_converse_success(
        self, mock_get_model_factory, mock_create_chain, async_session_with_engine
    ):
        """Test that conversation operations create usage logs"""
        service = LLMService()
        service.config.usage_logging.enabled = True
        session, _ = async_session_with_engine
        session_id = str(uuid.uuid4())

        # Patch usage logging to use the test session/engine
        @asynccontextmanager
        async def _get_async_session_override():
            try:
                yield session
            finally:
                pass

        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory

        mock_chain = AsyncMock()
        mock_chain.model_name = "gpt-5-mini"
        mock_chain.chat.return_value = "Hello! How can I help you?"
        mock_create_chain.return_value = mock_chain

        # Call the converse method using the patched session
        with patch(
            "inference_core.llm.usage_logging.get_async_session",
            new=_get_async_session_override,
        ):
            result = await service.converse(
                session_id=session_id, user_input="Hello", model_name="gpt-5-mini"
            )

        assert result.result["reply"] == "Hello! How can I help you?"
        assert result.result["session_id"] == session_id

        # Check that a usage log was created
        query = (
            select(LLMRequestLog)
            .where(
                LLMRequestLog.task_type == "conversation",
                LLMRequestLog.session_id == session_id,
            )
            .order_by(LLMRequestLog.created_at.desc())
            .limit(1)
        )

        result = await session.execute(query)
        log_entry = result.scalar_one_or_none()

        assert log_entry is not None
        assert log_entry.task_type == "conversation"
        assert log_entry.request_mode == "sync"
        assert log_entry.model_name == "gpt-5-mini"
        assert log_entry.session_id == session_id
        assert log_entry.success is True

        await session.execute(delete(LLMRequestLog))
        await session.commit()

    async def test_usage_stats_aggregation(self, async_session_with_engine):
        """Test that usage statistics are properly aggregated"""
        usage_service = get_llm_usage_service()
        session, _ = async_session_with_engine

        # Ensure the usage service queries the same test session
        @asynccontextmanager
        async def _get_async_session_override():
            try:
                yield session
            finally:
                pass

        usage_service_ctx_patch = patch(
            "inference_core.services.llm_usage_service.get_async_session",
            new=_get_async_session_override,
        )
        usage_service_ctx_patch.start()
        try:
            # Create pricing snapshots and link them via FK for valid aggregation
            # Compute required snapshot_hash and fields
            hash1 = LLMPricingSnapshot.compute_hash(
                provider="openai",
                model_name="gpt-5-nano",
                currency="USD",
                input_cost_per_1k=Decimal("0.25"),
                output_cost_per_1k=Decimal("1.25"),
                extras={},
            )
            snapshot1 = LLMPricingSnapshot(
                snapshot_hash=hash1,
                provider="openai",
                model_name="gpt-5-nano",
                currency="USD",
                input_cost_per_1k=Decimal("0.25"),
                output_cost_per_1k=Decimal("1.25"),
                extras=None,
            )
            hash2 = LLMPricingSnapshot.compute_hash(
                provider="openai",
                model_name="gpt-5-mini",
                currency="USD",
                input_cost_per_1k=Decimal("0.15"),
                output_cost_per_1k=Decimal("0.60"),
                extras={},
            )
            snapshot2 = LLMPricingSnapshot(
                snapshot_hash=hash2,
                provider="openai",
                model_name="gpt-5-mini",
                currency="USD",
                input_cost_per_1k=Decimal("0.15"),
                output_cost_per_1k=Decimal("0.60"),
                extras=None,
            )
            session.add_all([snapshot1, snapshot2])
            await session.flush()

            # Create some test log entries directly
            log1 = LLMRequestLog(
                task_type="explain",
                request_mode="sync",
                model_name="gpt-5-nano",
                provider="openai",
                success=True,
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cost_input_usd=0.025,  # 100/1000 * 0.25
                cost_output_usd=0.0625,  # 50/1000 * 1.25
                cost_total_usd=0.0875,
                usage_raw={"input_tokens": 100, "output_tokens": 50},
                pricing_snapshot_id=snapshot1.id,
                request_id=str(uuid.uuid4()),
            )

            log2 = LLMRequestLog(
                task_type="conversation",
                request_mode="sync",
                model_name="gpt-5-mini",
                provider="openai",
                success=True,
                input_tokens=200,
                output_tokens=100,
                total_tokens=300,
                cost_input_usd=0.03,  # 200/1000 * 0.15
                cost_output_usd=0.06,  # 100/1000 * 0.60
                cost_total_usd=0.09,
                usage_raw={"input_tokens": 200, "output_tokens": 100},
                pricing_snapshot_id=snapshot2.id,
                request_id=str(uuid.uuid4()),
            )

            session.add_all([log1, log2])
            await session.commit()

            # Get usage statistics
            stats = await usage_service.get_usage_stats()

            # Verify aggregated statistics
            assert stats["usage"]["total_requests"] >= 2
            assert stats["usage"]["successful_requests"] >= 2
            assert stats["usage"]["total_tokens"] >= 450  # 150 + 300
            assert stats["usage"]["input_tokens"] >= 300  # 100 + 200
            assert stats["usage"]["output_tokens"] >= 150  # 50 + 100

            # Verify cost statistics
            assert stats["cost"]["currency"] == "USD"
            assert stats["cost"]["total"] >= 0.1775  # 0.0875 + 0.09
            assert "gpt-5-nano" in stats["cost"]["by_model"]
            assert "gpt-5-mini" in stats["cost"]["by_model"]
            assert "explain" in stats["cost"]["by_task_type"]
            assert "conversation" in stats["cost"]["by_task_type"]

        finally:
            await session.execute(delete(LLMRequestLog))
            await session.execute(delete(LLMPricingSnapshot))
            await session.commit()
            usage_service_ctx_patch.stop()

    async def test_enhanced_usage_stats_in_llm_service(self, async_session_with_engine):
        """Test that LLM service returns enhanced usage stats"""
        service = LLMService()
        # Ensure usage logging is enabled so enhanced stats are present
        service.config.usage_logging.enabled = True
        session, _ = async_session_with_engine

        # Patch usage service session to the test session so tables exist
        @asynccontextmanager
        async def _get_async_session_override():
            try:
                yield session
            finally:
                pass

        with patch(
            "inference_core.services.llm_usage_service.get_async_session",
            new=_get_async_session_override,
        ):
            # Get usage statistics from the service (should include both legacy and enhanced)
            stats = await service.get_usage_stats()

        # Legacy fields may be present for backward compatibility; if present they should be integers
        if "requests_count" in stats:
            assert isinstance(stats["requests_count"], int)
        if "errors_count" in stats:
            assert isinstance(stats["errors_count"], int)

        # Should also have enhanced usage and cost fields
        assert "usage" in stats
        assert "cost" in stats
        assert stats["usage"]["total_requests"] >= 0
        assert stats["cost"]["currency"] == "USD"
        assert stats["cost"]["total"] >= 0.0

    @patch("inference_core.services.llm_service.create_explanation_chain")
    @patch("inference_core.services.llm_service.get_model_factory")
    async def test_usage_logging_disabled(
        self, mock_get_model_factory, mock_create_chain, async_session_with_engine
    ):
        """Test behavior when usage logging is disabled"""
        service = LLMService()
        service.config.usage_logging.enabled = True
        session, _ = async_session_with_engine

        # Mock factory & chain
        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory
        mock_chain = AsyncMock()
        mock_chain.model_name = "gpt-5-nano"
        mock_chain.generate_story.return_value = "Test response"
        mock_create_chain.return_value = mock_chain

        # Temporarily disable usage logging
        original_enabled = service.config.usage_logging.enabled
        service.config.usage_logging.enabled = False

        try:
            # Call should still work when logging is disabled
            result = await service.explain(
                question="Test question", model_name="gpt-5-nano"
            )
            assert result.result["answer"] == "Test response"
        finally:
            # Restore original setting
            service.config.usage_logging.enabled = original_enabled

            await session.execute(delete(LLMRequestLog))
            await session.commit()

    @patch("inference_core.services.llm_service.create_explanation_chain")
    @patch("inference_core.services.llm_service.get_model_factory")
    async def test_usage_logging_with_pricing_config(
        self, mock_get_model_factory, mock_create_chain, async_session_with_engine
    ):
        """Test usage logging with pricing configuration"""
        service = LLMService()
        # Ensure usage logging is enabled so pricing snapshot is recorded
        service.config.usage_logging.enabled = True
        session, _ = async_session_with_engine

        # Patch usage logging to use the test session
        @asynccontextmanager
        async def _get_async_session_override():
            try:
                yield session
            finally:
                pass

        # Ensure pricing snapshot cache is clean for this isolated DB
        # to avoid using an ID from another test database/engine
        import inference_core.llm.usage_logging as usage_logging_module

        usage_logging_module._PRICING_SNAPSHOT_CACHE.clear()

        mock_factory = MagicMock()
        mock_get_model_factory.return_value = mock_factory

        # Mock the chain with usage metadata
        mock_chain = AsyncMock()
        mock_chain.model_name = "gpt-5-mini"  # This model has pricing config
        mock_chain.generate_story.return_value = "Test explanation"
        mock_create_chain.return_value = mock_chain

        # Call explain
        with patch(
            "inference_core.llm.usage_logging.get_async_session",
            new=_get_async_session_override,
        ):
            await service.explain(question="What is AI?", model_name="gpt-5-mini")

        # Check that the log entry was created with pricing information

        query = (
            select(LLMRequestLog)
            .where(
                LLMRequestLog.task_type == "explain",
                LLMRequestLog.model_name == "gpt-5-mini",
            )
            .order_by(LLMRequestLog.created_at.desc())
            .limit(1)
        )

        result = await session.execute(query)
        log_entry = result.scalar_one_or_none()

        assert log_entry is not None
        assert log_entry.pricing_snapshot_id is not None
        snapshot = await session.get(LLMPricingSnapshot, log_entry.pricing_snapshot_id)
        assert snapshot is not None
        assert snapshot.currency == "USD"
        assert float(snapshot.input_cost_per_1k) > 0.0
        assert float(snapshot.output_cost_per_1k) > 0.0

        await session.execute(delete(LLMRequestLog))
        await session.commit()

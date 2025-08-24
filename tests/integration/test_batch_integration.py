"""
Simple Integration Test for Batch Models

Basic integration test to verify batch models work with the database
using the same pattern as existing integration tests.
"""

import pytest
from sqlalchemy import text

from inference_core.database.sql.models.batch import BatchJob, BatchJobStatus
from inference_core.schemas.batch import BatchJobCreate
from inference_core.services.batch_service import BatchService


@pytest.mark.integration
class TestBatchIntegration:
    """Test batch functionality with real database"""

    @pytest.mark.asyncio
    async def test_batch_models_database_integration(self, async_session_with_engine):
        """Test that batch models work with the database"""
        async_session, _ = async_session_with_engine

        # Test that we can create a batch job
        service = BatchService(async_session)

        job_data = BatchJobCreate(
            provider="openai",
            model="gpt-4",
            mode="chat",
            request_count=5,
            config_json={"temperature": 0.7},
        )

        job = await service.create_batch_job(job_data)

        # Verify the job was created correctly
        assert job.id is not None
        assert job.provider == "openai"
        assert job.model == "gpt-4"
        assert job.status == BatchJobStatus.CREATED
        assert job.config_json == {"temperature": 0.7}

        # Test that we can query the database directly
        result = await async_session.execute(
            text("SELECT COUNT(*) FROM batch_jobs WHERE provider = :provider"),
            {"provider": "openai"},
        )
        count = result.scalar()
        assert count == 1

        # Test that we can retrieve the job
        retrieved_job = await service.get_batch_job(job.id)
        assert retrieved_job is not None
        assert retrieved_job.id == job.id
        assert retrieved_job.config_json == {"temperature": 0.7}

    @pytest.mark.asyncio
    async def test_batch_tables_exist(self, async_session_with_engine, test_settings):
        """Test that batch tables were created correctly across supported DB backends"""
        async_session, _ = async_session_with_engine

        tables_to_check = ["batch_jobs", "batch_items", "batch_events"]

        for table_name in tables_to_check:
            if test_settings.is_sqlite:
                query = text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name = :table"
                )
            elif test_settings.is_mysql:
                query = text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = DATABASE() AND table_name = :table"
                )
            elif test_settings.is_postgresql:
                # Using pg_tables for portability (to_regclass could also be used)
                query = text(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = :table"
                )
            else:
                raise AssertionError("Unsupported database type for test")

            result = await async_session.execute(query, {"table": table_name})
            table_exists = result.scalar()
            assert (
                table_exists == table_name
            ), f"Table {table_name} was not created (db={test_settings.database_url})"

    @pytest.mark.asyncio
    async def test_batch_indexes_exist(self, async_session_with_engine, test_settings):
        """Test that batch indexes were created correctly across supported DB backends"""
        async_session, _ = async_session_with_engine

        key_indexes = [
            "ix_batch_jobs_provider",
            "ix_batch_jobs_status",
            "ix_batch_jobs_status_provider",
            "ix_batch_items_batch_job_id",
            "ix_batch_items_status",
            "ix_batch_events_batch_job_id",
        ]

        for index_name in key_indexes:
            if test_settings.is_sqlite:
                query = text(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name = :index"
                )
            elif test_settings.is_mysql:
                # information_schema.statistics lists indexes (one row per column); DISTINCT index_name ensures single match
                query = text(
                    "SELECT DISTINCT index_name FROM information_schema.statistics "
                    "WHERE table_schema = DATABASE() AND index_name = :index"
                )
            elif test_settings.is_postgresql:
                query = text(
                    "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND indexname = :index"
                )
            else:
                raise AssertionError("Unsupported database type for test")

            result = await async_session.execute(query, {"index": index_name})
            index_exists = result.scalar()
            assert (
                index_exists == index_name
            ), f"Index {index_name} was not created (db={test_settings.database_url})"

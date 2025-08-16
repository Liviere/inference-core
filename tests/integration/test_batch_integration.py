"""
Simple Integration Test for Batch Models

Basic integration test to verify batch models work with the database
using the same pattern as existing integration tests.
"""

import pytest
from sqlalchemy import text

from app.database.sql.models.batch import BatchJob, BatchJobStatus
from app.schemas.batch import BatchJobCreate
from app.services.batch_service import BatchService


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
            config_json={"temperature": 0.7}
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
            {"provider": "openai"}
        )
        count = result.scalar()
        assert count == 1
        
        # Test that we can retrieve the job
        retrieved_job = await service.get_batch_job(job.id)
        assert retrieved_job is not None
        assert retrieved_job.id == job.id
        assert retrieved_job.config_json == {"temperature": 0.7}

    @pytest.mark.asyncio
    async def test_batch_tables_exist(self, async_session_with_engine):
        """Test that batch tables were created correctly"""
        async_session, _ = async_session_with_engine
        
        # Check that all batch tables exist
        tables_to_check = ["batch_jobs", "batch_items", "batch_events"]
        
        for table_name in tables_to_check:
            result = await async_session.execute(
                text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            )
            table_exists = result.scalar()
            assert table_exists == table_name, f"Table {table_name} was not created"

    @pytest.mark.asyncio 
    async def test_batch_indexes_exist(self, async_session_with_engine):
        """Test that batch indexes were created correctly"""
        async_session, _ = async_session_with_engine
        
        # Check that key indexes exist
        key_indexes = [
            "ix_batch_jobs_provider",
            "ix_batch_jobs_status", 
            "ix_batch_jobs_status_provider",
            "ix_batch_items_batch_job_id",
            "ix_batch_items_status",
            "ix_batch_events_batch_job_id"
        ]
        
        for index_name in key_indexes:
            result = await async_session.execute(
                text(f"SELECT name FROM sqlite_master WHERE type='index' AND name='{index_name}'")
            )
            index_exists = result.scalar()
            assert index_exists == index_name, f"Index {index_name} was not created"
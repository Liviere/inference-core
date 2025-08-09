"""
Integration tests for the API endpoints.
"""

import pytest
from sqlalchemy import text

from app.database.sql.connection import Base


@pytest.mark.asyncio
async def test_db_session(async_session_with_engine):
    """Test that database session fixture works"""
    # Simple test to check if session is working
    from sqlalchemy import text

    async_session, _ = async_session_with_engine

    result = await async_session.execute(text("SELECT 1 as test"))
    row = result.fetchone()
    assert row[0] == 1


@pytest.mark.asyncio
async def test_create_tables(async_session_with_engine, test_settings):
    """Test that database tables can be created"""
    async_session, async_engine = async_session_with_engine

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    assert async_session is not None, "Session should not be None after creating tables"
    assert async_session.bind is not None, "Session should have a bound engine"

    # Assert that the table users exists
    if test_settings.is_sqlite:
        result = await async_session.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        )

    elif test_settings.is_mysql:
        result = await async_session.execute(
            text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()"
            )
        )

    elif test_settings.is_postgresql:
        result = await async_session.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )

    row = result.fetchone()
    assert row is not None, "Users table should exist after creating tables"


@pytest.mark.asyncio
async def test_drop_tables(async_session_with_engine, test_settings):
    """Test that database tables can be dropped"""
    async_session, async_engine = async_session_with_engine

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    # Assert that the table users does not exist
    if test_settings.is_sqlite:
        result = await async_session.execute(
            text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'users'"
            )
        )
    elif test_settings.is_mysql:
        result = await async_session.execute(
            text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = 'users'"
            )
        )
    elif test_settings.is_postgresql:
        result = await async_session.execute(
            text(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = 'users'"
            )
        )
    row = result.fetchone()
    assert row is None, "Users table should not exist after dropping tables"


@pytest.mark.asyncio
async def test_health(async_test_client):
    """Test health endpoint with test client"""
    response = await async_test_client.get("/api/v1/health/")
    # assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "timestamp" in data
    assert "components" in data
    assert data["components"]["database"]["status"] == "healthy"
    assert data["components"]["application"]["status"] == "healthy"
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_database_health(async_test_client):
    """Test database health endpoint with test client"""
    response = await async_test_client.get("/api/v1/health/database")
    data = response.json()
    assert response.status_code == 200
    assert "status" in data
    assert "details" in data
    assert "last_updated" in data
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_root(async_test_client):
    response = await async_test_client.get("/")
    data = response.json()
    assert response.status_code == 200
    assert "message" in data
    assert "app_name" in data
    assert "app_description" in data
    assert "app_title" in data
    assert "version" in data
    assert "environment" in data
    assert "debug" in data
    assert "docs" in data


@pytest.mark.asyncio
async def test_ping(async_test_client):
    """Test ping endpoint with test client"""
    response = await async_test_client.get("/api/v1/health/ping")
    data = response.json()
    assert response.status_code == 200
    assert "message" in data
    assert data["message"] == "pong"


@pytest.mark.asyncio
async def test_non_existent_endpoint(async_test_client):
    """Test non-existent endpoint returns 404"""
    response = await async_test_client.get("/api/v1/non_existent")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "Not Found"

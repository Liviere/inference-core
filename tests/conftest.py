"""Test configuration and fixtures for API tests.

IMPORTANT: We force ENVIRONMENT=testing BEFORE importing any application modules.
Otherwise the settings model would have already chosen .env instead of .env.test
at class definition time (model_config env_file decision), leading to mixed values.
"""

import os
from typing import AsyncGenerator

# Ensure test environment flag is present before importing app.* modules
os.environ.setdefault("ENVIRONMENT", "testing")

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from inference_core.core.dependecies import get_db
from inference_core.database.sql.connection import (
    Base,
    create_database_engine,
    get_non_singleton_session_maker,
)
from inference_core.main_factory import create_application


@pytest_asyncio.fixture()
async def test_settings() -> AsyncGenerator:
    """Fixture to provide application settings for tests."""
    from inference_core.core.config import get_settings

    # Set ENVIRONMENT var to 'testing'
    os.environ["ENVIRONMENT"] = "testing"

    settings = get_settings()
    yield settings


@pytest_asyncio.fixture()
async def async_engine() -> AsyncGenerator:
    """Create an async database engine for testing."""
    # Create a temporary database engine
    engine = create_database_engine()
    try:
        yield engine
    finally:
        # Properly close all connections before disposing
        await engine.dispose()


@pytest_asyncio.fixture()
async def async_session_with_engine(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async database session for testing."""
    # Ensure all tables are created before yielding the session. Some tests
    # (e.g. batch integration) directly use the session without calling any
    # API startup hooks, so we must explicitly create metadata here.
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = get_non_singleton_session_maker(engine=async_engine)
    async with session_maker() as session:
        try:
            yield session, async_engine
        except Exception:
            await session.rollback()
            raise
        finally:
            # Explicitly close the session before the engine is disposed
            await session.close()

    # Optional: drop tables after the fixture scope to keep DB clean between
    # tests when using the same engine instance. (Engine itself disposed in
    # async_engine fixture.) Safe best-effort cleanup.
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    except Exception:
        pass


@pytest_asyncio.fixture()
async def async_test_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client with a temporary database."""
    # Create engine and session maker
    engine = create_database_engine()
    # Ensure tables exist for endpoints that interact with DB
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = get_non_singleton_session_maker(engine=engine)

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_maker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    app = create_application()
    app.dependency_overrides[get_db] = override_get_db

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        # Clean up dependency overrides and dispose engine
        app.dependency_overrides.clear()
        # Drop tables after tests to keep environment clean
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        except Exception:
            # Best-effort cleanup; ignore if DB not reachable
            pass
        await engine.dispose()

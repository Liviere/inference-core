"""
Test configuration and fixtures for Story Teller API tests.

This module provides common fixtures and configuration for all tests,
including database setup, test client, and Celery testing utilities.
"""

import os
from typing import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependecies import get_db
from app.database.sql.connection import (
    create_database_engine,
    get_non_singleton_session_maker,
)
from app.main import app


@pytest_asyncio.fixture()
async def test_settings() -> AsyncGenerator:
    """Fixture to provide application settings for tests."""
    from app.core.config import get_settings

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


@pytest_asyncio.fixture()
async def async_test_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client with a temporary database."""
    # Create engine and session maker
    engine = create_database_engine()
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

    app.dependency_overrides[get_db] = override_get_db

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        # Clean up dependency overrides and dispose engine
        app.dependency_overrides.clear()
        await engine.dispose()

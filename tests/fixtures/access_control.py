"""
Test fixtures for LLM access control configuration.

These fixtures allow tests to easily configure different access control modes
for LLM endpoints without affecting production defaults.
"""

import os
from typing import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from inference_core.core.config import Settings, get_settings
from inference_core.core.dependecies import get_current_active_user, get_current_superuser
from inference_core.main_factory import create_application


@pytest.fixture
def clear_settings_cache():
    """Clear the settings cache before and after test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def public_access_client():
    """Test client with public LLM access mode (no auth required)."""
    app = create_application()
    
    # Override auth dependencies to allow public access
    async def override_no_auth():
        return {
            "id": "12345678-1234-5678-9012-123456789012",  # Valid UUID format
            "username": "public", 
            "is_superuser": True
        }
    
    # Override both user and superuser dependencies to allow access
    app.dependency_overrides[get_current_active_user] = override_no_auth
    app.dependency_overrides[get_current_superuser] = override_no_auth
    
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
async def public_access_async_client():
    """Async test client with public LLM access mode (no auth required)."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from inference_core.database.sql.connection import (
        Base,
        create_database_engine,
        get_non_singleton_session_maker,
    )
    from inference_core.core.dependecies import get_db
    
    # Create engine and session maker like async_test_client does
    engine = create_database_engine()
    # Ensure tables exist for endpoints that interact with DB
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = get_non_singleton_session_maker(engine=engine)

    async def override_get_db() -> AsyncSession:
        async with session_maker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    app = create_application()
    
    # Override database dependency
    app.dependency_overrides[get_db] = override_get_db
    
    # Override auth dependencies to allow public access
    async def override_no_auth():
        return {
            "id": "12345678-1234-5678-9012-123456789012",  # Valid UUID format
            "username": "public", 
            "is_superuser": True
        }
    
    # Override both user and superuser dependencies to allow access
    app.dependency_overrides[get_current_active_user] = override_no_auth
    app.dependency_overrides[get_current_superuser] = override_no_auth
    
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        # Drop tables after tests to keep environment clean
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        except Exception:
            # Best-effort cleanup; ignore if DB not reachable
            pass
        await engine.dispose()


@pytest.fixture  
def user_access_client():
    """Test client with user LLM access mode (requires regular user auth)."""
    app = create_application()
    
    # Override superuser dependency to fail (simulating user without superuser privileges)
    async def override_superuser_fail():
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser privileges required"
        )
    
    app.dependency_overrides[get_current_superuser] = override_superuser_fail
    
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
async def user_access_async_client():
    """Async test client with user LLM access mode (requires regular user auth)."""
    app = create_application()
    
    # Override superuser dependency to fail
    async def override_superuser_fail():
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser privileges required"
        )
    
    app.dependency_overrides[get_current_superuser] = override_superuser_fail
    
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def superuser_access_client():
    """Test client with superuser LLM access mode (default behavior)."""
    app = create_application()
    return TestClient(app)


@pytest.fixture
async def superuser_access_async_client():
    """Async test client with superuser LLM access mode (default behavior)."""
    app = create_application()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
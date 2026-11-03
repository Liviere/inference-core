"""
Unit tests for inference_core.core.dependecies module

Tests CommonQueryParams, user dependency functions and authentication checks.
"""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from inference_core.core.dependecies import (
    CommonQueryParams,
    get_current_active_user,
    get_current_superuser,
)
from inference_core.schemas.auth import TokenData


class TestCommonQueryParams:
    """Test CommonQueryParams class"""

    def test_default_values(self):
        """Test CommonQueryParams with default values"""
        # CommonQueryParams uses FastAPI Query objects, so we need to test differently
        # Create instance to test the values that would be assigned
        params = CommonQueryParams()

        # The values are actually Query objects with default values
        assert hasattr(params, "q")
        assert hasattr(params, "sort_by")
        assert hasattr(params, "sort_order")
        assert hasattr(params, "include_deleted")

    def test_custom_values(self):
        """Test CommonQueryParams with custom values"""
        params = CommonQueryParams(
            q="search term",
            sort_by="created_at",
            sort_order="desc",
            include_deleted=True,
        )

        assert params.q == "search term"
        assert params.sort_by == "created_at"
        assert params.sort_order == "desc"
        assert params.include_deleted is True

    def test_partial_custom_values(self):
        """Test CommonQueryParams with partially custom values"""
        # Test that CommonQueryParams class can be instantiated with args
        # Since this is a FastAPI dependency, we test the class structure
        params = CommonQueryParams(q="test query", sort_by="name")

        assert params.q == "test query"
        assert params.sort_by == "name"
        # The remaining parameters should maintain their defaults but are Query objects
        assert hasattr(params, "sort_order")
        assert hasattr(params, "include_deleted")


class TestGetCurrentActiveUser:
    """Test get_current_active_user dependency"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from inference_core.core.config import get_settings

        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_active_user_success(self):
        """Test get_current_active_user with active user"""
        current_user = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "username": "testuser",
            "email": "test@example.com",
            "is_active": True,
            "is_superuser": False,
        }

        result = await get_current_active_user(current_user)

        assert result == current_user

    @pytest.mark.asyncio
    async def test_inactive_user_raises_exception(self):
        """Test get_current_active_user raises exception for inactive user"""
        current_user = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "username": "testuser",
            "email": "test@example.com",
            "is_active": False,
            "is_superuser": False,
        }

        with pytest.raises(HTTPException) as exc_info:
            await get_current_active_user(current_user)

        assert exc_info.value.status_code == 400
        assert "Inactive user" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_user_with_missing_is_active_raises_exception(self):
        """Test get_current_active_user raises exception when is_active is missing"""
        current_user = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "username": "testuser",
            "email": "test@example.com",
            "is_superuser": False,
        }
        # Missing is_active key should be treated as falsy

        with pytest.raises(HTTPException) as exc_info:
            await get_current_active_user(current_user)

        assert exc_info.value.status_code == 400
        assert "Inactive user" in str(exc_info.value.detail)


class TestGetCurrentSuperuser:
    """Test get_current_superuser dependency"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from inference_core.core.config import get_settings

        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_superuser_success(self):
        """Test get_current_superuser with superuser"""
        current_user = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "username": "admin",
            "email": "admin@example.com",
            "is_active": True,
            "is_superuser": True,
        }

        result = await get_current_superuser(current_user)

        assert result == current_user

    @pytest.mark.asyncio
    async def test_non_superuser_raises_exception(self):
        """Test get_current_superuser raises exception for non-superuser"""
        current_user = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "username": "testuser",
            "email": "test@example.com",
            "is_active": True,
            "is_superuser": False,
        }

        with pytest.raises(HTTPException) as exc_info:
            await get_current_superuser(current_user)

        assert exc_info.value.status_code == 403
        assert "Not enough permissions" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_user_with_missing_is_superuser_raises_exception(self):
        """Test get_current_superuser raises exception when is_superuser is missing"""
        current_user = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "username": "testuser",
            "email": "test@example.com",
            "is_active": True,
        }
        # Missing is_superuser key should be treated as falsy

        with pytest.raises(HTTPException) as exc_info:
            await get_current_superuser(current_user)

        assert exc_info.value.status_code == 403
        assert "Not enough permissions" in str(exc_info.value.detail)


class TestGetCurrentUser:
    """Test get_current_user dependency (integration-style test with mocking)"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from inference_core.core.config import get_settings

        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_get_current_user_success(self):
        """Test get_current_user successfully retrieves user"""
        from inference_core.core.dependecies import get_current_user

        # Mock token data
        token_data = TokenData(user_id="123e4567-e89b-12d3-a456-426614174000")

        # Mock database session and user
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_user = MagicMock()
        mock_user.id = uuid.UUID("123e4567-e89b-12d3-a456-426614174000")
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.first_name = "Test"
        mock_user.last_name = "User"
        mock_user.is_active = True
        mock_user.is_superuser = False
        mock_user.is_verified = True
        mock_user.created_at = "2023-01-01T00:00:00"
        mock_user.updated_at = "2023-01-01T00:00:00"

        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result

        result = await get_current_user(token_data, mock_db)

        # Verify result structure
        assert result["id"] == "123e4567-e89b-12d3-a456-426614174000"
        assert result["username"] == "testuser"
        assert result["email"] == "test@example.com"
        assert result["first_name"] == "Test"
        assert result["last_name"] == "User"
        assert result["is_active"] is True
        assert result["is_superuser"] is False
        assert result["is_verified"] is True
        assert "created_at" in result
        assert "updated_at" in result

        # Verify database was queried
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_current_user_not_found(self):
        """Test get_current_user raises exception when user not found"""
        from inference_core.core.dependecies import get_current_user

        # Mock token data
        token_data = TokenData(user_id="123e4567-e89b-12d3-a456-426614174000")

        # Mock database session with no user found
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(token_data, mock_db)

        assert exc_info.value.status_code == 404
        assert "User not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_uuid(self):
        """Test get_current_user handles invalid UUID gracefully"""
        from inference_core.core.dependecies import get_current_user

        # Mock token data with invalid UUID
        token_data = TokenData(user_id="invalid-uuid")

        # Mock database session
        mock_db = AsyncMock()

        with pytest.raises(ValueError):
            await get_current_user(token_data, mock_db)


class TestGetCurrentUserInOwnSession:
    """``get_current_user`` on a session of its own, closed before it returns."""

    USER_ID = "123e4567-e89b-12d3-a456-426614174000"

    def _user(self):
        user = MagicMock()
        user.id = uuid.UUID(self.USER_ID)
        user.username = "testuser"
        user.email = "test@example.com"
        user.first_name = "Test"
        user.last_name = "User"
        user.is_active = True
        user.is_superuser = False
        user.is_verified = True
        user.created_at = "2023-01-01T00:00:00"
        user.updated_at = "2023-01-01T00:00:00"
        return user

    def _sessions(self, found, events):
        from contextlib import asynccontextmanager

        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = found
        session.execute.return_value = result

        @asynccontextmanager
        async def get_async_session():
            events.append("open")
            try:
                yield session
            except Exception:
                events.append("raised inside")
                raise
            finally:
                events.append("closed")

        return session, get_async_session

    @pytest.mark.asyncio
    async def test_the_same_user_as_get_current_user(self):
        from inference_core.core.dependecies import (
            get_current_user,
            get_current_user_in_own_session,
        )
        from inference_core.database.sql import connection

        token = TokenData(user_id=self.USER_ID)
        events = []
        session, sessions = self._sessions(self._user(), events)
        with patch.object(connection, "get_async_session", sessions):
            own = await get_current_user_in_own_session(token)

        assert own == await get_current_user(token, session)
        assert events == ["open", "closed"]

    @pytest.mark.asyncio
    async def test_the_session_is_closed_before_it_returns(self):
        from inference_core.core.dependecies import get_current_user_in_own_session
        from inference_core.database.sql import connection

        events = []
        _session, sessions = self._sessions(self._user(), events)
        with patch.object(connection, "get_async_session", sessions):
            await get_current_user_in_own_session(TokenData(user_id=self.USER_ID))
            assert events[-1] == "closed"

    @pytest.mark.asyncio
    async def test_a_missing_user_is_404_raised_after_the_session(self):
        from inference_core.core.dependecies import get_current_user_in_own_session
        from inference_core.database.sql import connection

        events = []
        _session, sessions = self._sessions(None, events)
        with patch.object(connection, "get_async_session", sessions):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_in_own_session(TokenData(user_id=self.USER_ID))

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "User not found"
        # Raised outside: get_async_session logs what is raised inside as an error.
        assert events == ["open", "closed"]

    @pytest.mark.asyncio
    async def test_it_overrides_get_current_user_for_the_active_user(self):
        """An app that overrides get_current_user gets it through
        get_current_active_user, and the request's get_db is never used."""
        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient

        from inference_core.core.dependecies import (
            get_current_active_user,
            get_current_user,
            get_current_user_in_own_session,
            get_db,
        )
        from inference_core.core.security import get_current_user_token
        from inference_core.database.sql import connection

        app = FastAPI()

        @app.get("/me")
        async def me(user: dict = Depends(get_current_active_user)):
            return {"id": user["id"]}

        def no_request_session():
            raise AssertionError("the request's session was asked for")

        app.dependency_overrides[get_current_user] = get_current_user_in_own_session
        app.dependency_overrides[get_db] = no_request_session
        app.dependency_overrides[get_current_user_token] = lambda: TokenData(
            user_id=self.USER_ID
        )
        events = []
        _session, sessions = self._sessions(self._user(), events)
        with patch.object(connection, "get_async_session", sessions):
            response = TestClient(app).get("/me")

        assert response.status_code == 200
        assert response.json() == {"id": self.USER_ID}
        assert events == ["open", "closed"]


class TestGetDbDependency:
    """Test get_db dependency"""

    @patch.dict(os.environ, {"ENVIRONMENT": "testing"})
    def setup_method(self, method):
        """Setup test environment"""
        from inference_core.core.config import get_settings

        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_get_db_yields_session(self):
        """Test get_db yields database session"""
        from inference_core.core.dependecies import get_db

        mock_session = AsyncMock()

        with patch(
            "inference_core.core.dependecies.get_async_session"
        ) as mock_get_session:
            # Mock the async context manager
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_session
            mock_context.__aexit__.return_value = None
            mock_get_session.return_value = mock_context

            # get_db is an async generator, so iterate over it
            db_gen = get_db()
            try:
                db = await db_gen.__anext__()
                assert db == mock_session
                # Verify session was properly closed when generator is closed
            finally:
                await db_gen.aclose()

            # Context manager should be exited (session closed inside)
            mock_context.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_db_handles_exception(self):
        """Test get_db properly handles exceptions and closes session"""
        from inference_core.core.dependecies import get_db

        mock_session = AsyncMock()

        with patch(
            "inference_core.core.dependecies.get_async_session"
        ) as mock_get_session:
            # Mock the async context manager
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_session
            mock_context.__aexit__.return_value = None
            mock_get_session.return_value = mock_context

            # Test exception handling with async generator
            db_gen = get_db()
            try:
                db = await db_gen.__anext__()
                assert db == mock_session
                # Simulate exception in usage
                raise ValueError("Test exception")
            except ValueError:
                pass  # Expected
            finally:
                await db_gen.aclose()

            # Verify context manager exit was called despite exception
            mock_context.__aexit__.assert_called_once()

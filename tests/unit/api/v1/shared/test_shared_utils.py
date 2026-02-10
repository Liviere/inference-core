"""Unit tests for shared API utilities."""

import pytest
from datetime import datetime, UTC
from uuid import UUID

from fastapi import HTTPException

from inference_core.api.v1.shared import (
    get_iso_timestamp,
    create_pagination_response,
    get_user_id_from_context,
    verify_resource_ownership,
    handle_api_errors,
)


class TestResponseUtils:
    """Tests for response_utils module."""
    
    def test_get_iso_timestamp_format(self):
        """Test that timestamp is in ISO 8601 format."""
        timestamp = get_iso_timestamp()
        assert isinstance(timestamp, str)
        # Should be parseable as ISO format
        parsed = datetime.fromisoformat(timestamp)
        assert parsed.tzinfo is not None  # Should include timezone
    
    def test_get_iso_timestamp_is_recent(self):
        """Test that timestamp is current."""
        timestamp = get_iso_timestamp()
        parsed = datetime.fromisoformat(timestamp)
        now = datetime.now(UTC)
        # Should be within 1 second of now
        diff = abs((now - parsed).total_seconds())
        assert diff < 1.0
    
    def test_create_pagination_response_basic(self):
        """Test basic pagination response creation."""
        items = [{"id": 1}, {"id": 2}]
        result = create_pagination_response(
            items=items,
            total_count=100,
            offset=0,
            limit=10
        )
        
        assert result["items"] == items
        assert result["total"] == 100
        assert result["offset"] == 0
        assert result["limit"] == 10
        assert result["has_more"] is True
    
    def test_create_pagination_response_last_page(self):
        """Test pagination response on last page."""
        items = [{"id": 91}, {"id": 92}]
        result = create_pagination_response(
            items=items,
            total_count=92,
            offset=90,
            limit=10
        )
        
        assert result["has_more"] is False
    
    def test_create_pagination_response_exact_boundary(self):
        """Test pagination response at exact boundary."""
        items = [{"id": i} for i in range(10)]
        result = create_pagination_response(
            items=items,
            total_count=20,
            offset=10,
            limit=10
        )
        
        assert result["has_more"] is False
    
    def test_create_pagination_response_empty(self):
        """Test pagination response with no items."""
        result = create_pagination_response(
            items=[],
            total_count=0,
            offset=0,
            limit=10
        )
        
        assert result["items"] == []
        assert result["total"] == 0
        assert result["has_more"] is False


class TestAuthUtils:
    """Tests for auth_utils module."""
    
    def test_get_user_id_from_context_valid(self):
        """Test extracting valid user ID."""
        user_id_str = "550e8400-e29b-41d4-a716-446655440000"
        current_user = {"id": user_id_str, "username": "testuser"}
        
        result = get_user_id_from_context(current_user)
        
        assert isinstance(result, UUID)
        assert str(result) == user_id_str
    
    def test_get_user_id_from_context_missing(self):
        """Test error when user ID is missing."""
        current_user = {"username": "testuser"}
        
        with pytest.raises(HTTPException) as exc_info:
            get_user_id_from_context(current_user)
        
        assert exc_info.value.status_code == 401
        assert "not found" in exc_info.value.detail.lower()
    
    def test_get_user_id_from_context_invalid_format(self):
        """Test error when user ID format is invalid."""
        current_user = {"id": "not-a-uuid"}
        
        with pytest.raises(HTTPException) as exc_info:
            get_user_id_from_context(current_user)
        
        assert exc_info.value.status_code == 401
        assert "invalid" in exc_info.value.detail.lower()
    
    def test_verify_resource_ownership_success(self):
        """Test successful ownership verification."""
        user_id = UUID("550e8400-e29b-41d4-a716-446655440000")
        
        # Should not raise any exception
        verify_resource_ownership(
            resource_owner_id=user_id,
            current_user_id=user_id,
            resource_type="Job",
            resource_id="job-123"
        )
    
    def test_verify_resource_ownership_failure(self):
        """Test ownership verification failure."""
        owner_id = UUID("550e8400-e29b-41d4-a716-446655440000")
        other_user_id = UUID("550e8400-e29b-41d4-a716-446655440001")
        
        with pytest.raises(HTTPException) as exc_info:
            verify_resource_ownership(
                resource_owner_id=owner_id,
                current_user_id=other_user_id,
                resource_type="Job",
                resource_id="job-123"
            )
        
        assert exc_info.value.status_code == 404
        assert "Job job-123" in exc_info.value.detail
        assert "not found or access denied" in exc_info.value.detail
    
    def test_verify_resource_ownership_no_resource_id(self):
        """Test ownership verification without resource ID."""
        owner_id = UUID("550e8400-e29b-41d4-a716-446655440000")
        other_user_id = UUID("550e8400-e29b-41d4-a716-446655440001")
        
        with pytest.raises(HTTPException) as exc_info:
            verify_resource_ownership(
                resource_owner_id=owner_id,
                current_user_id=other_user_id,
                resource_type="Resource"
            )
        
        assert exc_info.value.status_code == 404
        assert "Resource not found or access denied" in exc_info.value.detail


class TestDecorators:
    """Tests for decorators module."""
    
    @pytest.mark.asyncio
    async def test_handle_api_errors_success(self):
        """Test decorator with successful execution."""
        @handle_api_errors
        async def successful_func():
            return {"status": "success"}
        
        result = await successful_func()
        assert result == {"status": "success"}
    
    @pytest.mark.asyncio
    async def test_handle_api_errors_http_exception(self):
        """Test decorator re-raises HTTPException."""
        @handle_api_errors
        async def func_with_http_error():
            raise HTTPException(status_code=400, detail="Bad request")
        
        with pytest.raises(HTTPException) as exc_info:
            await func_with_http_error()
        
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Bad request"
    
    @pytest.mark.asyncio
    async def test_handle_api_errors_generic_exception(self):
        """Test decorator converts generic exception to 500."""
        @handle_api_errors
        async def func_with_generic_error():
            raise ValueError("Something went wrong")
        
        with pytest.raises(HTTPException) as exc_info:
            await func_with_generic_error()
        
        assert exc_info.value.status_code == 500
        assert "Internal server error" in exc_info.value.detail
        assert "Something went wrong" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_handle_api_errors_preserves_function_metadata(self):
        """Test decorator preserves original function metadata."""
        @handle_api_errors
        async def documented_func():
            """This function has documentation."""
            return True
        
        assert documented_func.__name__ == "documented_func"
        assert "documentation" in documented_func.__doc__

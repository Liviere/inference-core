"""
Shared response utilities for API routes.

Provides helpers for common response formatting patterns.
"""

from datetime import UTC, datetime
from typing import Any, Dict, List, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


def get_iso_timestamp() -> str:
    """
    Get current UTC timestamp in ISO 8601 format.
    
    Returns:
        String timestamp in ISO 8601 format (e.g., "2024-01-15T10:30:00.123456+00:00")
    
    Example:
        >>> timestamp = get_iso_timestamp()
        >>> print(timestamp)
        "2024-01-15T10:30:00.123456+00:00"
    """
    return str(datetime.now(UTC).isoformat())


def create_pagination_response(
    items: List[T],
    total_count: int,
    offset: int = 0,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Create a standardized pagination response.
    
    Args:
        items: List of items for the current page
        total_count: Total number of items available
        offset: Current offset (default: 0)
        limit: Items per page (default: 10)
        
    Returns:
        Dictionary with items, pagination metadata, and has_more flag
        
    Example:
        >>> items = [{"id": 1}, {"id": 2}]
        >>> response = create_pagination_response(items, total_count=100, offset=0, limit=10)
        >>> print(response)
        {
            "items": [{"id": 1}, {"id": 2}],
            "total": 100,
            "offset": 0,
            "limit": 10,
            "has_more": True
        }
    """
    has_more = offset + limit < total_count
    
    return {
        "items": items,
        "total": total_count,
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
    }


def serialize_model(
    model: BaseModel, exclude_unset: bool = False, exclude_none: bool = False
) -> Dict[str, Any]:
    """
    Serialize a Pydantic model to a dictionary with common options.
    
    Args:
        model: Pydantic model to serialize
        exclude_unset: Whether to exclude fields that were not explicitly set
        exclude_none: Whether to exclude fields with None values
        
    Returns:
        Dictionary representation of the model
        
    Example:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     email: str | None = None
        >>> user = User(name="John")
        >>> serialize_model(user, exclude_unset=True)
        {"name": "John"}
    """
    return model.model_dump(exclude_unset=exclude_unset, exclude_none=exclude_none)

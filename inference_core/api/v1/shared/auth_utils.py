"""
Shared authentication utilities for API routes.

Provides helpers for common authentication and authorization patterns.
"""

from typing import Any, Dict
from uuid import UUID

from fastapi import HTTPException, status


def get_user_id_from_context(current_user: Dict[str, Any]) -> UUID:
    """
    Extract and convert user ID from the current user context.
    
    Args:
        current_user: Dictionary containing user information from JWT/auth
        
    Returns:
        UUID representation of the user ID
        
    Raises:
        HTTPException: If user ID is missing or invalid
        
    Example:
        >>> current_user = {"id": "550e8400-e29b-41d4-a716-446655440000"}
        >>> user_id = get_user_id_from_context(current_user)
        >>> print(user_id)
        UUID('550e8400-e29b-41d4-a716-446655440000')
    """
    try:
        user_id_str = current_user.get("id")
        if not user_id_str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in authentication context",
            )
        return UUID(user_id_str)
    except (ValueError, AttributeError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid user ID format: {str(e)}",
        )


def verify_resource_ownership(
    resource_owner_id: UUID,
    current_user_id: UUID,
    resource_type: str = "Resource",
    resource_id: str | None = None,
) -> None:
    """
    Verify that the current user owns the specified resource.
    
    Args:
        resource_owner_id: UUID of the resource owner
        current_user_id: UUID of the current user
        resource_type: Type of resource for error message (default: "Resource")
        resource_id: Optional resource ID for error message
        
    Raises:
        HTTPException: 404 if ownership verification fails
        
    Example:
        >>> user_id = UUID('550e8400-e29b-41d4-a716-446655440000')
        >>> job_owner = UUID('550e8400-e29b-41d4-a716-446655440001')
        >>> verify_resource_ownership(job_owner, user_id, "Batch job", "job-123")
        HTTPException: 404 - Batch job job-123 not found or access denied
    """
    if resource_owner_id != current_user_id:
        detail = f"{resource_type} not found or access denied"
        if resource_id:
            detail = f"{resource_type} {resource_id} not found or access denied"
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        )

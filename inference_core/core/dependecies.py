"""
FastAPI Dependencies

Common dependencies for FastAPI endpoints including database,
authentication, pagination, and other shared functionality.
"""

from typing import AsyncGenerator, List, Optional

from fastapi import Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from inference_core.core.config import get_settings
from inference_core.core.security import get_current_user_token
from inference_core.database.sql.connection import get_async_session
from inference_core.database.sql.models.user import User
from inference_core.schemas.auth import TokenData


class CommonQueryParams:
    """Common query parameters"""

    def __init__(
        self,
        q: Optional[str] = Query(None, description="Search query"),
        sort_by: Optional[str] = Query(None, description="Sort field"),
        sort_order: str = Query(
            "asc", pattern="^(asc|desc)$", description="Sort order"
        ),
        include_deleted: bool = Query(False, description="Include deleted items"),
    ):
        self.q = q
        self.sort_by = sort_by
        self.sort_order = sort_order
        self.include_deleted = include_deleted


# Database dependencies
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session dependency

    Yields:
        AsyncSession: Database session
    """
    async with get_async_session() as session:
        try:
            yield session
        finally:
            await session.close()


# Authentication dependencies
async def get_current_user(
    token_data: TokenData = Depends(get_current_user_token),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get current authenticated user

    Args:
        token_data: Token data from JWT
        db: Database session

    Returns:
        User data

    Raises:
        HTTPException: If user not found
    """
    # Fetch user from database and map to a simple dict for downstream use
    import uuid

    from sqlalchemy import select

    user_uuid = uuid.UUID(token_data.user_id)
    result = await db.execute(select(User).where(User.id == user_uuid))
    user: User | None = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "is_verified": user.is_verified,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
    }


async def get_current_active_user(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Get current active user

    Args:
        current_user: Current user data

    Returns:
        Active user data

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )
    return current_user


async def get_current_superuser(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Get current superuser

    Args:
        current_user: Current user data

    Returns:
        Superuser data

    Raises:
        HTTPException: If user is not superuser
    """
    if not current_user.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )
    return current_user


def get_llm_router_dependencies() -> List[Depends]:
    """
    Get router-level dependencies for LLM endpoints based on access mode configuration.
    
    Returns:
        List of FastAPI dependencies based on the LLM_API_ACCESS_MODE setting:
        - "public": No dependencies (no authentication required)
        - "user": Requires authenticated active user
        - "superuser": Requires superuser privileges (default)
    """
    settings = get_settings()
    access_mode = settings.llm_api_access_mode
    
    if access_mode == "public":
        return []
    elif access_mode == "user":
        return [Depends(get_current_active_user)]
    elif access_mode == "superuser":
        return [Depends(get_current_superuser)]
    else:
        # This should not happen due to Literal typing, but handle gracefully
        return [Depends(get_current_superuser)]

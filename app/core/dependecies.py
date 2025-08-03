"""
FastAPI Dependencies

Common dependencies for FastAPI endpoints including database,
authentication, pagination, and other shared functionality.
"""

from typing import AsyncGenerator, Optional

from fastapi import Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.sql.connection import get_async_session


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

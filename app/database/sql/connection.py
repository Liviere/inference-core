"""
Database Connection and Session Management

Handles SQLAlchemy async connections, session management,
and database initialization.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import MetaData, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Database metadata and base
metadata = MetaData()


class Base(DeclarativeBase):
    """Declarative base for all database models"""

    metadata = metadata


# Import all models to register them with SQLAlchemy
# This must be done after Base is defined but before create_tables() is called
from app.database.sql.models import *  # noqa: F401,E402

# Global variables for engine and session maker
_engine = None
_async_session_maker = None


def create_database_engine(database_url: str = None) -> Engine:
    """
    Create database engine with proper configuration

    Returns:
        Async SQLAlchemy engine
    """
    settings = get_settings()

    # Use the enhanced engine configuration from settings
    engine_kwargs = settings.get_database_engine_args()
    engine_kwargs["future"] = True

    # Create engine
    engine = create_async_engine(database_url or settings.database_url, **engine_kwargs)

    # Database-specific event listeners
    if settings.is_sqlite:

        @event.listens_for(Engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Enable foreign keys and other SQLite optimizations"""
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            cursor.execute("PRAGMA synchronous=NORMAL")  # Better performance
            cursor.close()

    elif settings.is_mysql:

        @event.listens_for(Engine, "connect")
        def set_mysql_pragma(dbapi_connection, connection_record):
            """Set MySQL session variables"""
            cursor = dbapi_connection.cursor()
            cursor.execute("SET SESSION sql_mode='STRICT_TRANS_TABLES'")
            cursor.execute("SET SESSION time_zone='+00:00'")  # UTC
            cursor.close()

    return engine


def get_engine(database_url: str = None) -> Engine:
    """
    Get or create database engine

    Returns:
        Async SQLAlchemy engine
    """
    global _engine
    if _engine is None:
        _engine = create_database_engine(database_url)
    return _engine


def get_session_maker():
    """
    Get or create session maker

    Returns:
        Async session maker
    """
    global _async_session_maker
    if _async_session_maker is None:
        engine = get_engine()
        _async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    return _async_session_maker


def get_non_singleton_session_maker(
    database_url: str = None, engine: Engine = None
) -> async_sessionmaker:
    """
    Get a new session maker instance

    Returns:
        Async session maker
    """
    if engine is None:
        engine = create_database_engine(database_url)
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session with context manager

    Yields:
        AsyncSession: Database session
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def create_tables():
    """
    Create database tables
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")


async def drop_tables():
    """
    Drop database tables
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("Database tables dropped")


async def close_database():
    """
    Close database connections
    """
    global _engine, _async_session_maker

    if _engine:
        await _engine.dispose()
        _engine = None

    _async_session_maker = None
    logger.info("Database connections closed")


class DatabaseManager:
    """Database management utilities"""

    @staticmethod
    async def init_database():
        """Initialize database"""
        try:
            await create_tables()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    @staticmethod
    async def reset_database():
        """Reset database (drop and recreate tables)"""
        try:
            await drop_tables()
            await create_tables()
            logger.info("Database reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            raise

    @staticmethod
    async def health_check(session=None) -> bool:
        """
        Check database health

        Returns:
            True if database is healthy
        """
        try:
            if session is None:
                session = await get_async_session()
            result = await session.execute(text("SELECT 1"))
            return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    @staticmethod
    async def get_database_info(session=None) -> dict:
        """
        Get database information

        Returns:
            Database info dictionary
        """
        settings = get_settings()
        health = await DatabaseManager.health_check(session)

        return {
            "url": settings.database_url.split("://")[0] + "://***",  # Hide credentials
            "status": "healthy" if health else "unhealthy",
            "pool_size": settings.database_pool_size,
            "max_overflow": settings.database_max_overflow,
            "echo": settings.database_echo,
        }


# Database manager instance
db_manager = DatabaseManager()

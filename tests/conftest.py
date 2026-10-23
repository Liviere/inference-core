"""Test configuration and fixtures for API tests.

IMPORTANT: We force ENVIRONMENT=testing BEFORE importing any application modules.
Otherwise the settings model would have already chosen .env instead of .env.test
at class definition time (model_config env_file decision), leading to mixed values.
"""

import os

os.environ.setdefault("ENVIRONMENT", "testing")
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from inference_core.core.config import Settings, get_settings
from inference_core.core.dependecies import get_db
from inference_core.database.sql.connection import (
    Base,
    create_database_engine,
    get_non_singleton_session_maker,
)
from inference_core.main_factory import create_application
from tests.fixtures.access_control import *

# Ensure test environment flag is present before importing app.* modules


@pytest_asyncio.fixture()
async def test_settings() -> AsyncGenerator:
    """Fixture to provide application settings for tests."""
    from inference_core.core.config import get_settings

    os.environ.setdefault("ENVIRONMENT", "testing")

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


@pytest_asyncio.fixture()
def async_test_client_factory():
    async def _factory(
        dependency_overrides: dict | None = None, **overrides
    ) -> AsyncGenerator[AsyncClient, None]:
        engine = create_database_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_maker = get_non_singleton_session_maker(engine=engine)

        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            async with session_maker() as session:
                yield session

        # Initialize app with custom settings
        settings = Settings(**overrides)
        app = create_application(custom_settings=settings)
        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_settings] = lambda: settings
        # Apply any test-specific dependency overrides (e.g., auth, services)
        if dependency_overrides:
            for dep, fn in dependency_overrides.items():
                app.dependency_overrides[dep] = fn

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client

        # cleanup
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()

    return _factory


# -- No test publishes to a real broker ---------------------------------------
#
# The test environment points Celery at a real Redis (``CELERY_BROKER_URL`` in
# ``.env.test``), and a task left there waits for ever in a queue other test
# runs share. A Celery app made with ``broker="memory://"`` still publishes to
# that Redis, because Celery prefers the environment's broker. Every publish
# ends in kombu's ``Producer.publish``, so that is where a test is stopped:
# through ``send_task``, ``apply_async`` and ``delay`` alike. The in-memory
# transport is allowed; a test that has to reach a real broker says so with
# ``@pytest.mark.real_broker``.

REAL_BROKER_MARK = "real_broker"
_real_broker_allowed = False


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"{REAL_BROKER_MARK}: the test may publish Celery tasks to a real broker",
    )


def _transport_of(producer) -> str:
    connection = producer.__connection__ or producer.connection
    transport = getattr(connection, "transport_cls", None)
    if isinstance(transport, str):
        return transport
    return getattr(transport, "driver_type", None) or repr(transport)


class RealBrokerPublish(RuntimeError):
    """A test published a task to a real broker."""


@pytest.fixture(scope="session", autouse=True)
def no_real_broker_publish():
    from kombu.messaging import Producer

    publish = Producer.publish

    def guarded(self, body, routing_key=None, *args, **kwargs):
        transport = _transport_of(self)
        if transport != "memory" and not _real_broker_allowed:
            headers = kwargs.get("headers") or {}
            raise RealBrokerPublish(
                f"{os.environ.get('PYTEST_CURRENT_TEST', 'A test')} published "
                f"Celery task {headers.get('task', '?')!r} to queue "
                f"{routing_key!r} of a real broker ({transport}). Mock the "
                f"send, or mark the test with @pytest.mark.{REAL_BROKER_MARK}."
            )
        return publish(self, body, routing_key, *args, **kwargs)

    Producer.publish = guarded
    try:
        yield
    finally:
        Producer.publish = publish


@pytest.fixture(autouse=True)
def real_broker_mark(request):
    global _real_broker_allowed
    _real_broker_allowed = request.node.get_closest_marker(REAL_BROKER_MARK) is not None
    try:
        yield
    finally:
        _real_broker_allowed = False

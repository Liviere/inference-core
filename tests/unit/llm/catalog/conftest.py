"""Fixtures for the model catalog tests: a database of its own, in memory."""

from contextlib import asynccontextmanager

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from inference_core.database.sql.models.model_catalog import (
    LLMCatalogEvent,
    LLMCatalogModel,
    LLMCatalogProviderState,
)

_TABLES = (LLMCatalogModel, LLMCatalogEvent, LLMCatalogProviderState)


@pytest_asyncio.fixture()
async def session_factory():
    """A session factory on an in-memory database holding the catalog tables."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        for model in _TABLES:
            await conn.run_sync(model.__table__.create)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def factory():
        async with maker() as session:
            yield session

    try:
        yield factory
    finally:
        await engine.dispose()


def make_config(providers, models):
    """An ``LLMConfig`` with these providers (raw YAML entries) and models.

    ``models`` maps a model name to its provider.
    """
    from unittest.mock import patch

    from inference_core.llm.config import LLMConfig, ModelConfig

    with patch.object(LLMConfig, "_load_config"):
        config = LLMConfig()
    config.providers = providers
    config.models = {
        name: ModelConfig(name=name, provider=provider)
        for name, provider in models.items()
    }
    config.agent_models = {}
    config.agent_configs = {}
    config._yaml_config = {}
    return config

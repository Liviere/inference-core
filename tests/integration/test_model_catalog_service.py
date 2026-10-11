"""
Integration test for the model catalog on the real database.

The unit tests run on SQLite, which stores JSON as text and hands datetimes
back without a zone. This one checks the same round trip on the configured
database: aliases and attributes as JSON, dates with their zone, and the
removal rule's arithmetic on what comes back.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest

from inference_core.database.sql.connection import get_non_singleton_session_maker
from inference_core.llm.catalog import (
    register_model_lister,
    unregister_model_lister,
)
from inference_core.llm.catalog.service import (
    catalog_events,
    catalog_models,
    provider_states,
    refresh,
)
from inference_core.llm.catalog.types import DiscoveredModel, ModelLifecycle
from inference_core.llm.provider_registry import (
    register_chat_model_provider,
    unregister_chat_model_provider,
)
from tests.unit.llm.catalog.conftest import make_config

PROVIDER = "catalog_it_gateway"
NOW = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)
RETIRES = datetime(2026, 12, 1, tzinfo=timezone.utc)


@pytest.fixture
def listing():
    models = []

    async def list_models(runtime, client):
        return list(models)

    register_chat_model_provider(PROVIDER, lambda config, params: None)
    register_model_lister(PROVIDER, list_models)
    yield models
    unregister_model_lister(PROVIDER)
    unregister_chat_model_provider(PROVIDER)


@pytest.mark.integration
async def test_catalog_round_trip(async_session_with_engine, listing):
    _, engine = async_session_with_engine
    maker = get_non_singleton_session_maker(engine=engine)

    @asynccontextmanager
    async def session_factory():
        async with maker() as session:
            yield session

    config = make_config({}, {"a": PROVIDER})
    listing[:] = [
        DiscoveredModel(
            provider=PROVIDER,
            model_id="a",
            lifecycle=ModelLifecycle.DEPRECATED,
            retires_at=RETIRES,
            aliases=("a-latest",),
            attributes={"pricing": {"input": 1.25}, "context_window": 8192},
        ),
        DiscoveredModel(provider=PROVIDER, model_id="b"),
    ]

    await refresh(config, session_factory=session_factory, now=NOW)

    first, second = await catalog_models(
        provider=PROVIDER, session_factory=session_factory
    )
    assert first.aliases == ["a-latest"]
    assert first.attributes == {"pricing": {"input": 1.25}, "context_window": 8192}
    assert first.retires_at == RETIRES
    assert first.retires_at.tzinfo is not None
    (state,) = [
        s
        for s in await provider_states(session_factory=session_factory)
        if s.provider == PROVIDER
    ]
    assert state.baselined_at == NOW

    # "b" leaves the listing: two readings a day apart take it for removed.
    listing[:] = listing[:1]
    for days in (1, 2):
        await refresh(
            config, session_factory=session_factory, now=NOW + timedelta(days=days)
        )

    events = await catalog_events(provider=PROVIDER, session_factory=session_factory)
    assert [(e.model_id, e.event_type) for e in events] == [("b", "removed")]
    assert events[0].details["reason"] == "not_listed"
    assert events[0].detected_at == NOW + timedelta(days=2)

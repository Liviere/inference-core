"""
Unit tests for the catalog in numbers, as the gauges show it.

Covers:
- every provider with a lister is in the summary, covered or not
- counts of configured models by state, and the earliest retirement date
- changes of the last 24 hours only
- when a covered provider is stale
"""

from datetime import datetime, timedelta, timezone

import pytest

from inference_core.database.sql.models.model_catalog import (
    LLMCatalogEvent,
    LLMCatalogProviderState,
)
from inference_core.llm.catalog import (
    listed_providers,
    register_model_lister,
    unregister_model_lister,
)
from inference_core.llm.catalog.service import refresh
from inference_core.llm.catalog.summary import summarize
from inference_core.llm.catalog.types import DiscoveredModel, ModelLifecycle
from inference_core.llm.provider_registry import (
    register_chat_model_provider,
    unregister_chat_model_provider,
)

from .conftest import make_config

NOW = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)
DAY = timedelta(days=1)
ACME = "acme_gateway"


@pytest.fixture
def listing():
    models = []

    async def list_models(runtime, client):
        return list(models)

    register_chat_model_provider(ACME, lambda config, params: None)
    register_model_lister(ACME, list_models)
    yield models
    unregister_model_lister(ACME)
    unregister_chat_model_provider(ACME)


def _model(model_id, **fields):
    return DiscoveredModel(provider=ACME, model_id=model_id, **fields)


async def _summary(config, session_factory, now=NOW):
    return await summarize(
        config, interval=DAY, now=now, session_factory=session_factory
    )


async def test_empty_catalog(listing, session_factory):
    summary = await _summary(make_config({}, {"a": ACME}), session_factory)

    assert summary.providers == listed_providers()
    assert ACME in summary.providers
    assert summary.last_success == {}
    assert summary.stale == {}
    assert summary.configured == {}
    assert summary.recent_events == {}


async def test_counts_configured_models_by_state(listing, session_factory):
    listing[:] = [
        _model("ok"),
        _model("old", lifecycle=ModelLifecycle.DEPRECATED),
        _model(
            "later",
            lifecycle=ModelLifecycle.DEPRECATED,
            retires_at=datetime(2027, 3, 1, tzinfo=timezone.utc),
        ),
        _model(
            "sooner",
            lifecycle=ModelLifecycle.DEPRECATED,
            retires_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
        ),
        _model("unconfigured", lifecycle=ModelLifecycle.DEPRECATED),
    ]
    config = make_config(
        {}, {name: ACME for name in ("ok", "old", "later", "sooner", "typo")}
    )
    await refresh(config, session_factory=session_factory, now=NOW)

    summary = await _summary(config, session_factory)

    assert summary.configured == {
        (ACME, "deprecated"): 1,
        (ACME, "retiring"): 2,
        (ACME, "missing"): 1,
    }
    assert summary.soonest_retirement == {
        ACME: datetime(2026, 12, 1, tzinfo=timezone.utc).timestamp()
    }
    assert summary.model_counts == {ACME: 5}
    assert summary.last_success == {ACME: NOW.timestamp()}
    assert summary.stale == {ACME: False}


async def test_counts_changes_of_the_last_day_only(listing, session_factory):
    async with session_factory() as session:
        for model_id, event_type, age in (
            ("a", "added", timedelta(hours=1)),
            ("b", "added", timedelta(hours=23)),
            ("c", "removed", timedelta(hours=2)),
            ("d", "added", timedelta(hours=25)),
        ):
            session.add(
                LLMCatalogEvent(
                    provider=ACME,
                    model_id=model_id,
                    event_type=event_type,
                    detected_at=NOW - age,
                )
            )
        await session.commit()

    summary = await _summary(make_config({}, {"a": ACME}), session_factory)

    assert summary.recent_events == {(ACME, "added"): 2, (ACME, "removed"): 1}


async def test_stale_after_three_intervals_without_a_reading(listing, session_factory):
    listing[:] = [_model("a")]
    config = make_config({}, {"a": ACME})
    await refresh(config, session_factory=session_factory, now=NOW)

    fresh = await _summary(config, session_factory, now=NOW + 3 * DAY)
    stale = await _summary(config, session_factory, now=NOW + 3 * DAY + DAY / 24)

    assert (fresh.stale, stale.stale) == ({ACME: False}, {ACME: True})


async def test_never_read_but_tried_is_stale(listing, session_factory):
    async with session_factory() as session:
        session.add(
            LLMCatalogProviderState(
                provider=ACME, model_count=0, last_attempt_at=NOW, last_error="X"
            )
        )
        await session.commit()

    summary = await _summary(make_config({}, {"a": ACME}), session_factory)

    assert summary.stale == {ACME: True}


async def test_provider_that_is_not_covered_is_never_stale(listing, session_factory):
    listing[:] = [_model("a")]
    await refresh(
        make_config({}, {"a": ACME}), session_factory=session_factory, now=NOW
    )
    switched_off = make_config(
        {ACME: {"name": "Acme", "catalog": {"enabled": False}}}, {"a": ACME}
    )

    summary = await _summary(switched_off, session_factory, now=NOW + 30 * DAY)

    assert summary.stale == {}
    assert summary.configured == {}

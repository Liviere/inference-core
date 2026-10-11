"""
Unit tests for drift: configured models the catalog has something to say about.

Covers:
- a model the provider does not know, has deprecated, or will retire
- a configured alias is the model it stands for
- models nobody configured are never drift
- a provider that was not read yet, or is skipped, says nothing
"""

from datetime import datetime, timezone

import pytest

from inference_core.database.sql.models.model_catalog import (
    LLMCatalogModel,
    LLMCatalogProviderState,
)
from inference_core.llm.catalog import (
    register_model_lister,
    unregister_model_lister,
)
from inference_core.llm.catalog.drift import compute_drift, current_drift
from inference_core.llm.catalog.service import refresh
from inference_core.llm.catalog.types import DiscoveredModel, ModelLifecycle
from inference_core.llm.provider_registry import (
    register_chat_model_provider,
    unregister_chat_model_provider,
)

from .conftest import make_config

NOW = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)
RETIRES = datetime(2026, 12, 1, tzinfo=timezone.utc)
ACME = "acme_gateway"


async def _list_models(runtime, client):
    return []


@pytest.fixture(autouse=True)
def _acme():
    register_chat_model_provider(ACME, lambda config, params: None)
    register_model_lister(ACME, _list_models)
    yield
    unregister_model_lister(ACME)
    unregister_chat_model_provider(ACME)


def _row(model_id: str, **fields) -> LLMCatalogModel:
    fields.setdefault("lifecycle", "active")
    return LLMCatalogModel(
        provider=ACME,
        model_id=model_id,
        source="list",
        kind="chat",
        first_seen_at=NOW,
        last_seen_at=NOW,
        missed_runs=0,
        **fields,
    )


def _read(provider: str = ACME) -> LLMCatalogProviderState:
    return LLMCatalogProviderState(provider=provider, model_count=1, baselined_at=NOW)


def _drift(configured, rows, states=None, providers=None):
    config = make_config(providers or {}, {name: ACME for name in configured})
    return compute_drift(config, rows, [_read()] if states is None else states)


def test_active_models_are_not_drift():
    assert (
        _drift(["a"], [_row("a"), _row("unconfigured", lifecycle="deprecated")]) == []
    )


def test_unknown_lifecycle_is_not_drift():
    assert _drift(["a"], [_row("a", lifecycle="unknown")]) == []


def test_model_the_provider_does_not_know_is_missing():
    (entry,) = _drift(["a", "typo"], [_row("a")])

    assert (entry.provider, entry.model, entry.state) == (ACME, "typo", "missing")


def test_removed_model_is_missing():
    (entry,) = _drift(["a"], [_row("a", removed_at=NOW)])

    assert entry.state == "missing"


def test_deprecated_model():
    deprecated_at = datetime(2026, 9, 1, tzinfo=timezone.utc)
    rows = [
        _row("a", lifecycle="deprecated", deprecated_at=deprecated_at, replacement="b")
    ]

    (entry,) = _drift(["a"], rows)

    assert entry.state == "deprecated"
    assert (entry.deprecated_at, entry.retires_at, entry.replacement) == (
        deprecated_at,
        None,
        "b",
    )


def test_model_with_a_retirement_date_is_retiring():
    (entry,) = _drift(["a"], [_row("a", lifecycle="deprecated", retires_at=RETIRES)])

    assert (entry.state, entry.retires_at) == ("retiring", RETIRES)


def test_retired_model_is_retiring():
    (entry,) = _drift(["a"], [_row("a", lifecycle="retired")])

    assert entry.state == "retiring"


def test_configured_alias_is_the_model_it_stands_for():
    rows = [_row("a-2026", aliases=["a-latest"], lifecycle="deprecated")]

    (entry,) = _drift(["a-latest"], rows)

    assert (entry.model, entry.state) == ("a-latest", "deprecated")


def test_a_row_of_its_own_wins_over_an_alias():
    rows = [
        _row("a-2026", aliases=["a"], lifecycle="deprecated"),
        _row("a"),
    ]

    assert _drift(["a"], rows) == []


def test_alias_of_a_removed_model_is_missing():
    rows = [_row("a-2026", aliases=["a"], removed_at=NOW)]

    (entry,) = _drift(["a"], rows)

    assert entry.state == "missing"


def test_provider_that_was_not_read_says_nothing():
    never = LLMCatalogProviderState(provider=ACME, model_count=0, last_error="X")

    assert _drift(["a"], [], states=[]) == []
    assert _drift(["a"], [], states=[never]) == []


def test_skipped_provider_says_nothing():
    providers = {ACME: {"name": "Acme", "catalog": {"enabled": False}}}

    assert _drift(["a"], [], providers=providers) == []


def test_other_providers_rows_do_not_count():
    other = _row("a")
    other.provider = "someone_else"

    (entry,) = _drift(["a"], [other])

    assert entry.state == "missing"


async def test_current_drift_reads_the_stored_catalog(session_factory):
    async def list_models(runtime, client):
        return [
            DiscoveredModel(
                provider=ACME,
                model_id="a",
                lifecycle=ModelLifecycle.DEPRECATED,
                retires_at=RETIRES,
            )
        ]

    register_model_lister(ACME, list_models)
    config = make_config({}, {"a": ACME, "b": ACME})
    await refresh(config, session_factory=session_factory, now=NOW)

    entries = await current_drift(config, session_factory=session_factory)

    assert [(e.model, e.state) for e in entries] == [
        ("a", "retiring"),
        ("b", "missing"),
    ]
    assert entries[0].retires_at == RETIRES

"""
Unit tests for drift: configured models the catalog has something to say about.

Covers:
- a model the provider does not know, has deprecated, or will retire
- a configured alias is the model it stands for
- models nobody configured are never drift
- a provider that was not read yet, or is skipped, says nothing
- the list of configured models: every provider, the catalog's state where
  there is one, the display name, how many agents name a model
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
from inference_core.llm.catalog.drift import (
    compute_drift,
    configured_models,
    current_drift,
)
from inference_core.llm.catalog.service import refresh
from inference_core.llm.catalog.types import DiscoveredModel, ModelLifecycle
from inference_core.llm.config import AgentConfig, ModelConfig, ToolModelOverrideConfig
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


class TestConfiguredModels:
    """The list of what is configured, with the catalog's word on each."""

    def test_every_model_with_its_catalog_state(self):
        config = make_config(
            {}, {name: ACME for name in ("a", "typo", "old", "going", "alias")}
        )
        rows = [
            _row("a"),
            _row("old", lifecycle="deprecated"),
            _row("going", retires_at=RETIRES),
            _row("dated-1", aliases=["alias"]),
            _row("unconfigured"),
        ]

        entries = configured_models(config, rows, [_read()])

        assert [(e.provider, e.model, e.catalog_state) for e in entries] == [
            (ACME, "a", "ok"),
            (ACME, "typo", "missing"),
            (ACME, "old", "deprecated"),
            (ACME, "going", "retiring"),
            (ACME, "alias", "ok"),
        ]

    def test_agrees_with_drift(self):
        config = make_config({}, {name: ACME for name in ("a", "typo", "old")})
        rows = [_row("a"), _row("old", lifecycle="deprecated")]

        listed = configured_models(config, rows, [_read()])
        drift = compute_drift(config, rows, [_read()])

        assert [(e.model, e.catalog_state) for e in listed if e.catalog_state != "ok"]
        assert [
            (e.model, e.catalog_state) for e in listed if e.catalog_state != "ok"
        ] == [(e.model, e.state) for e in drift]

    def test_providers_by_name_and_models_in_the_configs_order(self):
        config = make_config({}, {"z": ACME, "local": "ollama", "b": ACME})

        entries = configured_models(config, [_row("z"), _row("b")], [_read()])

        assert [(e.provider, e.model) for e in entries] == [
            (ACME, "z"),
            (ACME, "b"),
            ("ollama", "local"),
        ]

    def test_no_state_where_the_catalog_cannot_say(self):
        """Not read yet, switched off, or a provider nobody lists."""
        config = make_config({}, {"a": ACME, "local": "ollama"})
        off = make_config({ACME: {"catalog": {"enabled": False}}}, {"a": ACME})

        unread = configured_models(config, [], [])
        no_lister = configured_models(config, [_row("a")], [_read()])
        skipped = configured_models(off, [_row("a")], [_read()])

        assert [e.catalog_state for e in unread] == [None, None]
        assert [(e.model, e.catalog_state) for e in no_lister] == [
            ("a", "ok"),
            ("local", None),
        ]
        assert [e.catalog_state for e in skipped] == [None]

    def test_the_config_alone(self):
        config = make_config({}, {"a": ACME})

        (entry,) = configured_models(config)

        assert (entry.model, entry.catalog_state) == ("a", None)

    def test_display_name(self):
        config = make_config({}, {"a": ACME, "b": ACME, "c": ACME})
        config.models["a"] = ModelConfig(
            name="a", provider=ACME, display_name="Model A"
        )
        config.models["b"] = ModelConfig(name="b", provider=ACME, display_name="  ")

        entries = configured_models(config)

        assert [e.display_name for e in entries] == ["Model A", None, None]

    def test_counts_the_agents_that_name_a_model(self):
        """An agent counts once per model, in whichever role it names it."""
        config = make_config({}, {name: ACME for name in "abcde"})
        config.agent_configs = {
            "one": AgentConfig(primary="a", fallback=["b", "a"]),
            "two": AgentConfig(
                primary="a",
                multimodal_support_model="c",
                tool_model_overrides=[
                    ToolModelOverrideConfig(tool_name="t", model="d")
                ],
            ),
        }
        # The primary in force, not the one written: "two" was moved to "b".
        config.agent_models = {"one": "a", "two": "b"}

        entries = configured_models(config)

        assert {e.model: e.agents for e in entries} == {
            "a": 1,
            "b": 2,
            "c": 1,
            "d": 1,
            "e": 0,
        }

"""
Unit tests for the model catalog command line.

Covers:
- the arguments each command takes
- refresh: what is printed, and a failed provider as the exit code
- status and events: what is stored, as a table
- drift: the report, the exit code of --check, and providers not read yet
"""

import functools
from datetime import datetime, timedelta, timezone

import pytest

from inference_core.database.sql.models.model_catalog import LLMCatalogEvent
from inference_core.llm.catalog import (
    cli,
    register_model_lister,
    service,
    unregister_model_lister,
)
from inference_core.llm.catalog.http import CatalogHTTPError
from inference_core.llm.catalog.types import DiscoveredModel, ModelLifecycle
from inference_core.llm.provider_registry import (
    register_chat_model_provider,
    unregister_chat_model_provider,
)

from .conftest import make_config

ACME = "acme_gateway"
OTHER = "other_gateway"


class _Listing:
    def __init__(self):
        self.models = []
        self.error = None

    async def list_models(self, runtime, client):
        if self.error:
            raise self.error
        return list(self.models)


@pytest.fixture
def providers():
    listings = {ACME: _Listing(), OTHER: _Listing()}
    for name, listing in listings.items():
        register_chat_model_provider(name, lambda config, params: None)
        register_model_lister(name, listing.list_models)
    yield listings
    for name in listings:
        unregister_model_lister(name)
        unregister_chat_model_provider(name)


@pytest.fixture
def catalog(monkeypatch, session_factory, providers):
    """The commands on the test database, with a config of two providers."""
    config = make_config({}, {"a": ACME, "old": ACME, "typo": ACME, "x": OTHER})
    monkeypatch.setattr("inference_core.llm.config.get_llm_config", lambda: config)
    for name in ("refresh", "provider_states", "catalog_models", "catalog_events"):
        monkeypatch.setattr(
            service,
            name,
            functools.partial(getattr(service, name), session_factory=session_factory),
        )
    return providers


def _args(*argv):
    return cli.build_parser().parse_args(list(argv))


def _model(provider, model_id, **fields):
    return DiscoveredModel(provider=provider, model_id=model_id, **fields)


class TestParser:
    def test_a_command_is_required(self):
        with pytest.raises(SystemExit):
            _args()

    def test_refresh_takes_providers(self):
        assert _args("refresh").provider is None
        assert _args("refresh", "--provider", "a", "--provider", "b").provider == [
            "a",
            "b",
        ]

    def test_events_since(self):
        assert _args("events").since == timedelta(days=7)
        assert _args("events", "--since", "24h").since == timedelta(hours=24)
        assert _args("events", "--since", "30D").since == timedelta(days=30)
        with pytest.raises(SystemExit):
            _args("events", "--since", "yesterday")

    def test_drift_check(self):
        assert _args("drift").check is False
        assert _args("drift", "--check").check is True


class TestRefresh:
    async def test_prints_what_became_of_each_provider(self, catalog, capsys):
        catalog[ACME].models = [_model(ACME, "a"), _model(ACME, "b")]
        catalog[OTHER].error = CatalogHTTPError("HTTP 401", status_code=401)

        code = await cli._refresh(_args("refresh"))

        out = capsys.readouterr().out.splitlines()
        assert code == 1
        assert out[0].split() == [ACME, "refreshed", "2", "models,", "0", "changes"]
        assert out[1].split() == [OTHER, "failed", "CatalogHTTPError", "(HTTP", "401)"]

    async def test_exit_code_is_zero_when_nothing_failed(self, catalog, capsys):
        catalog[ACME].models = [_model(ACME, "a")]

        code = await cli._refresh(_args("refresh", "--provider", ACME))

        assert code == 0
        assert OTHER not in capsys.readouterr().out


class TestStatus:
    async def test_before_any_reading(self, catalog, capsys):
        assert await cli._status(_args("status")) == 0

        lines = [line.split() for line in capsys.readouterr().out.splitlines()]
        assert lines[0] == ["PROVIDER", "CONFIGURED", "LISTED", "LAST", "READ", "STATE"]
        assert lines[1] == [ACME, "3", "-", "-", "not", "read", "yet"]
        assert lines[2] == [OTHER, "1", "-", "-", "not", "read", "yet"]

    async def test_after_a_reading(self, catalog, capsys):
        catalog[ACME].models = [_model(ACME, "a")]
        catalog[OTHER].error = RuntimeError("boom")
        await cli._refresh(_args("refresh"))
        capsys.readouterr()

        await cli._status(_args("status"))

        lines = capsys.readouterr().out.splitlines()
        assert lines[1].split()[:3] == [ACME, "3", "1"]
        assert lines[1].endswith("ok")
        assert lines[2].endswith("failed: RuntimeError")


class TestDrift:
    async def test_reports_configured_models_that_need_a_look(self, catalog, capsys):
        catalog[ACME].models = [
            _model(ACME, "a"),
            _model(
                ACME,
                "old",
                lifecycle=ModelLifecycle.DEPRECATED,
                retires_at=datetime(2099, 12, 1, tzinfo=timezone.utc),
                replacement="new",
            ),
            _model(
                ACME,
                "typo",
                lifecycle=ModelLifecycle.DEPRECATED,
                retires_at=datetime(2020, 6, 16, tzinfo=timezone.utc),
            ),
        ]
        catalog[OTHER].models = [
            _model(
                OTHER,
                "x",
                lifecycle=ModelLifecycle.DEPRECATED,
                deprecated_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
            )
        ]
        await cli._refresh(_args("refresh"))
        capsys.readouterr()

        assert await cli._drift(_args("drift")) == 0
        report = capsys.readouterr()
        assert await cli._drift(_args("drift", "--check")) == 1

        lines = report.out.splitlines()
        assert lines[0].split() == ["PROVIDER", "MODEL", "STATE", "DETAIL"]
        assert lines[1].split(None, 3) == [
            ACME,
            "old",
            "retiring",
            "retires 2099-12-01, replaced by new",
        ]
        assert lines[2].split(None, 3) == [
            ACME,
            "typo",
            "retiring",
            "retirement date 2020-06-16 has passed",
        ]
        assert lines[3].split(None, 3) == [
            OTHER,
            "x",
            "deprecated",
            "deprecated 2026-09-01",
        ]
        assert report.err == ""

    async def test_reports_a_model_the_provider_does_not_know(self, catalog, capsys):
        catalog[ACME].models = [_model(ACME, "a"), _model(ACME, "old")]
        await cli._refresh(_args("refresh", "--provider", ACME))
        capsys.readouterr()

        assert await cli._drift(_args("drift", "--check")) == 1

        lines = capsys.readouterr().out.splitlines()
        assert lines[1].split(None, 3) == [
            ACME,
            "typo",
            "missing",
            "the provider does not know this model",
        ]

    async def test_nothing_to_report(self, catalog, capsys):
        catalog[ACME].models = [_model(ACME, name) for name in ("a", "old", "typo")]
        catalog[OTHER].models = [_model(OTHER, "x")]
        await cli._refresh(_args("refresh"))
        capsys.readouterr()

        assert await cli._drift(_args("drift", "--check")) == 0

        assert "current" in capsys.readouterr().out

    async def test_says_which_providers_were_not_read(self, catalog, capsys):
        catalog[ACME].models = [_model(ACME, name) for name in ("a", "old", "typo")]
        await cli._refresh(_args("refresh", "--provider", ACME))
        capsys.readouterr()

        assert await cli._drift(_args("drift", "--check")) == 0

        assert OTHER in capsys.readouterr().err


class TestEvents:
    async def test_lists_changes_newest_first(self, catalog, session_factory, capsys):
        now = datetime.now(timezone.utc)
        async with session_factory() as session:
            for model_id, event_type, age in (
                ("m1", "added", timedelta(hours=30)),
                ("m2", "removed", timedelta(hours=2)),
                ("m3", "added", timedelta(days=9)),
            ):
                session.add(
                    LLMCatalogEvent(
                        provider=ACME,
                        model_id=model_id,
                        event_type=event_type,
                        detected_at=now - age,
                    )
                )
            await session.commit()

        assert await cli._events(_args("events")) == 0
        week = [line.split()[2:] for line in capsys.readouterr().out.splitlines()]
        await cli._events(_args("events", "--since", "24h"))
        day = [line.split()[2:] for line in capsys.readouterr().out.splitlines()]

        assert week == [[ACME, "removed", "m2"], [ACME, "added", "m1"]]
        assert day == [[ACME, "removed", "m2"]]

    async def test_no_changes(self, catalog, capsys):
        assert await cli._events(_args("events", "--provider", OTHER)) == 0

        assert "No changes" in capsys.readouterr().out

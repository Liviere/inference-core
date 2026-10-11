"""
Unit tests for reading providers' listings into the catalog.

Covers:
- which providers are covered, and why one is skipped
- a reading stores models and state; the first one reports nothing
- a configured model the listing does not show is asked about by id
- a failed reading changes nothing but the provider's state, and says no more
  than the kind of failure
- an empty listing from a provider that had models is a failure
- one provider failing does not stop another
- only providers that are due are read when asked so
"""

from datetime import datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import select

from inference_core.database.sql.models.model_catalog import (
    LLMCatalogEvent,
    LLMCatalogModel,
    LLMCatalogProviderState,
)
from inference_core.llm.catalog import (
    register_model_lister,
    unregister_model_lister,
)
from inference_core.llm.catalog.http import CatalogHTTPError
from inference_core.llm.catalog.service import (
    catalog_events,
    catalog_models,
    describe_error,
    plan_providers,
    provider_states,
    refresh,
)
from inference_core.llm.catalog.types import DiscoveredModel, ModelLifecycle
from inference_core.llm.provider_registry import (
    register_chat_model_provider,
    unregister_chat_model_provider,
)

from .conftest import make_config

NOW = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)
DAY = timedelta(days=1)
ACME = "acme_gateway"
OTHER = "other_gateway"


class FakeProvider:
    """A provider whose listing and single-model answers the test sets."""

    def __init__(self, name: str):
        self.name = name
        self.listing = []
        self.lookups = {}
        self.error = None
        self.list_calls = 0
        self.lookup_calls = []

    def models(self, *model_ids, **fields):
        self.listing = [
            DiscoveredModel(provider=self.name, model_id=model_id, **fields)
            for model_id in model_ids
        ]

    async def list_models(self, runtime, client):
        self.list_calls += 1
        if self.error is not None:
            raise self.error
        return list(self.listing)

    async def get_model(self, runtime, client, model_id):
        self.lookup_calls.append(model_id)
        return self.lookups.get(model_id)


@pytest.fixture
def acme():
    provider = FakeProvider(ACME)
    register_chat_model_provider(ACME, lambda config, params: None)
    register_model_lister(ACME, provider.list_models, get_model=provider.get_model)
    yield provider
    unregister_model_lister(ACME)
    unregister_chat_model_provider(ACME)


@pytest.fixture
def other():
    provider = FakeProvider(OTHER)
    register_chat_model_provider(OTHER, lambda config, params: None)
    register_model_lister(OTHER, provider.list_models)
    yield provider
    unregister_model_lister(OTHER)
    unregister_chat_model_provider(OTHER)


@pytest.fixture
def client():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the fake listers make no requests")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _config(**models):
    return make_config({}, models or {"a": ACME})


async def _refresh(config, session_factory, client, **options):
    options.setdefault("now", NOW)
    return await refresh(
        config, client=client, session_factory=session_factory, **options
    )


async def _all(session_factory, model):
    async with session_factory() as session:
        return list((await session.execute(select(model))).scalars())


class TestPlan:
    def test_covers_providers_with_a_model_and_a_lister(self, acme):
        config = make_config(
            {"ollama": {"name": "Ollama"}}, {"a": ACME, "b": ACME, "llama": "ollama"}
        )

        (plan,) = plan_providers(config)

        assert (plan.provider, plan.configured, plan.skip_reason) == (
            ACME,
            ("a", "b"),
            None,
        )

    def test_a_lister_without_a_configured_model_is_not_covered(self, acme):
        assert plan_providers(make_config({}, {})) == []

    def test_switched_off(self, acme):
        config = make_config(
            {ACME: {"name": "Acme", "catalog": {"enabled": False}}}, {"a": ACME}
        )

        assert plan_providers(config)[0].skip_reason == "disabled"

    def test_without_its_key(self, monkeypatch):
        monkeypatch.delenv("CATALOG_TEST_KEY", raising=False)
        entry = {
            "name": "OpenAI",
            "requires_api_key": True,
            "api_key_env": "CATALOG_TEST_KEY",
        }
        config = make_config({"openai": entry}, {"gpt-5": "openai"})

        assert plan_providers(config)[0].skip_reason == "no API key"

        monkeypatch.setenv("CATALOG_TEST_KEY", "sk-test")
        assert plan_providers(config)[0].skip_reason is None

    def test_compatible_server_without_an_address(self):
        entry = {"name": "Custom", "openai_compatible": True}
        config = make_config(
            {"custom_openai_compatible": entry}, {"m": "custom_openai_compatible"}
        )

        assert plan_providers(config)[0].skip_reason == "no base_url"


class TestRefresh:
    async def test_first_reading_stores_models_and_reports_nothing(
        self, acme, session_factory, client
    ):
        acme.models("a", "b", lifecycle=ModelLifecycle.ACTIVE)

        (result,) = await _refresh(_config(), session_factory, client)

        assert (result.provider, result.outcome) == (ACME, "refreshed")
        assert (result.model_count, result.events) == (2, 0)
        models = await catalog_models(session_factory=session_factory)
        assert [(m.model_id, m.lifecycle, m.source) for m in models] == [
            ("a", "active", "list"),
            ("b", "active", "list"),
        ]
        (state,) = await provider_states(session_factory=session_factory)
        assert state.provider == ACME
        assert state.model_count == 2
        assert state.last_error is None
        assert state.baselined_at is not None
        assert await _all(session_factory, LLMCatalogEvent) == []

    async def test_second_reading_reports_what_changed(
        self, acme, session_factory, client
    ):
        acme.models("a")
        await _refresh(_config(), session_factory, client)
        acme.models("a", "b")

        (result,) = await _refresh(_config(), session_factory, client, now=NOW + DAY)

        assert result.events == 1
        (event,) = await catalog_events(session_factory=session_factory)
        assert (event.provider, event.model_id, event.event_type) == (
            ACME,
            "b",
            "added",
        )

    async def test_same_listing_twice_reports_nothing(
        self, acme, session_factory, client
    ):
        acme.models("a", "b")
        await _refresh(_config(), session_factory, client)

        (result,) = await _refresh(_config(), session_factory, client, now=NOW + DAY)

        assert result.events == 0
        assert len(await _all(session_factory, LLMCatalogModel)) == 2

    async def test_unlisted_configured_model_is_asked_about(
        self, acme, session_factory, client
    ):
        acme.models("a-2026", aliases=("a",))
        acme.lookups["alias"] = DiscoveredModel(
            provider=ACME,
            model_id="b-2026",
            lifecycle=ModelLifecycle.DEPRECATED,
            attributes={"display_name": "B"},
        )
        config = _config(a=ACME, alias=ACME, gone=ACME)

        await _refresh(config, session_factory, client)

        # "a" is an alias the listing shows; only the other two are asked about.
        assert acme.lookup_calls == ["alias", "gone"]
        models = {
            m.model_id: m for m in await catalog_models(session_factory=session_factory)
        }
        assert sorted(models) == ["a-2026", "alias"]
        assert models["alias"].source == "lookup"
        assert models["alias"].lifecycle == "deprecated"
        assert models["alias"].attributes == {
            "display_name": "B",
            "resolves_to": "b-2026",
        }

    async def test_model_the_provider_no_longer_knows_is_removed(
        self, acme, session_factory, client
    ):
        acme.models("a", "b")
        await _refresh(_config(), session_factory, client)
        acme.models("b")

        (result,) = await _refresh(
            _config(), session_factory, client, now=NOW + timedelta(minutes=1)
        )

        assert acme.lookup_calls == ["a"]
        assert result.events == 1
        (event,) = await catalog_events(session_factory=session_factory)
        assert (event.model_id, event.event_type) == ("a", "removed")
        assert event.details["reason"] == "not_found"

    async def test_failed_reading_changes_only_the_state(
        self, acme, session_factory, client
    ):
        acme.models("a")
        await _refresh(_config(), session_factory, client)
        acme.error = CatalogHTTPError("HTTP 401", status_code=401)

        (result,) = await _refresh(_config(), session_factory, client, now=NOW + DAY)

        assert (result.outcome, result.detail) == (
            "failed",
            "CatalogHTTPError (HTTP 401)",
        )
        (state,) = await provider_states(session_factory=session_factory)
        assert state.last_error == "CatalogHTTPError (HTTP 401)"
        assert state.last_success_at.replace(tzinfo=timezone.utc) == NOW
        assert state.last_attempt_at.replace(tzinfo=timezone.utc) == NOW + DAY
        (model,) = await catalog_models(session_factory=session_factory)
        assert (model.missed_runs, model.removed_at) == (0, None)
        assert await _all(session_factory, LLMCatalogEvent) == []

    async def test_a_reading_that_works_clears_the_error(
        self, acme, session_factory, client
    ):
        acme.error = RuntimeError("boom")
        await _refresh(_config(), session_factory, client)
        acme.error = None
        acme.models("a")

        await _refresh(_config(), session_factory, client, now=NOW + DAY)

        (state,) = await provider_states(session_factory=session_factory)
        assert state.last_error is None
        assert state.model_count == 1

    async def test_empty_listing_after_models_is_a_failure(
        self, acme, session_factory, client
    ):
        acme.models("a", "b")
        await _refresh(_config(), session_factory, client)
        acme.models()

        (result,) = await _refresh(
            _config(), session_factory, client, now=NOW + 5 * DAY
        )

        assert (result.outcome, result.detail) == ("failed", "EmptyListing")
        models = await catalog_models(session_factory=session_factory)
        assert [m.missed_runs for m in models] == [0, 0]

    async def test_one_provider_failing_does_not_stop_another(
        self, acme, other, session_factory, client
    ):
        acme.error = httpx.ConnectError("https://secret.test/?key=sk-1 refused")
        other.models("x")

        results = await _refresh(_config(a=ACME, x=OTHER), session_factory, client)

        assert [(r.provider, r.outcome) for r in results] == [
            (ACME, "failed"),
            (OTHER, "refreshed"),
        ]
        assert results[0].detail == "ConnectError"
        states = {
            s.provider: s
            for s in await provider_states(session_factory=session_factory)
        }
        assert states[ACME].last_error == "ConnectError"
        assert states[ACME].last_success_at is None

    async def test_provider_without_lookup_is_only_listed(
        self, other, session_factory, client
    ):
        other.models("x")

        (result,) = await _refresh(_config(y=OTHER), session_factory, client)

        assert result.outcome == "refreshed"

    async def test_named_providers_only(self, acme, other, session_factory, client):
        acme.models("a")
        other.models("x")

        results = await _refresh(
            _config(a=ACME, x=OTHER), session_factory, client, providers=[OTHER, "nope"]
        )

        assert [(r.provider, r.outcome, r.detail) for r in results] == [
            ("nope", "skipped", "not in the catalog"),
            (OTHER, "refreshed", None),
        ]
        assert acme.list_calls == 0

    async def test_skipped_provider_is_not_read(self, acme, session_factory, client):
        config = make_config(
            {ACME: {"name": "Acme", "catalog": {"enabled": False}}}, {"a": ACME}
        )

        (result,) = await _refresh(config, session_factory, client)

        assert (result.outcome, result.detail) == ("skipped", "disabled")
        assert acme.list_calls == 0
        assert await provider_states(session_factory=session_factory) == []


class TestOnlyDue:
    async def test_read_again_only_after_the_interval(
        self, acme, session_factory, client
    ):
        acme.models("a")
        config = _config()
        await _refresh(config, session_factory, client, only_due=True)

        (early,) = await _refresh(
            config,
            session_factory,
            client,
            only_due=True,
            now=NOW + timedelta(hours=23),
        )
        (due,) = await _refresh(
            config, session_factory, client, only_due=True, now=NOW + DAY
        )

        assert (early.outcome, due.outcome) == ("not_due", "refreshed")
        assert acme.list_calls == 2

    async def test_failed_reading_is_tried_again_after_an_hour(
        self, acme, session_factory, client
    ):
        acme.error = RuntimeError("boom")
        config = _config()
        await _refresh(config, session_factory, client, only_due=True)

        (early,) = await _refresh(
            config,
            session_factory,
            client,
            only_due=True,
            now=NOW + timedelta(minutes=59),
        )
        (due,) = await _refresh(
            config, session_factory, client, only_due=True, now=NOW + timedelta(hours=1)
        )

        assert (early.outcome, due.outcome) == ("not_due", "failed")

    async def test_without_only_due_a_provider_is_always_read(
        self, acme, session_factory, client
    ):
        acme.models("a")
        await _refresh(_config(), session_factory, client)
        await _refresh(
            _config(), session_factory, client, now=NOW + timedelta(minutes=1)
        )

        assert acme.list_calls == 2


class TestDescribeError:
    def test_http_error_says_the_status(self):
        error = CatalogHTTPError("HTTP 429", status_code=429)

        assert describe_error(error) == "CatalogHTTPError (HTTP 429)"

    def test_other_errors_say_only_their_kind(self):
        error = httpx.ReadTimeout("https://api.test/v1/models?key=sk-secret")

        assert describe_error(error) == "ReadTimeout"

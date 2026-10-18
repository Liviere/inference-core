"""
The catalog asks the endpoint a model is called on.

Covers:
- a provider's listing is read at its base_url, the one its chat models use
- without a base_url the lister's default address is asked, as before
- a model entry with a base_url of its own is read at that one: by the
  listing there, by id when the listing does not show it, and as gone when
  that endpoint says there is no such model, whatever the provider's own
  endpoint lists under its name
- the rest of the other endpoint's listing is not stored
"""

from typing import Dict, List

import httpx
import pytest
import yaml

from inference_core.llm.catalog.service import (
    catalog_events,
    catalog_models,
    plan_providers,
    refresh,
)
from inference_core.llm.config import LLMConfig
from inference_core.llm.models import LLMModelFactory

from .test_service import NOW

GLOBAL = "https://api.mistral.ai/v1"
REGIONAL = "https://api.eu.example.com/v1"


def _listing(*model_ids: str, aliases: Dict[str, List[str]] = None) -> dict:
    aliases = aliases or {}
    return {
        "data": [
            {
                "id": model_id,
                "aliases": aliases.get(model_id, []),
                "capabilities": {"completion_chat": True},
            }
            for model_id in model_ids
        ]
    }


class Endpoints:
    """Mistral-shaped endpoints by base URL, and the requests they got."""

    def __init__(self):
        self.listings: Dict[str, dict] = {}
        self.models: Dict[str, Dict[str, dict]] = {}
        self.requests: List[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        for base, listing in self.listings.items():
            if url == f"{base}/models":
                return httpx.Response(200, json=listing)
            prefix = f"{base}/models/"
            if url.startswith(prefix):
                model = self.models.get(base, {}).get(url[len(prefix) :])
                if model is None:
                    return httpx.Response(404, json={"detail": "not found"})
                return httpx.Response(200, json=model)
        raise AssertionError(f"unexpected request: {url}")


@pytest.fixture
def endpoints():
    return Endpoints()


@pytest.fixture
def client(endpoints):
    return httpx.AsyncClient(transport=httpx.MockTransport(endpoints.handler))


@pytest.fixture
def load_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MISTRAL_KEY", "test-key")
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled", lambda: False
    )

    def load(provider_base_url=None, models=None) -> LLMConfig:
        provider = {
            "name": "Mistral AI",
            "api_key_env": "TEST_MISTRAL_KEY",
            "requires_api_key": True,
        }
        if provider_base_url:
            provider["base_url"] = provider_base_url
        raw = {
            "providers": {"mistral": provider},
            "models": models or {"mistral-small-latest": {"provider": "mistral"}},
        }
        path = tmp_path / "llm_config.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        monkeypatch.setenv("LLM_CONFIG_PATH", str(path))
        return LLMConfig()

    return load


async def _refresh(config, session_factory, client):
    return await refresh(
        config, client=client, session_factory=session_factory, now=NOW
    )


class TestProviderEndpoint:
    async def test_listing_is_read_where_the_chat_models_go(
        self, load_config, endpoints, client, session_factory
    ):
        config = load_config(provider_base_url=REGIONAL + "/")
        endpoints.listings[REGIONAL] = _listing("mistral-small-latest")

        (result,) = await _refresh(config, session_factory, client)

        assert result.outcome == "refreshed"
        assert endpoints.requests == [f"{REGIONAL}/models"]
        chat = LLMModelFactory(config).create_model("mistral-small-latest")
        assert str(chat.client.base_url).rstrip("/") == REGIONAL

    async def test_without_a_base_url_the_default_is_asked(
        self, load_config, endpoints, client, session_factory
    ):
        config = load_config()
        endpoints.listings[GLOBAL] = _listing("mistral-small-latest")

        (result,) = await _refresh(config, session_factory, client)

        assert result.outcome == "refreshed"
        assert endpoints.requests == [f"{GLOBAL}/models"]


class TestModelEndpoint:
    MODELS = {
        "mistral-small-latest": {"provider": "mistral"},
        "mistral-small-2603": {"provider": "mistral", "base_url": REGIONAL},
    }

    def test_plan_names_the_models_called_elsewhere(self, load_config):
        config = load_config(
            models={
                **self.MODELS,
                # The provider's own address, written once more: not elsewhere.
                "mistral-large-latest": {
                    "provider": "mistral",
                    "base_url": REGIONAL + "/",
                },
            },
            provider_base_url=REGIONAL,
        )
        (plan,) = plan_providers(config)
        assert plan.own_endpoints == ()

        config = load_config(models=self.MODELS)
        (plan,) = plan_providers(config)
        assert plan.own_endpoints == (("mistral-small-2603", REGIONAL),)

    async def test_read_at_its_own_endpoint(
        self, load_config, endpoints, client, session_factory
    ):
        config = load_config(models=self.MODELS)
        endpoints.listings[GLOBAL] = _listing(
            "mistral-small-latest", "mistral-small-2603", "mistral-medium-latest"
        )
        endpoints.listings[REGIONAL] = _listing(
            "mistral-small-2603", "regional-only-model"
        )

        (result,) = await _refresh(config, session_factory, client)

        assert result.outcome == "refreshed"
        assert endpoints.requests == [f"{GLOBAL}/models", f"{REGIONAL}/models"]
        models = {
            m.model_id: m for m in await catalog_models(session_factory=session_factory)
        }
        # The regional listing's other models are not the provider's.
        assert sorted(models) == [
            "mistral-medium-latest",
            "mistral-small-2603",
            "mistral-small-latest",
        ]
        assert models["mistral-small-2603"].source == "lookup"
        assert models["mistral-small-latest"].source == "list"

    async def test_an_alias_there_is_kept_under_the_configured_name(
        self, load_config, endpoints, client, session_factory
    ):
        config = load_config(models=self.MODELS)
        endpoints.listings[GLOBAL] = _listing("mistral-small-latest")
        endpoints.listings[REGIONAL] = _listing(
            "mistral-small-2603-v2",
            aliases={"mistral-small-2603-v2": ["mistral-small-2603"]},
        )

        await _refresh(config, session_factory, client)

        models = {
            m.model_id: m for m in await catalog_models(session_factory=session_factory)
        }
        assert models["mistral-small-2603"].attributes["resolves_to"] == (
            "mistral-small-2603-v2"
        )
        assert "mistral-small-2603-v2" not in models

    async def test_not_listed_there_it_is_asked_about_there(
        self, load_config, endpoints, client, session_factory
    ):
        config = load_config(models=self.MODELS)
        endpoints.listings[GLOBAL] = _listing("mistral-small-latest")
        endpoints.listings[REGIONAL] = _listing()
        endpoints.models[REGIONAL] = {
            "mistral-small-2603": {
                "id": "mistral-small-2603",
                "capabilities": {"completion_chat": True},
            }
        }

        await _refresh(config, session_factory, client)

        assert endpoints.requests[-1] == f"{REGIONAL}/models/mistral-small-2603"
        models = {
            m.model_id for m in await catalog_models(session_factory=session_factory)
        }
        assert "mistral-small-2603" in models

    async def test_gone_there_it_is_gone_whatever_the_provider_lists(
        self, load_config, endpoints, client, session_factory
    ):
        config = load_config(models=self.MODELS)
        endpoints.listings[GLOBAL] = _listing("mistral-small-latest")
        endpoints.listings[REGIONAL] = _listing("mistral-small-2603")
        await _refresh(config, session_factory, client)

        # Still served globally, no longer on its own endpoint.
        endpoints.listings[GLOBAL] = _listing(
            "mistral-small-latest", "mistral-small-2603"
        )
        endpoints.listings[REGIONAL] = _listing()
        endpoints.requests.clear()
        await refresh(
            config,
            client=client,
            session_factory=session_factory,
            now=NOW.replace(minute=1),
        )

        assert f"{REGIONAL}/models/mistral-small-2603" in endpoints.requests
        (event,) = await catalog_events(session_factory=session_factory)
        assert (event.model_id, event.event_type) == ("mistral-small-2603", "removed")
        assert event.details["reason"] == "not_found"

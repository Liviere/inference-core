"""
Unit tests for the listers of providers with an OpenAI-shaped ``/models``.

Covers:
- what each provider's fields become: lifecycle, aliases, prices, capabilities
- the configured ``base_url`` and the default one
- a listing of the wrong shape and a refused key are errors, never empty lists
- looking one model up: found, not found, wrong shape
"""

from datetime import datetime, timezone

import httpx
import pytest

from inference_core.llm.catalog.http import CatalogHTTPError, CatalogResponseError
from inference_core.llm.catalog.listers import openai_compatible as listers
from inference_core.llm.catalog.types import ModelKind, ModelLifecycle

RUNTIME = {"api_key": "sk-test"}


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _serving(payload, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=payload)

    return handler


class TestOpenAI:
    async def test_lists_models_with_shutdown_dates(self):
        seen = []
        payload = {
            "object": "list",
            "data": [
                {"id": "gpt-5", "created": 1754000000, "owned_by": "openai"},
                {"id": "gpt-4.1", "created": 1744000000, "shutdown_date": "2026-12-01"},
                {"id": "text-embedding-3-large", "created": 1704000000},
                {"object": "model"},
                "not an object",
            ],
        }
        async with _client(_serving(payload, seen)) as client:
            models = await listers.list_openai_models(RUNTIME, client)

        assert str(seen[0].url) == "https://api.openai.com/v1/models"
        assert seen[0].headers["authorization"] == "Bearer sk-test"
        assert [m.model_id for m in models] == [
            "gpt-5",
            "gpt-4.1",
            "text-embedding-3-large",
        ]
        current, retiring, embedding = models
        assert current.provider == "openai"
        assert current.lifecycle is ModelLifecycle.ACTIVE
        assert current.kind is ModelKind.CHAT
        assert current.attributes["created_at"].startswith("2025-")
        assert retiring.lifecycle is ModelLifecycle.DEPRECATED
        assert retiring.retires_at == datetime(2026, 12, 1, tzinfo=timezone.utc)
        assert embedding.kind is ModelKind.EMBEDDING

    async def test_uses_the_configured_base_url(self):
        seen = []
        runtime = {"api_key": "k", "base_url": "https://gateway.test/openai/v1/"}
        async with _client(_serving({"data": []}, seen)) as client:
            assert await listers.list_openai_models(runtime, client) == []

        assert str(seen[0].url) == "https://gateway.test/openai/v1/models"

    @pytest.mark.parametrize("payload", [{"object": "list"}, {"data": {}}, [], "x"])
    async def test_wrong_shape_is_an_error(self, payload):
        async with _client(_serving(payload)) as client:
            with pytest.raises(CatalogResponseError):
                await listers.list_openai_models(RUNTIME, client)

    async def test_refused_key_is_an_error(self):
        async with _client(lambda request: httpx.Response(401)) as client:
            with pytest.raises(CatalogHTTPError) as excinfo:
                await listers.list_openai_models(RUNTIME, client)

        assert excinfo.value.status_code == 401

    async def test_bad_ids_are_left_out(self):
        payload = {"data": [{"id": "a b"}, {"id": "x" * 300}, {"id": 7}, {"id": "ok"}]}
        async with _client(_serving(payload)) as client:
            models = await listers.list_openai_models(RUNTIME, client)

        assert [m.model_id for m in models] == ["ok"]

    async def test_looks_one_model_up(self):
        seen = []
        async with _client(_serving({"id": "gpt-5"}, seen)) as client:
            model = await listers.get_openai_model(RUNTIME, client, "gpt-5")

        assert model.model_id == "gpt-5"
        assert str(seen[0].url) == "https://api.openai.com/v1/models/gpt-5"

    async def test_lookup_of_an_unknown_model_is_none(self):
        async with _client(lambda request: httpx.Response(404)) as client:
            assert await listers.get_openai_model(RUNTIME, client, "nope") is None

    async def test_lookup_with_a_wrong_shape_is_an_error(self):
        async with _client(_serving(["gpt-5"])) as client:
            with pytest.raises(CatalogResponseError):
                await listers.get_openai_model(RUNTIME, client, "gpt-5")

    async def test_lookup_keeps_the_id_in_one_path_segment(self):
        seen = []
        async with _client(_serving({"id": "a/b"}, seen)) as client:
            await listers.get_openai_model(RUNTIME, client, "a/../b?x=1")

        assert seen[0].url.raw_path == b"/v1/models/a%2F..%2Fb%3Fx%3D1"


class TestMistral:
    async def test_lists_models_with_aliases_and_deprecation(self):
        seen = []
        payload = {
            "object": "list",
            "data": [
                {
                    "id": "mistral-small-2506",
                    "created": 1756746619,
                    "name": "mistral-small-2506",
                    "max_context_length": 131072,
                    "aliases": ["mistral-small-latest"],
                    "deprecation": None,
                    "capabilities": {
                        "completion_chat": True,
                        "function_calling": True,
                        "vision": True,
                        "reasoning": False,
                    },
                    "type": "base",
                },
                {
                    "id": "mistral-medium-2312",
                    "aliases": [],
                    "deprecation": "2026-11-30T12:00:00Z",
                    "deprecation_replacement_model": "mistral-medium-latest",
                    "capabilities": {"completion_chat": True},
                },
                {"id": "mistral-embed", "capabilities": {"completion_chat": False}},
            ],
        }
        async with _client(_serving(payload, seen)) as client:
            small, medium, embed = await listers.list_mistral_models(RUNTIME, client)

        assert str(seen[0].url) == "https://api.mistral.ai/v1/models"
        assert small.aliases == ("mistral-small-latest",)
        assert small.lifecycle is ModelLifecycle.ACTIVE
        assert small.kind is ModelKind.CHAT
        assert small.attributes["context_window"] == 131072
        assert small.attributes["capabilities"] == {
            "vision": True,
            "tools": True,
            "reasoning": False,
        }
        assert medium.lifecycle is ModelLifecycle.DEPRECATED
        assert medium.deprecated_at == datetime(2026, 11, 30, 12, tzinfo=timezone.utc)
        assert medium.replacement == "mistral-medium-latest"
        assert embed.kind is ModelKind.EMBEDDING


class TestXAI:
    async def test_lists_language_models_with_prices(self):
        seen = []
        payload = {
            "models": [
                {
                    "id": "grok-4.3-0709",
                    "created": 1776556800,
                    "aliases": ["grok-4.3", "grok-4.3-latest"],
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "prompt_text_token_price": 12500,
                    "cached_prompt_text_token_price": 2000,
                    "completion_text_token_price": 25000,
                }
            ]
        }
        async with _client(_serving(payload, seen)) as client:
            (model,) = await listers.list_xai_models(
                {"api_key": "k", "base_url": "https://api.x.ai/v1"}, client
            )

        assert str(seen[0].url) == "https://api.x.ai/v1/language-models"
        assert model.aliases == ("grok-4.3", "grok-4.3-latest")
        assert model.kind is ModelKind.CHAT
        assert model.lifecycle is ModelLifecycle.UNKNOWN
        assert model.attributes["capabilities"] == {"vision": True}
        assert model.attributes["pricing"] == {
            "input": 1.25,
            "output": 2.5,
            "cache_read": 0.2,
        }


class TestCompatibleServer:
    async def test_keeps_only_the_id(self):
        seen = []
        runtime = {"base_url": "http://llm.internal:8000/v1"}
        payload = {"object": "list", "data": [{"id": "local-model", "created": 1}]}
        async with _client(_serving(payload, seen)) as client:
            (model,) = await listers.list_compatible_models(runtime, client)

        assert str(seen[0].url) == "http://llm.internal:8000/v1/models"
        assert "authorization" not in seen[0].headers
        assert model.provider == "custom_openai_compatible"
        assert model.model_id == "local-model"
        assert model.lifecycle is ModelLifecycle.UNKNOWN
        assert model.attributes == {}

"""
Unit tests for the DeepInfra lister.

Covers:
- the provider's own listing: kinds, deprecation with a replacement, prices
- a deprecated model stays in the listing
- looking a model up by its ``owner/name``
"""

from datetime import datetime, timezone

import httpx
import pytest

from inference_core.llm.catalog.http import CatalogResponseError
from inference_core.llm.catalog.listers import deepinfra
from inference_core.llm.catalog.types import ModelKind, ModelLifecycle

LISTING = [
    {
        "model_name": "moonshotai/Kimi-K2.6",
        "type": "text-generation",
        "reported_type": "text-generation",
        "description": "A long description nobody stores.",
        "pricing": {
            "type": "tokens",
            "cents_per_input_token": 3.5e-05,
            "cents_per_output_token": 4e-05,
        },
        "max_tokens": 262144,
        "replaced_by": None,
        "deprecated": None,
        "create_ts": "2026-04-06T20:19:33+00:00",
    },
    {
        "model_name": "moonshotai/Kimi-K2.5",
        "reported_type": "text-generation",
        "pricing": {"type": "tokens", "cents_per_input_token": 2e-05},
        "replaced_by": "moonshotai/Kimi-K2.6",
        "deprecated": 1788824437,
    },
    {
        "model_name": "black-forest-labs/FLUX-2",
        "reported_type": "text-to-image",
        "pricing": {"type": "image_units", "cents_per_input_token": 1},
        "deprecated": None,
    },
    {"model_name": "acme/world-1", "reported_type": "world-model"},
]


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_lists_models():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=LISTING)

    async with _client(handler) as client:
        current, old, image, other = await deepinfra.list_models(
            {"api_key": "k"}, client
        )

    assert str(seen[0].url) == "https://api.deepinfra.com/models/list"
    assert seen[0].headers["authorization"] == "Bearer k"
    assert current.provider == "deepinfra"
    assert current.kind is ModelKind.CHAT
    assert current.lifecycle is ModelLifecycle.ACTIVE
    assert current.attributes == {
        "created_at": "2026-04-06T20:19:33+00:00",
        "context_window": 262144,
        "pricing": {"input": 0.35, "output": 0.4},
    }
    assert old.lifecycle is ModelLifecycle.DEPRECATED
    assert old.deprecated_at == datetime.fromtimestamp(1788824437, tz=timezone.utc)
    assert old.replacement == "moonshotai/Kimi-K2.6"
    assert old.attributes["pricing"] == {"input": 0.2}
    assert image.kind is ModelKind.IMAGE
    assert "pricing" not in image.attributes
    assert other.kind is ModelKind.OTHER


async def test_listing_must_be_a_list():
    async with _client(
        lambda request: httpx.Response(200, json={"data": []})
    ) as client:
        with pytest.raises(CatalogResponseError):
            await deepinfra.list_models({}, client)


async def test_lookup_keeps_the_slash_of_the_name():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json={"model_name": "google/gemma-4-31B-it", "max_output_tokens": 8192}
        )

    async with _client(handler) as client:
        model = await deepinfra.get_model({}, client, "google/gemma-4-31B-it")

    assert seen[0].url.raw_path == b"/models/google/gemma-4-31B-it"
    assert "authorization" not in seen[0].headers
    assert model.attributes == {"max_output_tokens": 8192}


async def test_lookup_of_an_unknown_model_is_none():
    async with _client(lambda request: httpx.Response(404)) as client:
        assert await deepinfra.get_model({}, client, "google/gemma-4-31B-iter") is None

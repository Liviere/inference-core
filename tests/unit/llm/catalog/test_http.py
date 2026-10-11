"""
Unit tests for the catalog's bounded HTTP helper.

Covers:
- a 200 with JSON is returned as parsed
- any other status, an oversized body and a non-JSON body are errors
- an error says the status and never the URL, the key or the body
- a 404 is "no such model" only where one model is looked up
"""

import httpx
import pytest

from inference_core.llm.catalog.http import (
    CatalogHTTPError,
    CatalogResponseError,
    build_client,
    get_json,
    get_json_or_none,
)

URL = "https://provider.test/v1/models"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_returns_parsed_json():
    async with _client(
        lambda request: httpx.Response(200, json={"data": []})
    ) as client:
        assert await get_json(client, URL) == {"data": []}


async def test_sends_headers_and_params():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["query"] = request.url.query.decode()
        return httpx.Response(200, json=[])

    async with _client(handler) as client:
        await get_json(
            client, URL, headers={"Authorization": "Bearer k"}, params={"limit": 5}
        )

    assert seen == {"auth": "Bearer k", "query": "limit=5"}


@pytest.mark.parametrize("status", [301, 401, 403, 404, 429, 500])
async def test_other_status_is_an_error(status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "secret-detail"})

    async with _client(handler) as client:
        with pytest.raises(CatalogHTTPError) as excinfo:
            await get_json(client, URL, headers={"Authorization": "Bearer sk-secret"})

    assert excinfo.value.status_code == status
    assert str(excinfo.value) == f"HTTP {status}"


async def test_oversized_body_is_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"[" + b"1," * 600 + b"1]")

    async with _client(handler) as client:
        with pytest.raises(CatalogResponseError, match="too large"):
            await get_json(client, URL, max_bytes=1000)


async def test_non_json_body_is_an_error():
    async with _client(lambda request: httpx.Response(200, text="<html>")) as client:
        with pytest.raises(CatalogResponseError, match="not JSON"):
            await get_json(client, URL)


async def test_lookup_returns_none_for_404_only():
    async with _client(lambda request: httpx.Response(404)) as client:
        assert await get_json_or_none(client, URL) is None

    async with _client(lambda request: httpx.Response(500)) as client:
        with pytest.raises(CatalogHTTPError):
            await get_json_or_none(client, URL)


async def test_client_does_not_follow_redirects():
    async with build_client() as client:
        assert client.follow_redirects is False
        assert client.timeout.connect == 5.0

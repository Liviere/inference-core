"""
Unit tests for the listers of providers whose listings come in pages:
Anthropic, Gemini and Fireworks.

Covers:
- every page is read and the cursor is passed on
- a listing that promises more without a cursor, or never ends, is an error
- what each provider's fields become
- the API key travels in a header, never in the URL
- looking one model up, including an alias and an id that cannot be one
"""

from datetime import datetime, timezone

import httpx
import pytest

from inference_core.llm.catalog.http import (
    MAX_PAGES,
    CatalogHTTPError,
    CatalogResponseError,
)
from inference_core.llm.catalog.listers import anthropic, fireworks, gemini
from inference_core.llm.catalog.types import ModelKind, ModelLifecycle

RUNTIME = {"api_key": "key-123"}


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _pages(pages, seen):
    """Serve ``pages`` in order, recording each request."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=pages[len(seen) - 1])

    return handler


class TestAnthropic:
    async def test_reads_every_page(self):
        seen = []
        pages = [
            {
                "data": [
                    {
                        "type": "model",
                        "id": "claude-opus-5",
                        "display_name": "Claude Opus 5",
                        "created_at": "2026-07-24T00:00:00Z",
                        "lifecycle": "active",
                        "max_input_tokens": 1000000,
                        "max_tokens": 64000,
                        "capabilities": {
                            "image_input": {"supported": True},
                            "thinking": {"supported": True},
                        },
                    }
                ],
                "has_more": True,
                "last_id": "claude-opus-5",
            },
            {
                "data": [
                    {
                        "id": "claude-sonnet-4-20250514",
                        "lifecycle": "deprecated",
                        "deprecated_at": "2026-08-01T00:00:00Z",
                        "retires_at": "2026-12-15T00:00:00Z",
                    }
                ],
                "has_more": False,
                "last_id": "claude-sonnet-4-20250514",
            },
        ]
        async with _client(_pages(pages, seen)) as client:
            current, old = await anthropic.list_models(RUNTIME, client)

        assert seen[0].url.path == "/v1/models"
        assert seen[0].headers["x-api-key"] == "key-123"
        assert seen[0].headers["anthropic-version"] == "2023-06-01"
        assert "after_id" not in seen[0].url.params
        assert seen[1].url.params["after_id"] == "claude-opus-5"
        assert current.provider == "claude"
        assert current.lifecycle is ModelLifecycle.ACTIVE
        assert current.kind is ModelKind.CHAT
        assert current.attributes == {
            "display_name": "Claude Opus 5",
            "created_at": "2026-07-24T00:00:00+00:00",
            "context_window": 1000000,
            "max_output_tokens": 64000,
            "capabilities": {"vision": True, "reasoning": True},
        }
        assert old.lifecycle is ModelLifecycle.DEPRECATED
        assert old.retires_at == datetime(2026, 12, 15, tzinfo=timezone.utc)

    async def test_more_pages_without_a_cursor_is_an_error(self):
        pages = [{"data": [{"id": "claude-opus-5"}], "has_more": True}]
        async with _client(_pages(pages, [])) as client:
            with pytest.raises(CatalogResponseError, match="no cursor"):
                await anthropic.list_models(RUNTIME, client)

    async def test_a_listing_that_never_ends_is_an_error(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(
                200, json={"data": [{"id": "m"}], "has_more": True, "last_id": "m"}
            )

        async with _client(handler) as client:
            with pytest.raises(CatalogResponseError, match="does not end"):
                await anthropic.list_models(RUNTIME, client)

        assert len(seen) == MAX_PAGES

    async def test_a_failed_page_fails_the_listing(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if len(seen) == 1:
                return httpx.Response(
                    200, json={"data": [{"id": "m"}], "has_more": True, "last_id": "m"}
                )
            return httpx.Response(529)

        async with _client(handler) as client:
            with pytest.raises(CatalogHTTPError):
                await anthropic.list_models(RUNTIME, client)

    async def test_lookup_resolves_an_alias(self):
        seen = []
        pages = [{"id": "claude-haiku-4-5-20251001", "lifecycle": "active"}]
        async with _client(_pages(pages, seen)) as client:
            model = await anthropic.get_model(RUNTIME, client, "claude-haiku-4-5")

        assert seen[0].url.path == "/v1/models/claude-haiku-4-5"
        assert model.model_id == "claude-haiku-4-5-20251001"

    async def test_lookup_of_an_unknown_model_is_none(self):
        async with _client(lambda request: httpx.Response(404)) as client:
            assert await anthropic.get_model(RUNTIME, client, "claude-0") is None


class TestGemini:
    async def test_reads_every_page_and_strips_the_prefix(self):
        seen = []
        pages = [
            {
                "models": [
                    {
                        "name": "models/gemini-3-flash-preview",
                        "displayName": "Gemini 3 Flash Preview",
                        "inputTokenLimit": 1048576,
                        "outputTokenLimit": 65536,
                        "supportedGenerationMethods": [
                            "generateContent",
                            "countTokens",
                        ],
                        "thinking": True,
                    }
                ],
                "nextPageToken": "page-2",
            },
            {
                "models": [
                    {
                        "name": "models/gemini-embedding-001",
                        "supportedGenerationMethods": ["embedContent"],
                    }
                ]
            },
        ]
        async with _client(_pages(pages, seen)) as client:
            flash, embedding = await gemini.list_models(RUNTIME, client)

        assert seen[0].url.path == "/v1beta/models"
        assert seen[0].headers["x-goog-api-key"] == "key-123"
        assert "key" not in seen[0].url.params
        assert "key-123" not in str(seen[0].url)
        assert seen[1].url.params["pageToken"] == "page-2"
        assert flash.model_id == "gemini-3-flash-preview"
        assert flash.kind is ModelKind.CHAT
        assert flash.lifecycle is ModelLifecycle.UNKNOWN
        assert flash.attributes == {
            "display_name": "Gemini 3 Flash Preview",
            "context_window": 1048576,
            "max_output_tokens": 65536,
            "capabilities": {"reasoning": True},
        }
        assert embedding.kind is ModelKind.EMBEDDING

    async def test_an_empty_last_page_may_leave_the_list_out(self):
        pages = [{"models": [{"name": "models/m"}], "nextPageToken": "t"}, {}]
        async with _client(_pages(pages, [])) as client:
            models = await gemini.list_models(RUNTIME, client)

        assert [m.model_id for m in models] == ["m"]

    async def test_models_of_the_wrong_type_is_an_error(self):
        async with _client(_pages([{"models": "none"}], [])) as client:
            with pytest.raises(CatalogResponseError):
                await gemini.list_models(RUNTIME, client)

    async def test_lookup(self):
        seen = []
        pages = [{"name": "models/gemini-flash-latest"}]
        async with _client(_pages(pages, seen)) as client:
            model = await gemini.get_model(RUNTIME, client, "gemini-flash-latest")

        assert seen[0].url.path == "/v1beta/models/gemini-flash-latest"
        assert model.model_id == "gemini-flash-latest"


class TestFireworks:
    async def test_reads_every_page_of_the_public_account(self):
        seen = []
        pages = [
            {
                "models": [
                    {
                        "name": "accounts/fireworks/models/kimi-k2p6",
                        "displayName": "Kimi K2.6",
                        "createTime": "2026-03-01T10:00:00Z",
                        "kind": "HF_BASE_MODEL",
                        "contextLength": 262144,
                        "supportsImageInput": False,
                        "supportsTools": True,
                        "supportsServerless": True,
                        "conversationConfig": {"style": "chat"},
                    }
                ],
                "nextPageToken": "next",
                "totalSize": 2,
            },
            {
                "models": [
                    {
                        "name": "accounts/fireworks/models/kimi-k2p5",
                        "deprecationDate": {"year": 2026, "month": 12, "day": 1},
                    }
                ]
            },
        ]
        async with _client(_pages(pages, seen)) as client:
            current, old = await fireworks.list_models(RUNTIME, client)

        assert seen[0].url.path == "/v1/accounts/fireworks/models"
        assert seen[0].headers["authorization"] == "Bearer key-123"
        assert seen[1].url.params["pageToken"] == "next"
        assert current.kind is ModelKind.CHAT
        assert current.lifecycle is ModelLifecycle.ACTIVE
        assert current.attributes["capabilities"] == {"vision": False, "tools": True}
        assert current.attributes["serverless"] is True
        assert "serverless" not in old.attributes
        assert old.lifecycle is ModelLifecycle.DEPRECATED
        assert old.retires_at == datetime(2026, 12, 1, tzinfo=timezone.utc)

    async def test_an_unreadable_date_is_no_date(self):
        pages = [
            {
                "models": [
                    {
                        "name": "accounts/fireworks/models/m",
                        "deprecationDate": {"year": 0, "month": 0, "day": 0},
                    }
                ]
            }
        ]
        async with _client(_pages(pages, [])) as client:
            (model,) = await fireworks.list_models(RUNTIME, client)

        assert model.retires_at is None
        assert model.lifecycle is ModelLifecycle.ACTIVE

    async def test_lookup_of_another_accounts_model(self):
        seen = []
        pages = [{"name": "accounts/acme/models/tuned-1"}]
        async with _client(_pages(pages, seen)) as client:
            model = await fireworks.get_model(
                RUNTIME, client, "accounts/acme/models/tuned-1"
            )

        assert seen[0].url.path == "/v1/accounts/acme/models/tuned-1"
        assert model.model_id == "accounts/acme/models/tuned-1"

    @pytest.mark.parametrize(
        "model_id",
        ["kimi-k2p6", "accounts/a/models/../../keys", "accounts/a/models/m?x"],
    )
    async def test_lookup_of_something_that_is_no_model_id_asks_nothing(self, model_id):
        seen = []
        async with _client(_pages([{}], seen)) as client:
            assert await fireworks.get_model(RUNTIME, client, model_id) is None

        assert seen == []

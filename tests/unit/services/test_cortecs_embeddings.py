"""Embeddings through the Cortecs gateway, their price and their usage reports.

Covers:
- the request the OpenAI-compatible client sends: texts as they are, floats,
  the provider's request options, no dimensions, the configured address
- batches, the vectors in order, the usage added up
- the remote backend for cortecs: its address and key from the config only
- the entry's pricing, scaled by the provider's price multiplier
- the usage hooks: one report per call, never the texts, an estimate where
  the provider says nothing, a failure reported and re-raised, a failing hook
  that does not fail the call, nothing measured without a hook
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest
import yaml

from inference_core.services.embedding_service import (
    EmbeddingService,
    FakeDeterministicBackend,
    RemoteLangChainBackend,
)
from inference_core.services.embedding_usage import (
    EmbeddingUsage,
    ReportedUsage,
    estimate_tokens,
    register_embedding_usage_hook,
    unregister_embedding_usage_hook,
)
from inference_core.services.openai_compatible_embeddings import (
    OpenAICompatibleEmbeddings,
)

BASE_URL = "https://gateway.cortecs.example/v1"
SECRET_TEXT = "Wymyślona notatka o spotkaniu w czwartek"


def _embedding_response(request: httpx.Request, cost: float = 0.5):
    body = json.loads(request.content)
    texts = body["input"]
    return httpx.Response(
        200,
        json={
            "object": "list",
            "model": "bge-m3",
            "provider": "ovh",
            # Out of order on purpose: the client sorts by index.
            "data": [
                {"object": "embedding", "index": i, "embedding": [float(i), 1.0]}
                for i in reversed(range(len(texts)))
            ],
            "usage": {
                "prompt_tokens": 3 * len(texts),
                "completion_tokens": 0,
                "total_tokens": 3 * len(texts),
                "cost": cost,
            },
        },
    )


def _client(seen, **kwargs):
    from openai import AsyncOpenAI, OpenAI

    def handler(request):
        seen.append(request)
        return _embedding_response(request)

    client = OpenAICompatibleEmbeddings(
        model="bge-m3",
        base_url=BASE_URL,
        api_key="test-key",
        request_options={"eu_native": True, "allow_zero_data_retention": True},
        **kwargs,
    )
    client._client = OpenAI(
        api_key="test-key",
        base_url=BASE_URL,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    async def ahandler(request):
        return handler(request)

    client._async_client = AsyncOpenAI(
        api_key="test-key",
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(ahandler)),
    )
    return client


class TestOpenAICompatibleClient:
    def test_sends_the_texts_with_the_request_options(self):
        seen = []

        vectors, usage = _client(seen).embed_with_usage(["a", "b"])

        (request,) = seen
        assert str(request.url) == f"{BASE_URL}/embeddings"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body == {
            "model": "bge-m3",
            "input": ["a", "b"],
            "encoding_format": "float",
            "eu_native": True,
            "allow_zero_data_retention": True,
        }
        assert "dimensions" not in body
        assert vectors == [[0.0, 1.0], [1.0, 1.0]]
        assert usage == ReportedUsage(
            input_tokens=6, reported_model="bge-m3", provider_cost=0.5
        )

    def test_batches_and_adds_the_usage_up(self):
        seen = []

        vectors, usage = _client(seen, batch_size=2).embed_with_usage(
            ["a", "b", "c", "d", "e"]
        )

        assert [len(json.loads(r.content)["input"]) for r in seen] == [2, 2, 1]
        assert len(vectors) == 5
        assert vectors[2] == [0.0, 1.0]
        assert usage.input_tokens == 15
        assert usage.provider_cost == pytest.approx(1.5)

    async def test_async_too(self):
        seen = []

        vectors, usage = await _client(seen).aembed_with_usage(["a"])

        assert vectors == [[0.0, 1.0]]
        assert usage.input_tokens == 3
        assert json.loads(seen[0].content)["eu_native"] is True

    def test_langchain_interface(self):
        client = _client([])

        assert client.embed_documents(["a", "b"]) == [[0.0, 1.0], [1.0, 1.0]]
        assert client.embed_query("a") == [0.0, 1.0]

    def test_a_short_answer_is_an_error(self):
        from openai import OpenAI

        client = _client([])

        def short(request):
            response = _embedding_response(request)
            payload = json.loads(response.content)
            payload["data"] = payload["data"][:1]
            return httpx.Response(200, json=payload)

        client._client = OpenAI(
            api_key="k",
            base_url=BASE_URL,
            http_client=httpx.Client(transport=httpx.MockTransport(short)),
        )
        with pytest.raises(ValueError, match="1 vectors for 2 texts"):
            client.embed_documents(["a", "b"])

    def test_needs_an_address_and_a_key(self):
        with pytest.raises(ValueError, match="base_url"):
            OpenAICompatibleEmbeddings(model="m", base_url="", api_key="k")
        with pytest.raises(ValueError, match="API key"):
            OpenAICompatibleEmbeddings(model="m", base_url=BASE_URL, api_key=None)


def _llm_config(tmp_path, monkeypatch, embedding=None, provider=None):
    from inference_core.llm.config import LLMConfig

    provider_entry = {
        "name": "Cortecs",
        "base_url": BASE_URL,
        "api_key_env": "TEST_CORTECS_KEY",
        "requires_api_key": True,
        "price_multiplier": 1.2,
        "request_options": {"eu_native": True},
    }
    provider_entry.update(provider or {})
    embedding_entry = {
        "provider": "cortecs",
        "model": "bge-m3",
        "dimensions": 1024,
        "pricing": {"currency": "USD", "input": {"cost_per_1m": 0.01}},
    }
    embedding_entry.update(embedding or {})
    path = tmp_path / "llm_config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "providers": {"cortecs": provider_entry},
                "models": {},
                "embeddings": {"default": embedding_entry},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_CONFIG_PATH", str(path))
    return LLMConfig()


class TestCortecsBackend:
    def _backend(self, monkeypatch, config):
        monkeypatch.setattr("inference_core.llm.config.get_llm_config", lambda: config)
        settings = MagicMock()
        settings.embedding_backend = "remote"
        return RemoteLangChainBackend(settings)

    def test_reads_its_address_key_and_options_from_the_config(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TEST_CORTECS_KEY", "test-key")
        backend = self._backend(monkeypatch, _llm_config(tmp_path, monkeypatch))

        client = backend._embeddings
        assert isinstance(client, OpenAICompatibleEmbeddings)
        assert client.base_url == BASE_URL
        assert client._extra_body == {"eu_native": True}
        assert (backend.provider_name, backend.model_name) == ("cortecs", "bge-m3")
        assert backend.get_dimension() == 1024

    def test_the_entrys_own_address_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_CORTECS_KEY", "test-key")
        config = _llm_config(
            tmp_path, monkeypatch, embedding={"base_url": "https://eu.example/v1"}
        )

        backend = self._backend(monkeypatch, config)

        assert backend._embeddings.base_url == "https://eu.example/v1"

    def test_without_a_key_it_says_which(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TEST_CORTECS_KEY", raising=False)
        config = _llm_config(tmp_path, monkeypatch)

        with pytest.raises(ValueError, match="TEST_CORTECS_KEY"):
            self._backend(monkeypatch, config)

    def test_without_an_address_it_fails(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_CORTECS_KEY", "test-key")
        config = _llm_config(tmp_path, monkeypatch, provider={"base_url": None})

        with pytest.raises(ValueError, match="base_url"):
            self._backend(monkeypatch, config)

    def test_its_price_is_scaled_by_the_providers_multiplier(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TEST_CORTECS_KEY", "test-key")
        backend = self._backend(monkeypatch, _llm_config(tmp_path, monkeypatch))

        assert backend.pricing.input.cost_per_1k == pytest.approx(0.000012)
        assert backend.pricing.output.cost_per_1k == 0.0

    def test_the_entrys_own_multiplier_wins(self, tmp_path, monkeypatch):
        config = _llm_config(tmp_path, monkeypatch, embedding={"price_multiplier": 1})

        pricing = config.get_embedding_config("default").pricing

        assert pricing.input.cost_per_1k == pytest.approx(0.00001)


class TestEmbeddingConfig:
    def test_fireworks_and_cortecs_are_embedding_providers(self):
        from inference_core.llm.config import EmbeddingConfig

        for provider in ("fireworks", "cortecs"):
            assert EmbeddingConfig(provider=provider, model="m").provider.value == (
                provider
            )

    def test_without_pricing_there_is_no_price(self):
        from inference_core.llm.config import EmbeddingConfig

        assert EmbeddingConfig(provider="cortecs", model="m").pricing is None


@pytest.fixture
def reports():
    seen: list[EmbeddingUsage] = []
    register_embedding_usage_hook(seen.append)
    yield seen
    unregister_embedding_usage_hook(seen.append)


def _fake_service():
    settings = MagicMock()
    settings.vector_dim = 8
    return EmbeddingService(backend=FakeDeterministicBackend(settings))


class _ReportingBackend(FakeDeterministicBackend):
    backend_name = "remote"
    provider_name = "cortecs"
    model_name = "bge-m3"

    def embed_texts_with_usage(self, texts):
        return self.embed_texts(texts), ReportedUsage(
            input_tokens=42, reported_model="bge-m3", provider_cost=0.25
        )

    async def aembed_texts_with_usage(self, texts):
        return self.embed_texts_with_usage(texts)


class _FailingBackend(FakeDeterministicBackend):
    def embed_texts_with_usage(self, texts):
        raise RuntimeError(SECRET_TEXT)


class TestUsageHooks:
    def test_without_a_hook_the_backend_is_called_as_before(self):
        backend = MagicMock()
        backend.embed_texts.return_value = [[1.0]]
        service = EmbeddingService(backend=backend)

        assert service.embed_texts(["a"]) == [[1.0]]
        backend.embed_texts.assert_called_once_with(["a"])
        backend.embed_texts_with_usage.assert_not_called()

    def test_one_report_per_call_without_the_texts(self, reports):
        _fake_service().embed_texts([SECRET_TEXT, "druga"])

        (usage,) = reports
        assert (usage.backend, usage.provider, usage.texts, usage.success) == (
            "fake",
            "fake",
            2,
            True,
        )
        assert usage.input_tokens and usage.estimated is True
        assert SECRET_TEXT not in repr(usage)
        assert "druga" not in repr(usage)

    def test_the_providers_count_is_taken_as_it_is(self, reports):
        settings = MagicMock()
        settings.vector_dim = 4
        EmbeddingService(backend=_ReportingBackend(settings)).embed_texts(["a"])

        (usage,) = reports
        assert (usage.provider, usage.model, usage.input_tokens) == (
            "cortecs",
            "bge-m3",
            42,
        )
        assert usage.estimated is False
        assert usage.provider_cost == 0.25

    async def test_async_calls_and_queries_are_reported(self, reports):
        service = _fake_service()

        await service.aembed_texts(["a"])
        await service.aembed_query("b")
        service.embed_query("c")

        assert [usage.texts for usage in reports] == [1, 1, 1]

    def test_the_stores_embed_function_is_reported(self, reports):
        _fake_service().get_embed_fn()(["a", "b", "c"])

        (usage,) = reports
        assert usage.texts == 3

    def test_a_failure_is_reported_and_raised(self, reports):
        settings = MagicMock()
        settings.vector_dim = 4
        service = EmbeddingService(backend=_FailingBackend(settings))

        with pytest.raises(RuntimeError):
            service.embed_texts(["a"])

        (usage,) = reports
        assert (usage.success, usage.error_type, usage.input_tokens) == (
            False,
            "RuntimeError",
            None,
        )
        assert SECRET_TEXT not in repr(usage)

    def test_a_failing_hook_does_not_fail_the_call(self, caplog):
        def broken(usage):
            raise ValueError(SECRET_TEXT)

        register_embedding_usage_hook(broken)
        try:
            vectors = _fake_service().embed_texts(["a"])
        finally:
            unregister_embedding_usage_hook(broken)

        assert len(vectors) == 1
        assert "ValueError" in caplog.text
        assert SECRET_TEXT not in caplog.text

    def test_a_duck_typed_backend_is_reported_with_an_estimate(self, reports):
        backend = MagicMock()
        backend.embed_texts.return_value = [[1.0]]

        assert EmbeddingService(backend=backend).embed_texts(["abc"]) == [[1.0]]

        (usage,) = reports
        assert usage.estimated is True

    def test_registering_twice_reports_once(self, reports):
        register_embedding_usage_hook(reports.append)

        _fake_service().embed_texts(["a"])

        assert len(reports) == 1


def test_estimate_counts_only_texts():
    assert estimate_tokens([]) is None
    assert estimate_tokens([None]) is None
    assert estimate_tokens(["hello world"]) >= 1

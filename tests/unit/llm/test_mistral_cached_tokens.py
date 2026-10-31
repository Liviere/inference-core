"""What a Mistral response read from the prompt cache reaches its price.

``USAGE`` is what Mistral's API answered, in a plain response and in a stream's
last chunk alike, on a second call of a prompt of 8,380 tokens (recorded
2026-10-31). The model runs on the library's real client over a mocked
transport, so a change of ``langchain-mistralai`` fails here. The prices are
those of Mistral Small: 0.15 per million of input, 0.015 of cached input, 0.6 of
output.
"""

import json
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI

from inference_core.agents.middleware.cost_tracking import CostTrackingMiddleware
from inference_core.llm.config import ModelConfig, ModelProvider, PricingConfig
from inference_core.llm.mistral_usage import ChatMistralAICachedTokens
from inference_core.llm.models import LLMModelFactory
from inference_core.llm.usage_logging import PricingCalculator, UsageSession

BASE_URL = "https://api.mistral.ai/v1"

USAGE = {
    "prompt_tokens": 8380,
    "total_tokens": 8382,
    "completion_tokens": 2,
    "prompt_tokens_details": {"cached_tokens": 8352},
    "completion_tokens_details": {"reasoning_tokens": 0},
    "service_tier": "standard",
}
NOTHING_READ = {**USAGE, "prompt_tokens_details": {"cached_tokens": 0}}

PRICING = {
    "input": {"cost_per_1m": 0.15},
    "output": {"cost_per_1m": 0.6},
    "extras": {"cache_read_tokens": {"cost_per_1m": 0.015}},
    "extras_policy": {"breakdown_keys": ["total_tokens", "reasoning_tokens"]},
}
NO_CACHE_PRICE = {key: PRICING[key] for key in ("input", "output", "extras_policy")}


@pytest.fixture(autouse=True)
def _real_models(monkeypatch):
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled", lambda: False
    )


def _transport(usage: dict[str, Any]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            chunks = [
                {
                    "id": "c",
                    "model": "mistral-small-latest",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "OK"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "c",
                    "model": "mistral-small-latest",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": ""},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                },
            ]
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            return httpx.Response(
                200,
                content=(body + "data: [DONE]\n\n").encode(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "id": "c",
                "model": "mistral-small-latest",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "OK"},
                    }
                ],
                "usage": usage,
            },
        )

    return httpx.MockTransport(handler)


def _model(pricing: Any, usage: dict[str, Any] = USAGE) -> ChatMistralAI:
    config = ModelConfig(
        name="mistral-small-latest",
        provider=ModelProvider.MISTRAL,
        api_key="test-key",
        pricing=PricingConfig(**pricing) if pricing else None,
    )
    llm_config = MagicMock()
    llm_config.enable_caching = False
    llm_config.get_model_config.return_value = config
    model = LLMModelFactory(llm_config).create_model(config.name)
    transport = _transport(usage)
    model.client = httpx.Client(base_url=BASE_URL, transport=transport)
    model.async_client = httpx.AsyncClient(base_url=BASE_URL, transport=transport)
    return model


def _plain_call(model: ChatMistralAI) -> Any:
    # The factory streams; a copy that does not asks for a plain response.
    return model.model_copy(update={"streaming": False}).invoke(
        [HumanMessage(content="Say OK.")]
    )


def _stream_call(model: ChatMistralAI) -> Any:
    chunks = list(model.stream([HumanMessage(content="Say OK.")]))
    return sum(chunks[1:], chunks[0])


@pytest.mark.parametrize("call", [_plain_call, _stream_call])
def test_the_usage_names_what_was_read(call):
    usage = call(_model(PRICING)).usage_metadata

    assert usage["input_tokens"] == 8380
    assert usage["input_token_details"] == {"cache_read": 8352}


@pytest.mark.asyncio
async def test_an_async_stream_names_it_too():
    model = _model(PRICING)
    chunks = [chunk async for chunk in model.astream([HumanMessage(content="Hi")])]

    assert sum(chunks[1:], chunks[0]).usage_metadata["input_token_details"] == {
        "cache_read": 8352
    }


@pytest.mark.asyncio
async def test_a_call_the_factory_streams_names_it():
    """``ainvoke`` on a streaming model, as an agent's call is made."""
    message = await _model(PRICING).ainvoke([HumanMessage(content="Hi")])

    assert message.content == "OK"
    assert message.usage_metadata["input_token_details"] == {"cache_read": 8352}


@pytest.mark.parametrize("call", [_plain_call, _stream_call])
def test_nothing_read_is_not_named(call):
    usage = call(_model(PRICING, NOTHING_READ)).usage_metadata

    assert "input_token_details" not in usage


def _cost(message: Any, pricing: PricingConfig) -> tuple[dict, dict]:
    fragment, extra = CostTrackingMiddleware._extract_usage_fragment(
        message.usage_metadata
    )
    return extra, PricingCalculator.compute_cost(fragment, pricing)


@pytest.mark.asyncio
async def test_the_log_row_has_the_read_and_its_price():
    fragment, _extra = CostTrackingMiddleware._extract_usage_fragment(
        _stream_call(_model(PRICING)).usage_metadata
    )
    added = []
    db = MagicMock()
    db.add = added.append
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    # The price's snapshot is there already.
    found = MagicMock()
    found.scalars.return_value.first.return_value = MagicMock(id=uuid4())
    db.execute = AsyncMock(return_value=found)

    @asynccontextmanager
    async def fake_session():
        yield db

    session = UsageSession(
        task_type="agent",
        request_mode="sync",
        model_name="mistral-small-latest",
        provider="mistral",
        pricing_config=PricingConfig(**PRICING),
    )
    with patch("inference_core.llm.usage_logging.get_async_session", fake_session):
        await session.finalize(success=True, final_usage=fragment)

    [row] = added
    assert row.extra_tokens["cache_read_tokens"] == 8352
    assert row.cost_extras_usd == pytest.approx(8352 * 0.015 / 1e6, abs=1e-6)
    assert row.cost_total_usd == pytest.approx(0.000131, abs=1e-6)
    assert row.cost_estimated is False


def test_a_read_costs_a_tenth():
    """28 uncached at 0.15, 8,352 read at 0.015, 2 written out at 0.6; in full,
    the same call cost 0.001258 USD."""
    extra, cost = _cost(_stream_call(_model(PRICING)), PricingConfig(**PRICING))

    assert extra == {"cache_read_tokens": 8352}
    assert cost["cost_input_usd"] == pytest.approx(28 * 0.15 / 1e6)
    assert cost["extra_costs"]["cache_read_tokens"] == pytest.approx(8352 * 0.015 / 1e6)
    assert cost["cost_total_usd"] == pytest.approx(0.000131, abs=1e-6)
    assert cost["cost_estimated"] is False


def test_the_regions_multiplier_scales_a_read_too():
    pricing = PricingConfig(**PRICING).scaled(1.1)
    _extra, cost = _cost(_stream_call(_model(PRICING)), pricing)

    assert cost["extra_costs"]["cache_read_tokens"] == pytest.approx(
        8352 * 0.015 * 1.1 / 1e6
    )


@pytest.mark.parametrize("pricing", [NO_CACHE_PRICE, None])
def test_a_model_without_a_cache_price_is_as_it_was(pricing):
    model = _model(pricing)

    assert type(model) is ChatMistralAI
    usage = _stream_call(model).usage_metadata
    assert "input_token_details" not in usage
    if pricing:
        _extra, cost = _cost(_stream_call(model), PricingConfig(**pricing))
        assert cost["cost_input_usd"] == pytest.approx(8380 * 0.15 / 1e6)
        assert cost["cost_estimated"] is False


def test_a_model_with_a_cache_price_reports_reads():
    assert type(_model(PRICING)) is ChatMistralAICachedTokens

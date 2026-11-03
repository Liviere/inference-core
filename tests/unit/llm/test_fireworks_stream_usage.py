"""A streamed Fireworks response names its usage when the entry asks for it.

``tests/fixtures/fireworks_stream_usage.json`` is a stream Fireworks sent
(MiniMax M3, 2026-11-03): the second call of a prompt of 6,926 tokens with the
same ``prompt_cache_key``, asked with ``stream_options.include_usage``. Its text
is replaced by filler of the same length and it carries no identifier; the
shape and the numbers are the provider's. Its last chunk has no ``choices``
and only ``usage``, with 6,919 tokens read from the cache and 315 of 321
output tokens of reasoning.

The model runs on the Fireworks SDK's real client over a mocked transport, so
what goes on the wire is what is checked. Without ``streamed_usage`` the model
behaves as it did before the option: nothing asked for, nothing read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fireworks import AsyncFireworks, Fireworks
from langchain_core.messages import HumanMessage

from inference_core.agents.middleware.cost_tracking import CostTrackingMiddleware
from inference_core.llm.chat_fireworks import ChatFireworksReasoning
from inference_core.llm.config import ModelConfig, ModelProvider, PricingConfig
from inference_core.llm.models import LLMModelFactory
from inference_core.llm.usage_logging import PricingCalculator

NAME = "accounts/fireworks/models/minimax-m3"
RECORDING = Path(__file__).parents[2] / "fixtures" / "fireworks_stream_usage.json"
CHUNKS: list[dict[str, Any]] = json.loads(RECORDING.read_text())
USAGE = CHUNKS[-1]["usage"]

#: MiniMax M3 on Fireworks: 0.30 per million of input, 0.06 of cached input,
#: 1.20 of output.
PRICING = {
    "input": {"cost_per_1m": 0.30},
    "output": {"cost_per_1m": 1.20},
    "extras": {"cache_read_tokens": {"cost_per_1m": 0.06}},
    "extras_policy": {"breakdown_keys": ["total_tokens", "reasoning_tokens"]},
}
NO_CACHE_PRICE = {key: PRICING[key] for key in ("input", "output", "extras_policy")}


@pytest.fixture(autouse=True)
def _real_models(monkeypatch):
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled", lambda: False
    )


class _Wire:
    """A mocked transport that answers with the recording and keeps every
    request's JSON body."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if body.get("stream"):
            # Only a stream that asked for usage gets its last chunk.
            asked = (body.get("stream_options") or {}).get("include_usage")
            chunks = CHUNKS if asked else CHUNKS[:-1]
            sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            return httpx.Response(
                200,
                content=(sse + "data: [DONE]\n\n").encode(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-recorded",
                "object": "chat.completion",
                "created": 0,
                "model": NAME,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Ashford",
                            "reasoning_content": "xxxx",
                        },
                    }
                ],
                "usage": USAGE,
            },
        )


def _model(
    *, streamed_usage: bool, pricing: Any = PRICING
) -> tuple[ChatFireworksReasoning, _Wire]:
    config = ModelConfig(
        name=NAME,
        provider=ModelProvider.FIREWORKS,
        api_key="test-key",
        streamed_usage=streamed_usage,
        pricing=PricingConfig(**pricing) if pricing else None,
    )
    llm_config = MagicMock()
    llm_config.enable_caching = False
    llm_config.get_model_config.return_value = config
    model = LLMModelFactory(llm_config).create_model(NAME)
    assert isinstance(model, ChatFireworksReasoning)
    wire = _Wire()
    transport = httpx.MockTransport(wire.handler)
    model.client = Fireworks(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(transport=transport),
    ).chat.completions
    model.async_client = AsyncFireworks(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=transport),
    ).chat.completions
    return model, wire


def _streamed(model: ChatFireworksReasoning, **kwargs: Any) -> Any:
    chunks = list(model.stream([HumanMessage(content="Which town?")], **kwargs))
    return sum(chunks[1:], chunks[0])


def _plain(model: ChatFireworksReasoning) -> Any:
    # The factory streams; a copy that does not asks for a plain response.
    return model.model_copy(update={"streaming": False}).invoke(
        [HumanMessage(content="Which town?")]
    )


# --------------------------------------------------------------------------- #
# Without the option: as before
# --------------------------------------------------------------------------- #


def test_without_the_option_nothing_is_asked_for():
    model, wire = _model(streamed_usage=False)

    message = _streamed(model)

    assert "stream_options" not in wire.bodies[0]
    assert model.include_stream_usage is False
    assert message.usage_metadata is None
    assert message.content and set(message.content) == {"x"}
    assert message.additional_kwargs["reasoning_content"]


def test_without_the_option_a_usage_chunk_is_still_skipped():
    """Even a stream that carries usage (asked for by the call itself)."""
    model, wire = _model(streamed_usage=False)

    message = _streamed(model, extra_body={"stream_options": {"include_usage": True}})

    assert wire.bodies[0]["stream_options"] == {"include_usage": True}
    assert message.usage_metadata is None


def test_without_the_option_a_plain_response_has_no_breakdown():
    model, _wire = _model(streamed_usage=False)

    usage = _plain(model).usage_metadata

    assert usage == {"input_tokens": 6926, "output_tokens": 321, "total_tokens": 7247}


# --------------------------------------------------------------------------- #
# With the option
# --------------------------------------------------------------------------- #


def test_the_stream_asks_for_usage_in_the_body():
    model, wire = _model(streamed_usage=True)

    _streamed(model)

    assert wire.bodies[0]["stream_options"] == {"include_usage": True}
    assert wire.bodies[0]["stream"] is True


def test_a_call_s_own_stream_options_win():
    model, wire = _model(streamed_usage=True)

    _streamed(model, stream_options={"include_usage": False})

    assert wire.bodies[0]["stream_options"] == {"include_usage": False}


def test_the_stream_names_its_usage_reads_and_reasoning():
    model, _wire = _model(streamed_usage=True)

    message = _streamed(model)

    assert message.usage_metadata == {
        "input_tokens": 6926,
        "output_tokens": 321,
        "total_tokens": 7247,
        "input_token_details": {"cache_read": 6919},
        "output_token_details": {"reasoning": 315},
    }
    # The reasoning and the answer are where they were.
    assert len(message.additional_kwargs["reasoning_content"]) > 100
    assert message.content


@pytest.mark.asyncio
async def test_an_async_stream_names_it_too():
    model, wire = _model(streamed_usage=True)

    message = await model.ainvoke([HumanMessage(content="Which town?")])

    assert wire.bodies[0]["stream_options"] == {"include_usage": True}
    assert message.usage_metadata["input_token_details"] == {"cache_read": 6919}
    assert message.usage_metadata["output_tokens"] == 321


def test_a_plain_response_names_reads_and_reasoning():
    model, _wire = _model(streamed_usage=True)

    usage = _plain(model).usage_metadata

    assert usage["input_token_details"] == {"cache_read": 6919}
    assert usage["output_token_details"] == {"reasoning": 315}


@pytest.mark.parametrize("call", [_streamed, _plain])
def test_reads_without_a_price_are_not_named(call):
    """Cache reads without a price would mark the cost estimated."""
    model, _wire = _model(streamed_usage=True, pricing=NO_CACHE_PRICE)

    usage = call(model).usage_metadata

    assert "input_token_details" not in usage
    assert usage["output_token_details"] == {"reasoning": 315}


def test_the_cost_has_the_reads_at_their_price():
    model, _wire = _model(streamed_usage=True)

    fragment, _extra = CostTrackingMiddleware._extract_usage_fragment(
        _streamed(model).usage_metadata
    )
    cost = PricingCalculator.compute_cost(fragment, PricingConfig(**PRICING))

    assert fragment["cache_read_tokens"] == 6919
    assert cost["cost_estimated"] is False
    # 7 input at 0.30, 6,919 read at 0.06, 321 output at 1.20 (per million).
    assert cost["cost_total_usd"] == pytest.approx(
        (7 * 0.30 + 6919 * 0.06 + 321 * 1.20) / 1e6, rel=1e-6
    )


def test_another_provider_s_model_ignores_the_option(caplog):
    config = ModelConfig(
        name="gpt-x",
        provider=ModelProvider.OPENAI,
        api_key="test-key",
        streamed_usage=True,
    )
    llm_config = MagicMock()
    llm_config.enable_caching = False
    llm_config.get_model_config.return_value = config

    with caplog.at_level("WARNING"):
        model = LLMModelFactory(llm_config).create_model("gpt-x")

    assert model is not None
    assert not hasattr(model, "include_stream_usage")
    assert "streamed_usage is ignored for model 'gpt-x'" in caplog.text

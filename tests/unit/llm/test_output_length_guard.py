"""A model whose provider ignores the output limit reads its stream only up to it.

The provider here is a mocked transport that, like the Cortecs gateway on
2026-11-03 (Mistral Small: 1,816 tokens for a limit of 60; Mistral Large 4:
5,272 for 400, 2,476 of them reasoning), streams far past ``max_tokens``: some
``reasoning_content`` deltas, then ``content`` deltas, ``finish_reason: stop``
and a last chunk with only ``usage``. Each delta carries one word, which
``cl100k_base`` counts as one token. The body is a generator, so a test sees
how much of the stream was read before the connection was closed.

The model runs on the OpenAI SDK's real client, built by the model factory
from an entry with ``enforces_max_tokens: false``; without it the model is
the plain ``ChatOpenAI`` it was and reads every chunk.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Iterator
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from inference_core.agents.middleware.cost_tracking import CostTrackingMiddleware
from inference_core.llm.config import ModelConfig, ModelProvider
from inference_core.llm.models import LLMModelFactory
from inference_core.llm.output_length_guard import (
    OutputLengthGuard,
    count_text_tokens,
    guarded,
    usage_is_estimated,
)

NAME = "cortecs-mistral-small-2603"
BASE_URL = "https://gateway.test/v1"


@pytest.fixture(autouse=True)
def _real_models(monkeypatch):
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled", lambda: False
    )


def _chunk(delta: dict | None, *, finish: str | None = None, usage=None) -> bytes:
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "mistral-small-2603",
        "choices": (
            [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
            if delta is not None
            else []
        ),
    }
    if usage is not None:
        body["usage"] = usage
    return f"data: {json.dumps(body)}\n\n".encode()


class _Gateway:
    """Streams *reasoning* words of reasoning and *answer* words of answer,
    whatever the request's limit, and counts the chunks it sent."""

    def __init__(self, reasoning: int = 0, answer: int = 400) -> None:
        self.reasoning = reasoning
        self.answer = answer
        self.sent = 0
        self.bodies: list[dict[str, Any]] = []

    @property
    def total(self) -> int:
        """Every chunk: the role, the deltas, the finish, the usage, [DONE]."""
        return self.reasoning + self.answer + 4

    def _chunks(self) -> Iterator[bytes]:
        yield _chunk({"role": "assistant", "content": ""})
        for _ in range(self.reasoning):
            yield _chunk({"reasoning_content": " think"})
        for _ in range(self.answer):
            yield _chunk({"content": " word"})
        yield _chunk({"content": ""}, finish="stop")
        output = self.reasoning + self.answer
        yield _chunk(
            None,
            usage={
                "prompt_tokens": 57,
                "completion_tokens": output,
                "total_tokens": 57 + output,
                "completion_tokens_details": {"reasoning_tokens": self.reasoning},
            },
        )
        yield b"data: [DONE]\n\n"

    def _counted(self) -> Iterator[bytes]:
        for part in self._chunks():
            self.sent += 1
            yield part

    async def _acounted(self):
        for part in self._chunks():
            self.sent += 1
            yield part

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=self._counted(),
            headers={"content-type": "text/event-stream"},
        )

    async def ahandler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=self._acounted(),
            headers={"content-type": "text/event-stream"},
        )


def _model(gateway: _Gateway, *, enforces: bool = False, max_tokens: int = 50):
    config = ModelConfig(
        name=NAME,
        provider=ModelProvider.CORTECS,
        provider_model_id="mistral-small-2603",
        api_key="test-key",
        base_url=BASE_URL,
        max_tokens=max_tokens,
        enforces_max_tokens=enforces,
    )
    llm_config = MagicMock()
    llm_config.enable_caching = False
    llm_config.get_model_config.return_value = config
    llm_config.get_provider_config.side_effect = KeyError("no provider entry")
    model = LLMModelFactory(llm_config).create_model(NAME)
    root = openai.OpenAI(
        api_key="test-key",
        base_url=BASE_URL,
        http_client=httpx.Client(transport=httpx.MockTransport(gateway.handler)),
    )
    root_async = openai.AsyncOpenAI(
        api_key="test-key",
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(gateway.ahandler)
        ),
    )
    model.root_client = root
    model.root_async_client = root_async
    model.client = root.chat.completions
    model.async_client = root_async.chat.completions
    return model


def _ask() -> list[Any]:
    return [SystemMessage("Answer in English."), HumanMessage("Plan the deliveries.")]


# --------------------------------------------------------------------------- #
# The cut
# --------------------------------------------------------------------------- #


def test_the_stream_is_closed_at_the_limit():
    gateway = _Gateway(answer=400)
    model = _model(gateway, max_tokens=50)
    assert isinstance(model, OutputLengthGuard)
    assert isinstance(model, ChatOpenAI)

    message = model.invoke(_ask())

    assert message.usage_metadata["output_tokens"] == 50
    assert message.usage_metadata["input_tokens"] > 0
    assert message.response_metadata["finish_reason"] == "length"
    assert usage_is_estimated(message)
    assert message.content.count("word") == 50
    # The rest of the stream was never read (the transport may run a few
    # chunks ahead of the reader).
    assert gateway.sent < 70 < gateway.total


@pytest.mark.asyncio
async def test_an_async_call_is_cut_too():
    gateway = _Gateway(answer=400)
    model = _model(gateway, max_tokens=50)

    message = await model.ainvoke(_ask())

    assert message.usage_metadata["output_tokens"] == 50
    assert message.response_metadata["finish_reason"] == "length"
    assert gateway.sent < 70


def test_the_call_s_own_limit_wins():
    gateway = _Gateway(answer=400)
    model = _model(gateway, max_tokens=300)

    message = model.bind(max_tokens=20).invoke(_ask())

    assert message.usage_metadata["output_tokens"] == 20
    assert gateway.bodies[0]["max_completion_tokens"] == 20


def test_reasoning_counts_towards_the_limit():
    """The reasoning ChatOpenAI does not read is counted on the raw chunk."""
    gateway = _Gateway(reasoning=300, answer=100)
    model = _model(gateway, max_tokens=40)

    message = model.invoke(_ask())

    assert message.usage_metadata["output_tokens"] == 40
    assert message.content == ""
    assert gateway.sent < 60


def test_a_call_without_streaming_is_streamed_and_cut():
    """As the vision helper calls a model: streaming off."""
    gateway = _Gateway(answer=400)
    model = _model(gateway, max_tokens=50)
    model.streaming = False
    model.disable_streaming = True

    message = model.invoke(_ask())

    assert gateway.bodies[0]["stream"] is True
    assert message.usage_metadata["output_tokens"] == 50
    assert message.response_metadata["finish_reason"] == "length"
    assert usage_is_estimated(message)


def test_a_response_within_the_limit_keeps_the_provider_s_usage():
    gateway = _Gateway(answer=30)
    model = _model(gateway, max_tokens=50)

    message = model.invoke(_ask())

    assert message.usage_metadata["input_tokens"] == 57
    assert message.usage_metadata["output_tokens"] == 30
    assert not usage_is_estimated(message)
    assert message.response_metadata.get("finish_reason") == "stop"
    assert gateway.sent == gateway.total


@pytest.mark.asyncio
async def test_concurrent_calls_on_one_model_count_apart():
    """Two calls at once on one model object, each with its own limit."""
    model = _model(_Gateway(answer=400), max_tokens=300)

    results = await asyncio.gather(
        model.bind(max_tokens=30).ainvoke(_ask()),
        model.bind(max_tokens=70).ainvoke(_ask()),
        model.bind(max_tokens=45).ainvoke(_ask()),
    )

    assert [m.usage_metadata["output_tokens"] for m in results] == [30, 70, 45]


def test_no_limit_reads_the_whole_stream():
    gateway = _Gateway(answer=120)
    model = _model(gateway, max_tokens=50)
    model.max_tokens = None

    message = model.invoke(_ask())

    assert message.usage_metadata["output_tokens"] == 120
    assert not usage_is_estimated(message)
    assert gateway.sent == gateway.total


# --------------------------------------------------------------------------- #
# Without the option: as before
# --------------------------------------------------------------------------- #


def test_without_the_option_the_model_is_plain_and_reads_everything():
    gateway = _Gateway(answer=400)
    model = _model(gateway, enforces=True, max_tokens=50)

    message = model.invoke(_ask())

    assert type(model) is ChatOpenAI
    assert message.usage_metadata["output_tokens"] == 400
    assert not usage_is_estimated(message)
    assert gateway.sent == gateway.total


def test_another_provider_s_model_ignores_the_option(caplog):
    config = ModelConfig(
        name="mistral-small-latest",
        provider=ModelProvider.MISTRAL,
        api_key="test-key",
        enforces_max_tokens=False,
    )
    llm_config = MagicMock()
    llm_config.enable_caching = False
    llm_config.get_model_config.return_value = config

    with caplog.at_level("WARNING"):
        model = LLMModelFactory(llm_config).create_model("mistral-small-latest")

    assert model is not None
    assert not isinstance(model, OutputLengthGuard)
    assert (
        "enforces_max_tokens is ignored for model 'mistral-small-latest'"
        in caplog.text
    )


def test_one_guarded_class_per_base():
    assert guarded(ChatOpenAI) is guarded(ChatOpenAI)
    assert guarded(guarded(ChatOpenAI)) is guarded(ChatOpenAI)


def test_the_counter():
    assert count_text_tokens("") == 0
    assert count_text_tokens(" word" * 10) == 10


# --------------------------------------------------------------------------- #
# The log
# --------------------------------------------------------------------------- #


def test_the_log_counts_what_was_received_as_an_estimate():
    gateway = _Gateway(reasoning=100, answer=300)
    model = _model(gateway, max_tokens=60)
    message = model.invoke(_ask())
    middleware = CostTrackingMiddleware(pricing_config=None)
    request = MagicMock(system_message=None, messages=[], model=None)
    response = MagicMock(result=[message])

    prepared = middleware._build_usage_session(response, request)

    assert prepared.is_estimated is True
    assert prepared.details["estimated"] is True
    assert prepared.fragment["output_tokens"] == 60


def test_a_provider_s_usage_is_not_marked():
    middleware = CostTrackingMiddleware(pricing_config=None)
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
    )
    request = MagicMock(system_message=None, messages=[], model=None)

    prepared = middleware._build_usage_session(MagicMock(result=[message]), request)

    assert prepared.is_estimated is False
    assert not (prepared.details or {}).get("estimated")


def test_a_deepinfra_model_keeps_its_own_class_under_the_guard():
    from inference_core.llm.chat_deepinfra import ChatDeepInfraReasoning

    config = ModelConfig(
        name="XiaomiMiMo/MiMo-V2.6-Flash",
        provider=ModelProvider.DEEPINFRA,
        api_key="test-key",
        enforces_max_tokens=False,
    )
    llm_config = MagicMock()
    llm_config.enable_caching = False
    llm_config.get_model_config.return_value = config

    model = LLMModelFactory(llm_config).create_model(config.name)

    assert isinstance(model, ChatDeepInfraReasoning)
    assert isinstance(model, OutputLengthGuard)

"""
Contract tests for the Mistral provider against the real ChatMistralAI class.

WHY: The rest of the stack relies on behaviour owned by ``langchain-mistralai``
rather than by this repo: ``thinking`` chunks surfacing as standard
``reasoning`` content blocks, usage arriving on the stream, and
``model_kwargs`` reaching the request payload. These tests pin that contract
with a mocked HTTP transport (no network), so a dependency upgrade that breaks
it fails here instead of silently dropping reasoning in the UI.
"""

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_mistralai import ChatMistralAI

from inference_core.llm.config import ModelConfig, ModelProvider
from inference_core.llm.models import LLMModelFactory
from inference_core.services.agents_service import AgentService

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}

COMPLETION_RESPONSE = {
    "id": "cmpl-1",
    "object": "chat.completion",
    "model": "mistral-small-latest",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": [{"type": "text", "text": "Two plus two."}],
                    },
                    {"type": "text", "text": "4"},
                ],
            },
        }
    ],
    "usage": USAGE,
}


def _stream_chunk(delta: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Build one SSE chunk in Mistral's streaming wire format."""
    choice = {"index": 0, "delta": delta, "finish_reason": extra.pop("finish", None)}
    return {
        "id": "cmpl-1",
        "model": "mistral-small-latest",
        "choices": [choice],
        **extra,
    }


STREAM_CHUNKS = [
    _stream_chunk(
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": [{"type": "text", "text": "Two "}]}
            ],
        }
    ),
    _stream_chunk(
        {
            "content": [
                {
                    "type": "thinking",
                    "thinking": [{"type": "text", "text": "plus two."}],
                }
            ]
        }
    ),
    _stream_chunk({"content": "4"}, finish="stop", usage=USAGE),
]


@pytest.fixture(autouse=True)
def _disable_llm_emulation(monkeypatch):
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled",
        lambda: False,
    )


@pytest.fixture
def sent_payloads() -> list[dict[str, Any]]:
    """Request bodies captured by the mocked Mistral endpoint."""
    return []


@pytest.fixture
def transport(sent_payloads: list[dict[str, Any]]) -> httpx.MockTransport:
    """Mocked Mistral chat-completions endpoint (plain and SSE responses)."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent_payloads.append(payload)
        if payload.get("stream"):
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in STREAM_CHUNKS)
            return httpx.Response(
                200,
                content=(body + "data: [DONE]\n\n").encode(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=COMPLETION_RESPONSE)

    return httpx.MockTransport(handler)


@pytest.fixture
def reasoning_model_config() -> ModelConfig:
    return ModelConfig(
        name="mistral-small-latest",
        provider=ModelProvider.MISTRAL,
        api_key="test-key",
        max_tokens=64,
        reasoning_config={"model_kwargs": {"reasoning_effort": "high"}},
    )


def _build_model(
    config: ModelConfig, transport: httpx.MockTransport, **kwargs: Any
) -> ChatMistralAI:
    """Create the model through the factory, then point it at the mock endpoint."""
    llm_config = MagicMock()
    llm_config.enable_caching = False
    llm_config.get_model_config.return_value = config

    model = LLMModelFactory(llm_config).create_model(config.name, **kwargs)
    assert isinstance(model, ChatMistralAI)

    model.client = httpx.Client(base_url=MISTRAL_BASE_URL, transport=transport)
    model.async_client = httpx.AsyncClient(
        base_url=MISTRAL_BASE_URL, transport=transport
    )
    return model


class TestMistralFactoryContract:
    """The factory builds a working ChatMistralAI from a YAML-shaped config."""

    def test_factory_maps_config_onto_model_fields(
        self, reasoning_model_config, transport
    ):
        model = _build_model(reasoning_model_config, transport, reasoning_output=True)

        assert model.model == "mistral-small-latest"
        assert model.mistral_api_key.get_secret_value() == "test-key"
        assert model.endpoint == MISTRAL_BASE_URL
        assert model.timeout == 60
        assert model.max_tokens == 64
        assert model.model_kwargs == {"reasoning_effort": "high"}
        # The factory forces streaming on; the missing ``stream_usage`` field
        # on ChatMistralAI must not break model creation.
        assert model.streaming is True

    def test_temperature_above_one_is_rejected(self, transport):
        """ChatMistralAI only accepts temperature in [0, 1]."""
        config = ModelConfig(
            name="mistral-large-latest",
            provider=ModelProvider.MISTRAL,
            api_key="test-key",
            temperature=1.5,
        )
        llm_config = MagicMock()
        llm_config.enable_caching = False
        llm_config.get_model_config.return_value = config

        assert LLMModelFactory(llm_config).create_model(config.name) is None


class TestMistralReasoningContract:
    """Reasoning reaches the agent stream as standard ``reasoning`` blocks."""

    def test_invoke_sends_reasoning_effort_and_returns_reasoning_block(
        self, reasoning_model_config, transport, sent_payloads
    ):
        model = _build_model(reasoning_model_config, transport, reasoning_output=True)
        model.streaming = False

        message = model.invoke("2+2?")

        payload = sent_payloads[-1]
        assert payload["model"] == "mistral-small-latest"
        assert payload["reasoning_effort"] == "high"
        assert payload["max_tokens"] == 64
        assert "frequency_penalty" not in payload
        assert "presence_penalty" not in payload

        assert message.content_blocks == [
            {"type": "reasoning", "reasoning": "Two plus two."},
            {"type": "text", "text": "4"},
        ]
        assert message.usage_metadata == {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
        }

    def test_reasoning_effort_absent_without_reasoning_output(
        self, reasoning_model_config, transport, sent_payloads
    ):
        model = _build_model(reasoning_model_config, transport)
        model.streaming = False

        model.invoke("2+2?")

        assert "reasoning_effort" not in sent_payloads[-1]

    async def test_stream_yields_reasoning_segments_and_usage(
        self, reasoning_model_config, transport, sent_payloads
    ):
        model = _build_model(reasoning_model_config, transport, reasoning_output=True)

        segments: list[tuple[str, str]] = []
        usage = None
        async for chunk in model.astream("2+2?"):
            for text, meta in AgentService._extract_message_segments(
                chunk, {"langgraph_node": "model"}
            ):
                segments.append((meta["type"], text))
            usage = chunk.usage_metadata or usage

        assert sent_payloads[-1]["stream"] is True
        assert segments == [
            ("reasoning", "Two "),
            ("reasoning", "plus two."),
            ("text", "4"),
        ]
        assert usage == {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}

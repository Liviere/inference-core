"""
Contract tests for Claude thinking against the real ChatAnthropic class.

WHY: Claude models from the 5.5 generation think by default, but return the
text of their thinking only when the request asks for it with
``thinking.display: "summarized"``; the default (``"omitted"``) streams
thinking blocks with an empty text and a signature. Showing the reasoning
therefore depends on behaviour owned by ``langchain-anthropic`` and
``langchain-core`` rather than by this repo: the ``thinking`` dict reaching
the request verbatim, ``thinking_delta`` events surfacing as standard
``reasoning`` content blocks, and empty or redacted blocks yielding nothing.
These tests pin that contract with a mocked HTTP transport (no network).
"""

import json
from typing import Any
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest
from langchain_anthropic import ChatAnthropic

from inference_core.llm.config import ModelConfig, ModelProvider
from inference_core.llm.models import LLMModelFactory
from inference_core.services.agents_service import AgentService

ANTHROPIC_BASE_URL = "https://api.anthropic.com"

SUMMARIZED = {"type": "adaptive", "display": "summarized"}


def _event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n"


def _message_start() -> str:
    return _event(
        "message_start",
        {
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 1},
            }
        },
    )


def _block(index: int, block: dict[str, Any], deltas: list[dict[str, Any]]) -> str:
    events = [_event("content_block_start", {"index": index, "content_block": block})]
    events += [
        _event("content_block_delta", {"index": index, "delta": delta})
        for delta in deltas
    ]
    events.append(_event("content_block_stop", {"index": index}))
    return "".join(events)


def _message_end() -> str:
    return _event(
        "message_delta",
        {
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "output_tokens": 40,
                "output_tokens_details": {"thinking_tokens": 30},
            },
        },
    ) + _event("message_stop", {})


# What a summarized turn looks like: a thinking block with text, one with an
# empty text (the default ``display: "omitted"``), a redacted block, the answer.
SUMMARIZED_STREAM = (
    _message_start()
    + _block(
        0,
        {"type": "thinking", "thinking": "", "signature": ""},
        [
            {"type": "thinking_delta", "thinking": "Two plus "},
            {"type": "thinking_delta", "thinking": "two is four."},
            {"type": "signature_delta", "signature": "sig-1"},
        ],
    )
    + _block(
        1,
        {"type": "thinking", "thinking": "", "signature": ""},
        [{"type": "signature_delta", "signature": "sig-2"}],
    )
    + _block(2, {"type": "redacted_thinking", "data": "opaque"}, [])
    + _block(
        3,
        {"type": "text", "text": ""},
        [{"type": "text_delta", "text": "4"}],
    )
    + _message_end()
)

# The default ``display: "omitted"``: thinking happens, its text is empty.
OMITTED_STREAM = (
    _message_start()
    + _block(
        0,
        {"type": "thinking", "thinking": "", "signature": ""},
        [{"type": "signature_delta", "signature": "sig-1"}],
    )
    + _block(1, {"type": "text", "text": ""}, [{"type": "text_delta", "text": "4"}])
    + _message_end()
)


@pytest.fixture(autouse=True)
def _disable_llm_emulation(monkeypatch):
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled",
        lambda: False,
    )


@pytest.fixture
def sent_payloads() -> list[dict[str, Any]]:
    """Request bodies captured by the mocked Messages endpoint."""
    return []


def _transport(sent_payloads: list[dict[str, Any]], body: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        sent_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    return httpx.MockTransport(handler)


def _config(thinking: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        name="claude-haiku-5-5",
        provider=ModelProvider.CLAUDE,
        api_key="test-key",
        max_tokens=256,
        temperature=0.7,
        reasoning_config={"thinking": thinking},
    )


def _build_model(
    config: ModelConfig, transport: httpx.MockTransport, **kwargs: Any
) -> ChatAnthropic:
    """Create the model through the factory, then point it at the mock endpoint."""
    llm_config = MagicMock()
    llm_config.enable_caching = False
    llm_config.get_model_config.return_value = config

    model = LLMModelFactory(llm_config).create_model(config.name, **kwargs)
    assert isinstance(model, ChatAnthropic)

    # ``_client`` / ``_async_client`` are cached properties: setting them in
    # the instance dict replaces the clients the model would build.
    model.__dict__["_client"] = anthropic.Anthropic(
        api_key="test-key",
        base_url=ANTHROPIC_BASE_URL,
        http_client=httpx.Client(transport=transport),
    )
    model.__dict__["_async_client"] = anthropic.AsyncAnthropic(
        api_key="test-key",
        base_url=ANTHROPIC_BASE_URL,
        http_client=httpx.AsyncClient(transport=transport),
    )
    return model


async def _segments(model: ChatAnthropic) -> tuple[list[tuple[str, str]], Any]:
    segments: list[tuple[str, str]] = []
    usage = None
    async for chunk in model.astream("2+2?"):
        for text, meta in AgentService._extract_message_segments(
            chunk, {"langgraph_node": "model"}
        ):
            segments.append((meta["type"], text))
        usage = chunk.usage_metadata or usage
    return segments, usage


class TestClaudeThinkingRequest:
    """The ``thinking`` dict from ``reasoning_config`` reaches the request as is."""

    async def test_display_reaches_the_request_with_reasoning_output(
        self, sent_payloads
    ):
        model = _build_model(
            _config(SUMMARIZED),
            _transport(sent_payloads, SUMMARIZED_STREAM),
            reasoning_output=True,
        )

        await _segments(model)

        payload = sent_payloads[-1]
        assert payload["thinking"] == SUMMARIZED
        # The API refuses sampling parameters on these models.
        assert "temperature" not in payload

    async def test_no_thinking_parameter_without_reasoning_output(self, sent_payloads):
        model = _build_model(
            _config(SUMMARIZED), _transport(sent_payloads, OMITTED_STREAM)
        )

        await _segments(model)

        assert "thinking" not in sent_payloads[-1]


class TestClaudeThinkingStream:
    """Thinking reaches the agent stream as ``reasoning`` segments."""

    async def test_summarized_thinking_becomes_reasoning_segments(self, sent_payloads):
        model = _build_model(
            _config(SUMMARIZED),
            _transport(sent_payloads, SUMMARIZED_STREAM),
            reasoning_output=True,
        )

        segments, usage = await _segments(model)

        # The empty block and the redacted one yield nothing.
        assert segments == [
            ("reasoning", "Two plus "),
            ("reasoning", "two is four."),
            ("text", "4"),
        ]
        # Thinking tokens are a part of ``output_tokens``, not added to them.
        assert usage["output_tokens"] == 40
        assert usage["output_token_details"] == {"reasoning": 30}

    async def test_omitted_thinking_yields_no_reasoning(self, sent_payloads):
        model = _build_model(
            _config({"type": "adaptive"}),
            _transport(sent_payloads, OMITTED_STREAM),
            reasoning_output=True,
        )

        segments, _ = await _segments(model)

        assert sent_payloads[-1]["thinking"] == {"type": "adaptive"}
        assert segments == [("text", "4")]

"""An OpenAI model marks the end of its system prompt when its entry asks.

``prompt_cache_breakpoints: true`` puts ``prompt_cache_breakpoint`` on the
last block of the system prompt and ``prompt_cache_options`` on the request,
for Chat Completions and for the Responses API (which ``ChatOpenAI`` picks for
GPT-6 with tools or reasoning). The Chat Completions body is read off a mocked
transport under the OpenAI SDK's real client. Without the option the model is
the plain ``ChatOpenAI`` it was, and its request is the same.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from inference_core.llm.config import ModelConfig, ModelProvider
from inference_core.llm.models import LLMModelFactory, without_prompt_cache
from inference_core.llm.openai_cache import (
    BREAKPOINT,
    CACHE_OPTIONS,
    ChatOpenAICacheBreakpoints,
    mark_system_prompt,
)

TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look a delivery up.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}


@pytest.fixture(autouse=True)
def _real_models(monkeypatch):
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled", lambda: False
    )


def _factory(name: str = "gpt-6-luna", provider=ModelProvider.OPENAI, **entry):
    config = ModelConfig(name=name, provider=provider, api_key="test-key", **entry)
    llm_config = MagicMock()
    llm_config.enable_caching = False
    llm_config.get_model_config.return_value = config
    return LLMModelFactory(llm_config)


def _conversation() -> list[Any]:
    return [
        SystemMessage("You keep a made-up delivery ledger. Answer in one word."),
        HumanMessage("Which town is in entry 1?"),
        AIMessage("Ashford"),
        HumanMessage("And in entry 2?"),
    ]


class _Wire:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if body.get("stream"):
            chunks = [
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "Larkfield"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ]
            sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            return httpx.Response(
                200,
                content=(sse + "data: [DONE]\n\n").encode(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Larkfield"},
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 1},
            },
        )


def _on_wire(model: ChatOpenAI) -> _Wire:
    wire = _Wire()
    transport = httpx.MockTransport(wire.handler)
    root = openai.OpenAI(
        api_key="test-key", http_client=httpx.Client(transport=transport)
    )
    root_async = openai.AsyncOpenAI(
        api_key="test-key", http_client=httpx.AsyncClient(transport=transport)
    )
    model.root_client = root
    model.root_async_client = root_async
    model.client = root.chat.completions
    model.async_client = root_async.chat.completions
    return wire


# --------------------------------------------------------------------------- #
# On the wire (Chat Completions)
# --------------------------------------------------------------------------- #


def test_the_request_marks_the_end_of_the_system_prompt():
    model = _factory(prompt_cache_breakpoints=True).create_model("gpt-6-luna")
    assert isinstance(model, ChatOpenAICacheBreakpoints)
    wire = _on_wire(model)

    model.invoke(_conversation())

    body = wire.bodies[0]
    assert body["prompt_cache_options"] == CACHE_OPTIONS
    system, *rest = body["messages"]
    assert system["role"] == "system"
    assert system["content"] == [
        {
            "type": "text",
            "text": "You keep a made-up delivery ledger. Answer in one word.",
            "prompt_cache_breakpoint": BREAKPOINT,
        }
    ]
    # Nothing else carries one.
    assert "prompt_cache_breakpoint" not in json.dumps(rest)


@pytest.mark.asyncio
async def test_an_async_call_too():
    model = _factory(prompt_cache_breakpoints=True).create_model("gpt-6-luna")
    wire = _on_wire(model)

    await model.ainvoke(_conversation())

    assert wire.bodies[0]["prompt_cache_options"] == CACHE_OPTIONS
    assert (
        wire.bodies[0]["messages"][0]["content"][0]["prompt_cache_breakpoint"]
        == BREAKPOINT
    )


def test_the_messages_the_run_keeps_are_not_changed():
    """The breakpoint is put on the request, never on a message, so the next
    model of a fallback (another provider) is sent the conversation as it
    was."""
    model = _factory(prompt_cache_breakpoints=True).create_model("gpt-6-luna")
    _on_wire(model)
    messages = _conversation()
    before = [message.model_dump() for message in messages]

    model.invoke(messages)

    assert [message.model_dump() for message in messages] == before
    assert isinstance(messages[0].content, str)


def test_without_the_option_the_request_is_as_before():
    model = _factory().create_model("gpt-6-luna")
    assert type(model) is ChatOpenAI
    wire = _on_wire(model)
    plain = ChatOpenAI(model="gpt-6-luna", api_key="test-key", streaming=True)
    plain.stream_usage = True
    plain_wire = _on_wire(plain)

    # Without tools: GPT-6 with tools goes to the Responses API.
    model.invoke(_conversation(), temperature=0.2)
    plain.invoke(_conversation(), temperature=0.2)

    assert wire.bodies[0] == plain_wire.bodies[0]
    assert "prompt_cache_options" not in wire.bodies[0]
    assert wire.bodies[0]["messages"][0]["content"] == (
        "You keep a made-up delivery ledger. Answer in one word."
    )


# --------------------------------------------------------------------------- #
# The payload: Responses API and shapes of a system prompt
# --------------------------------------------------------------------------- #


def test_a_responses_request_marks_it_too():
    """GPT-6 with reasoning goes to the Responses API."""
    model = _factory(
        prompt_cache_breakpoints=True, reasoning={"effort": "low"}
    ).create_model("gpt-6-luna")

    payload = model._get_request_payload(_conversation(), tools=[TOOL])

    assert "input" in payload
    assert payload["prompt_cache_options"] == CACHE_OPTIONS
    system = payload["input"][0]
    assert system["role"] == "system"
    assert system["content"] == [
        {
            "type": "input_text",
            "text": "You keep a made-up delivery ledger. Answer in one word.",
            "prompt_cache_breakpoint": BREAKPOINT,
        }
    ]


def test_the_last_message_of_a_system_prompt_of_blocks_gets_it():
    payload = {
        "messages": [
            {"role": "system", "content": "First part."},
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Second part."},
                    {"type": "text", "text": "Third part."},
                ],
            },
            {"role": "user", "content": "Hi"},
        ]
    }

    assert mark_system_prompt(payload) is True

    first, second, user = payload["messages"]
    assert first["content"] == "First part."
    assert "prompt_cache_breakpoint" not in second["content"][0]
    assert second["content"][1]["prompt_cache_breakpoint"] == BREAKPOINT
    assert user["content"] == "Hi"


def test_a_request_without_a_system_prompt_gets_nothing():
    model = _factory(prompt_cache_breakpoints=True).create_model("gpt-6-luna")

    payload = model._get_request_payload([HumanMessage("Hi")])

    assert "prompt_cache_options" not in payload
    assert "prompt_cache_breakpoint" not in json.dumps(payload)


def test_a_call_s_own_cache_options_win():
    model = _factory(prompt_cache_breakpoints=True).create_model("gpt-6-luna")

    payload = model._get_request_payload(
        _conversation(), prompt_cache_options={"mode": "explicit", "ttl": "30m"}
    )

    assert payload["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert payload["messages"][0]["content"][0]["prompt_cache_breakpoint"]


# --------------------------------------------------------------------------- #
# Side jobs and other providers
# --------------------------------------------------------------------------- #


def test_a_model_built_without_the_prompt_cache_has_no_breakpoint():
    model = _factory(prompt_cache_breakpoints=True).create_model(
        "gpt-6-luna", prompt_cache=False
    )

    assert type(model) is ChatOpenAI
    payload = model._get_request_payload(_conversation())
    assert "prompt_cache_options" not in payload
    assert isinstance(payload["messages"][0]["content"], str)


def test_without_prompt_cache_drops_the_breakpoint_of_a_built_model():
    model = _factory(prompt_cache_breakpoints=True).create_model("gpt-6-luna")

    copy = without_prompt_cache(model)

    assert copy is not model
    assert copy.prompt_cache_breakpoints is False
    payload = copy._get_request_payload(_conversation())
    assert "prompt_cache_options" not in payload
    # The model itself keeps it.
    assert model.prompt_cache_breakpoints is True


def test_without_prompt_cache_leaves_a_plain_model_alone():
    model = _factory().create_model("gpt-6-luna")

    assert without_prompt_cache(model) is model


def test_another_provider_s_model_ignores_the_option(caplog):
    factory = _factory(
        "mistral-small-latest",
        provider=ModelProvider.MISTRAL,
        prompt_cache_breakpoints=True,
    )

    with caplog.at_level("WARNING"):
        model = factory.create_model("mistral-small-latest")

    assert model is not None
    assert not hasattr(model, "prompt_cache_breakpoints")
    assert (
        "prompt_cache_breakpoints is ignored for model 'mistral-small-latest'"
        in caplog.text
    )

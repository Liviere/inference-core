"""Claude's thinking blocks in a history that goes to another provider.

A conversation that ran on Claude keeps its answers' ``thinking`` and
``redacted_thinking`` blocks. When the next call goes to a model of another
provider (a fallback after Claude failed, say), langchain-mistralai sends
those blocks to Mistral as they are: Anthropic's shape, with a ``signature``,
which is not Mistral's own ``thinking`` chunk (a list of text chunks).

``strip_foreign_thinking`` (off by default) takes them out of the request
for a model that is not Anthropic's; text and tool calls stay.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from inference_core.agents.middleware.model_fallback import (
    CancelAwareModelFallbackMiddleware,
    build_model_fallback_middleware,
    strip_foreign_thinking,
)

langchain_mistralai = pytest.importorskip("langchain_mistralai")
langchain_anthropic = pytest.importorskip("langchain_anthropic")

THINKING = {
    "type": "thinking",
    "thinking": "The user asks about the weather; no tool is needed.",
    "signature": "EqQBCkYIBxgCKkBsig==",
}
REDACTED = {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix/LafPsn4aDFIT"}
MISTRAL_THINKING = {
    "type": "thinking",
    "thinking": [{"type": "text", "text": "Mistral's own reasoning."}],
}


def _history() -> list[Any]:
    """A conversation with two answers of Claude and one tool round."""
    return [
        HumanMessage("What is the weather in Warsaw?"),
        AIMessage(
            content=[THINKING, {"type": "text", "text": "It is sunny."}],
            response_metadata={"model_provider": "anthropic"},
        ),
        HumanMessage("And in Krakow? Check it."),
        AIMessage(
            content=[REDACTED, THINKING],
            tool_calls=[
                {"name": "weather", "args": {"city": "Krakow"}, "id": "toolu_01"}
            ],
            response_metadata={"model_provider": "anthropic"},
        ),
        ToolMessage("Rain, 12 C", tool_call_id="toolu_01"),
        AIMessage(
            content=[THINKING],
            response_metadata={"model_provider": "anthropic"},
        ),
        HumanMessage("Thanks. Anything else?"),
    ]


def _mistral_with_recorder(bodies: list[dict]) -> Any:
    """ChatMistralAI whose requests are recorded and answered locally."""

    def answer(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "object": "chat.completion",
                "model": "mistral-small-latest",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Done."},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
        )

    model = langchain_mistralai.ChatMistralAI(
        model="mistral-small-latest", api_key="test-key", max_retries=0
    )
    model.client = httpx.Client(
        base_url="https://api.mistral.ai/v1", transport=httpx.MockTransport(answer)
    )
    return model


def _anthropic_blocks(body: dict) -> list[dict]:
    """Blocks in Anthropic's shape among the assistant messages of a request."""
    found = []
    for message in body["messages"]:
        content = message.get("content")
        if message.get("role") != "assistant" or not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "redacted_thinking" or "signature" in block:
                found.append(block)
    return found


def test_today_mistral_is_sent_claudes_thinking_blocks():
    """The request langchain-mistralai builds from a Claude history carries
    Anthropic's blocks (``signature``, ``redacted_thinking``): what Mistral
    is asked to read when Claude fails and it takes over."""
    bodies: list[dict] = []
    model = _mistral_with_recorder(bodies)

    model.invoke(_history())

    blocks = _anthropic_blocks(bodies[0])
    assert {b["type"] for b in blocks} == {"thinking"}
    assert all("signature" in b for b in blocks)
    # The answer of nothing but thinking goes as a message of nothing but it.
    only_thinking = [
        m
        for m in bodies[0]["messages"]
        if m["role"] == "assistant"
        and isinstance(m.get("content"), list)
        and all(b.get("type") == "thinking" for b in m["content"])
    ]
    assert only_thinking


# ----------------------------------------------------------------------------
# strip_foreign_thinking
# ----------------------------------------------------------------------------


def test_for_a_model_of_another_provider_claudes_blocks_go():
    model = _mistral_with_recorder([])

    messages = strip_foreign_thinking(_history(), model)

    first_answer = messages[1]
    assert first_answer.content == [{"type": "text", "text": "It is sunny."}]
    # The tool call stays, without the blocks.
    tool_round = messages[3]
    assert tool_round.tool_calls[0]["id"] == "toolu_01"
    assert tool_round.content == []
    # An answer of nothing but thinking and no tool call is left out: Mistral
    # refuses an assistant message with neither content nor tool calls.
    assert [type(m).__name__ for m in messages] == [
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "HumanMessage",
    ]


def test_the_history_itself_is_not_changed():
    history = _history()

    strip_foreign_thinking(history, _mistral_with_recorder([]))

    assert history[1].content[0] == THINKING
    assert len(history) == 7


def test_for_a_claude_model_nothing_changes():
    claude = langchain_anthropic.ChatAnthropic(
        model="claude-haiku-5-5", api_key="test-key"
    )
    history = _history()

    assert strip_foreign_thinking(history, claude) is history


def test_mistrals_own_thinking_stays():
    history = [
        HumanMessage("Hi"),
        AIMessage(content=[MISTRAL_THINKING, {"type": "text", "text": "Hello."}]),
        HumanMessage("Again"),
    ]

    assert strip_foreign_thinking(history, _mistral_with_recorder([])) is history


def test_a_model_given_by_name_is_left_alone():
    history = _history()

    assert strip_foreign_thinking(history, "mistral-small-latest") is history


# ----------------------------------------------------------------------------
# In the fallback middleware, on a real graph
# ----------------------------------------------------------------------------


class _Unavailable(FakeListChatModel):
    """A primary model that is down."""

    def _call(self, *args: Any, **kwargs: Any) -> str:  # noqa: D401
        raise RuntimeError("primary is down")


def _run(middleware, bodies):
    agent = create_agent(model=_Unavailable(responses=["never"]), middleware=[middleware])
    return agent.invoke({"messages": _history()})


def test_off_by_default_the_fallback_sends_the_blocks():
    bodies: list[dict] = []
    middleware = CancelAwareModelFallbackMiddleware(_mistral_with_recorder(bodies))

    _run(middleware, bodies)

    assert _anthropic_blocks(bodies[0])


def test_switched_on_the_fallback_sends_none():
    bodies: list[dict] = []
    middleware = CancelAwareModelFallbackMiddleware(
        _mistral_with_recorder(bodies), strip_foreign_thinking=True
    )

    result = _run(middleware, bodies)

    assert _anthropic_blocks(bodies[0]) == []
    assert result["messages"][-1].content == "Done."
    # Text and the tool round reached Mistral.
    sent = bodies[0]["messages"]
    assert any(m.get("tool_calls") for m in sent)
    assert any(
        m["role"] == "assistant" and "It is sunny." in json.dumps(m["content"])
        for m in sent
    )


def test_the_builder_passes_the_switch():
    class _Factory:
        config = None

        def create_model(self, name, **kwargs):
            return _mistral_with_recorder([])

    middleware = build_model_fallback_middleware(
        model_factory=_Factory(),
        fallback_models=["mistral-small-latest"],
        strip_foreign_thinking=True,
    )

    assert middleware.strip_foreign_thinking is True
    assert (
        build_model_fallback_middleware(
            model_factory=_Factory(), fallback_models=["mistral-small-latest"]
        ).strip_foreign_thinking
        is False
    )

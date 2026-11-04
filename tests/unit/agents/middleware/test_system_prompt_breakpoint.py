"""A cache breakpoint at the end of the system prompt, next to the automatic one.

A Claude model with ``prompt_cache`` caches its prompt up to the request's
last block, so a new conversation of the same agent found no entry at the end
of the tools and system prompt and wrote them again. The middleware marks
that end, with the automatic breakpoint's lifetime; another model is left as
it is. Checked on the requests a real ChatAnthropic sends, through a mocked
transport.
"""

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI

from inference_core.agents.middleware.system_prompt_breakpoint import (
    MAX_BREAKPOINTS,
    SystemPromptBreakpointMiddleware,
    automatic_cache_lifetime,
    breakpoints_of,
    marked_system_message,
)
from inference_core.llm.config import ModelConfig, ModelProvider
from inference_core.llm.models import without_prompt_cache

from tests.unit.llm.test_anthropic_thinking_blocks import (
    OMITTED_STREAM,
    _build_model,
    _header_transport,
)

SYSTEM = "You help with the user's projects."


@pytest.fixture(autouse=True)
def _disable_llm_emulation(monkeypatch):
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled", lambda: False
    )


def _claude(sent: list, prompt_cache: str | None = "1h"):
    config = ModelConfig(
        name="claude-haiku-5-5",
        provider=ModelProvider.CLAUDE,
        api_key="test-key",
        max_tokens=256,
        prompt_cache=prompt_cache,
    )
    return _build_model(config, _header_transport(sent, OMITTED_STREAM))


def _request(model, system: Any = SYSTEM, messages=None, tools=None):
    return ModelRequest(
        model=model,
        messages=messages or [HumanMessage("What is due this week?")],
        system_message=SystemMessage(system) if system is not None else None,
        tools=tools or [],
    )


def _passed(request):
    """The request the middleware hands on."""
    seen = []

    def handler(req):
        seen.append(req)
        return AIMessage("ok")

    SystemPromptBreakpointMiddleware().wrap_model_call(request, handler)
    return seen[0]


async def _sent_body(request) -> dict:
    """The body the model sends for the request the middleware hands on."""
    passed = _passed(request)
    messages = [passed.system_message, *passed.messages]
    async for _ in passed.model.astream(messages):
        pass
    return passed.model._sent[-1][0]


def _model(prompt_cache: str | None = "1h"):
    sent: list = []
    model = _claude(sent, prompt_cache)
    model.__dict__["_sent"] = sent
    return model


class TestOnTheWire:
    async def test_the_system_prompt_ends_in_a_breakpoint_and_the_automatic_one_stays(
        self,
    ):
        body = await _sent_body(_request(_model("1h")))

        assert body["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert body["system"] == [
            {
                "type": "text",
                "text": SYSTEM,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]

    async def test_the_lifetime_is_the_automatic_ones(self):
        body = await _sent_body(_request(_model("5m")))

        assert body["system"][-1]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "5m",
        }

    async def test_a_model_without_the_cache_is_sent_as_it_is(self):
        body = await _sent_body(_request(_model(None)))

        assert "cache_control" not in body
        assert body["system"] == SYSTEM

    async def test_in_an_agent_the_tools_and_the_system_prompt_are_one_prefix(self):
        @tool
        def list_tasks(week: str) -> str:
            """List the tasks due in a week."""
            return "none"

        model = _model("1h")
        agent = create_agent(
            model,
            tools=[list_tasks],
            system_prompt=SYSTEM,
            middleware=[SystemPromptBreakpointMiddleware()],
        )

        await agent.ainvoke({"messages": [HumanMessage("What is due?")]})

        body = model._sent[-1][0]
        assert body["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert body["system"][-1]["cache_control"]["ttl"] == "1h"
        assert body["system"][-1]["text"] == SYSTEM
        assert [t["name"] for t in body["tools"]] == ["list_tasks"]
        assert "cache_control" not in body["tools"][0]


class TestTheSystemPrompt:
    def test_its_last_text_block_is_marked_and_no_text_changes(self):
        system = SystemMessage(
            [
                {"type": "text", "text": "Role."},
                {"type": "text", "text": "Skills."},
            ]
        )

        marked = marked_system_message(system, "1h")

        assert marked.content == [
            {"type": "text", "text": "Role."},
            {
                "type": "text",
                "text": "Skills.",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        ]
        assert system.content[1] == {"type": "text", "text": "Skills."}

    def test_a_block_marked_already_is_left_alone(self):
        system = SystemMessage(
            [{"type": "text", "text": "Role.", "cache_control": {"type": "ephemeral"}}]
        )

        assert marked_system_message(system, "1h") is None
        request = _request(_model(), system=system.content)
        assert _passed(request) is request

    def test_an_empty_one_is_left_alone(self):
        assert marked_system_message(SystemMessage(""), "1h") is None

    def test_a_request_without_one_is_left_alone(self):
        request = _request(_model(), system=None)

        assert _passed(request) is request


class TestOtherModels:
    def test_another_provider_is_left_alone(self):
        mistral = ChatMistralAI(model="mistral-small-latest", api_key="test-key")
        request = _request(mistral)

        assert _passed(request) is request

    def test_a_model_without_the_cache_is_left_alone(self):
        request = _request(without_prompt_cache(_model("1h")))

        assert automatic_cache_lifetime(request.model) is None
        assert _passed(request) is request

    def test_it_marks_the_model_the_request_is_sent_to(self):
        """A fallback's middleware swaps the request's model; this one sits
        inside it and decides by what it is given."""
        mistral = ChatMistralAI(model="mistral-small-latest", api_key="test-key")
        request = _request(mistral).override(model=_model("1h"))

        passed = _passed(request)

        assert passed.system_message.content[-1]["cache_control"]["ttl"] == "1h"


def _marked_message(text: str) -> HumanMessage:
    return HumanMessage(
        [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    )


class TestFourBreakpoints:
    def test_with_two_of_its_own_the_request_gets_the_fourth(self):
        messages = [_marked_message("a"), _marked_message("b")]
        request = _request(_model(), messages=messages)

        passed = _passed(request)

        assert breakpoints_of(passed.system_message, passed.messages, passed.tools) == 3
        assert 3 + 1 == MAX_BREAKPOINTS

    def test_with_three_of_its_own_it_is_left_as_it_is(self, caplog):
        messages = [_marked_message(t) for t in ("a", "b", "c")]
        request = _request(_model(), messages=messages)

        assert _passed(request) is request
        assert "No cache breakpoint on the system prompt" in caplog.text

    def test_a_tool_marked_counts(self):
        tools = [
            {
                "name": f"t{i}",
                "input_schema": {},
                "cache_control": {"type": "ephemeral"},
            }
            for i in range(3)
        ]
        request = _request(_model(), tools=tools)

        assert _passed(request) is request


async def test_an_async_call_is_marked_too():
    request = _request(_model("1h"))
    seen = []

    async def handler(req):
        seen.append(req)
        return AIMessage("ok")

    await SystemPromptBreakpointMiddleware().awrap_model_call(request, handler)

    assert seen[0].system_message.content[-1]["cache_control"]["ttl"] == "1h"

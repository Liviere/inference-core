"""The routing key of a provider's prompt cache (``prompt_cache_routing``).

A model entry with ``prompt_cache_routing: true`` has every model call of an
agent carry a key that sends the requests of one conversation to the same
server: ``prompt_cache_key`` in the body (OpenAI, Mistral, Fireworks,
DeepInfra) or the ``x-grok-conv-id`` header (xAI). The key is a hash, the
model a request actually goes to decides (a fallback of another provider
gets its own key or none), and without the option nothing is sent.
"""

import json
import re
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import MagicMock

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import inference_core.agents.middleware.prompt_cache_routing as routing
from inference_core.agents.middleware.cost_tracking import CostTrackingMiddleware
from inference_core.agents.middleware.model_fallback import (
    CancelAwareModelFallbackMiddleware,
)
from inference_core.agents.middleware.prompt_cache_routing import (
    PromptCacheRoutingMiddleware,
    routing_key,
    routing_settings,
)
from inference_core.llm.config import ModelConfig, ModelProvider
from inference_core.llm.model_identity import tag_model_identity
from inference_core.services.agents_service import AgentService

THREAD = "5f0c2a8e-1d4b-4c7a-9e2f-3b6d8a1c0e47"


def _entry(name: str, provider: ModelProvider, routed: bool = True) -> ModelConfig:
    return ModelConfig(
        name=name, provider=provider, api_key="test", prompt_cache_routing=routed
    )


def _config(*entries: ModelConfig) -> SimpleNamespace:
    return SimpleNamespace(models={entry.name: entry for entry in entries})


@pytest.fixture
def in_a_thread(monkeypatch):
    monkeypatch.setattr(routing, "_thread_id", lambda: THREAD)


class _Request(SimpleNamespace):
    def override(self, **changes):
        return _Request(**{**vars(self), **changes})


def _request(name: Optional[str], provider: str = "openai", **model_settings):
    model = MagicMock()
    model.metadata = {}
    if name:
        tag_model_identity(model, name=name, provider=provider)
    return _Request(model=model, model_settings=model_settings)


def _settings_sent(middleware, request) -> dict:
    seen = []
    middleware.wrap_model_call(request, lambda r: seen.append(r) or "answer")
    return seen[-1].model_settings


# --------------------------------------------------------------------------- #
# The key
# --------------------------------------------------------------------------- #


def test_the_key_is_the_same_for_a_thread_and_a_model():
    assert routing_key(THREAD, "gpt-6-luna") == routing_key(THREAD, "gpt-6-luna")
    assert routing_key(THREAD, "gpt-6-luna") != routing_key(THREAD, "gpt-5-nano")
    assert routing_key(THREAD, "gpt-6-luna") != routing_key("other", "gpt-6-luna")


def test_the_key_says_nothing_of_the_thread_or_the_model():
    key = routing_key(THREAD, "gpt-6-luna")

    assert re.fullmatch(r"[0-9a-f]{32}", key)
    assert THREAD not in key and "gpt" not in key
    for part in THREAD.split("-"):
        assert part not in key


@pytest.mark.parametrize(
    "provider, expected",
    [
        ("openai", {"prompt_cache_key": "k"}),
        ("mistral", {"prompt_cache_key": "k"}),
        ("fireworks", {"prompt_cache_key": "k"}),
        ("deepinfra", {"prompt_cache_key": "k"}),
        ("xai", {"extra_headers": {"x-grok-conv-id": "k"}}),
        ("claude", {}),
        ("gemini", {}),
        ("cortecs", {}),
        (None, {}),
    ],
)
def test_where_each_provider_takes_it(provider, expected):
    assert routing_settings(provider, "k") == expected


# --------------------------------------------------------------------------- #
# The middleware
# --------------------------------------------------------------------------- #


def test_a_routed_model_gets_the_key(in_a_thread):
    middleware = PromptCacheRoutingMiddleware(
        _config(_entry("gpt-6-luna", ModelProvider.OPENAI))
    )

    sent = _settings_sent(middleware, _request("gpt-6-luna", temperature=0.2))

    assert sent == {
        "temperature": 0.2,
        "prompt_cache_key": routing_key(THREAD, "gpt-6-luna"),
    }


def test_xai_gets_the_header_next_to_the_ones_it_had(in_a_thread):
    middleware = PromptCacheRoutingMiddleware(
        _config(_entry("grok-4.3-latest", ModelProvider.XAI))
    )

    sent = _settings_sent(
        middleware,
        _request("grok-4.3-latest", "xai", extra_headers={"x-other": "1"}),
    )

    assert sent == {
        "extra_headers": {
            "x-other": "1",
            "x-grok-conv-id": routing_key(THREAD, "grok-4.3-latest"),
        }
    }


def test_a_model_without_the_option_goes_as_it_is(in_a_thread):
    middleware = PromptCacheRoutingMiddleware(
        _config(_entry("gpt-6-luna", ModelProvider.OPENAI, routed=False))
    )
    request = _request("gpt-6-luna")

    seen = []
    middleware.wrap_model_call(request, lambda r: seen.append(r))

    assert seen == [request]


def test_a_model_without_a_name_or_an_entry_goes_as_it_is(in_a_thread):
    middleware = PromptCacheRoutingMiddleware(
        _config(_entry("gpt-6-luna", ModelProvider.OPENAI))
    )

    assert _settings_sent(middleware, _request(None)) == {}
    assert _settings_sent(middleware, _request("not-configured")) == {}


def test_a_call_outside_a_thread_goes_as_it_is(monkeypatch):
    monkeypatch.setattr(routing, "_thread_id", lambda: None)
    middleware = PromptCacheRoutingMiddleware(
        _config(_entry("gpt-6-luna", ModelProvider.OPENAI))
    )

    assert _settings_sent(middleware, _request("gpt-6-luna")) == {}


def test_a_provider_that_takes_no_key_is_warned_about_once(in_a_thread, caplog):
    middleware = PromptCacheRoutingMiddleware(
        _config(_entry("claude-haiku-5-5", ModelProvider.CLAUDE))
    )

    with caplog.at_level("WARNING"):
        for _ in range(3):
            assert _settings_sent(middleware, _request("claude-haiku-5-5")) == {}

    assert caplog.text.count("prompt_cache_routing is ignored") == 1


async def test_the_async_call_gets_the_key(in_a_thread):
    middleware = PromptCacheRoutingMiddleware(
        _config(_entry("mistral-small-latest", ModelProvider.MISTRAL))
    )
    seen = []

    async def handler(request):
        seen.append(request)
        return "answer"

    await middleware.awrap_model_call(_request("mistral-small-latest"), handler)

    assert seen[-1].model_settings == {
        "prompt_cache_key": routing_key(THREAD, "mistral-small-latest")
    }


def test_the_thread_is_read_from_the_runs_config():
    from langchain_core.runnables import RunnableLambda

    seen = RunnableLambda(lambda _: routing._thread_id()).invoke(
        None, config={"configurable": {"thread_id": THREAD}}
    )

    assert seen == THREAD
    assert routing._thread_id() is None


# --------------------------------------------------------------------------- #
# In a real graph, with a fallback of another provider
# --------------------------------------------------------------------------- #


class _Recording(BaseChatModel):
    """A model that records the keyword arguments of its call."""

    calls: List[dict] = []
    fails: bool = False

    @property
    def _llm_type(self) -> str:
        return "recording"

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kw: Any
    ) -> ChatResult:
        self.calls.append(kw)
        if self.fails:
            raise RuntimeError("provider down")
        message = AIMessage(content="answer")
        return ChatResult(generations=[ChatGeneration(message=message)])


def _graph_call(primary, fallback, config):
    agent = create_agent(
        primary,
        tools=[],
        middleware=[
            CancelAwareModelFallbackMiddleware(fallback),
            PromptCacheRoutingMiddleware(config),
        ],
    )
    return agent.invoke(
        {"messages": [("user", "hi")]},
        config={"configurable": {"thread_id": THREAD}},
    )


def test_a_fallback_of_another_provider_gets_its_own_key():
    primary = _Recording(calls=[], fails=True)
    tag_model_identity(primary, name="grok-4.3-latest", provider="xai")
    fallback = _Recording(calls=[])
    tag_model_identity(fallback, name="mistral-small-latest", provider="mistral")
    config = _config(
        _entry("grok-4.3-latest", ModelProvider.XAI),
        _entry("mistral-small-latest", ModelProvider.MISTRAL),
    )

    state = _graph_call(primary, fallback, config)

    assert state["messages"][-1].content == "answer"
    [primary_call] = primary.calls
    [fallback_call] = fallback.calls
    assert primary_call["extra_headers"] == {
        "x-grok-conv-id": routing_key(THREAD, "grok-4.3-latest")
    }
    assert "prompt_cache_key" not in primary_call
    assert fallback_call["prompt_cache_key"] == routing_key(
        THREAD, "mistral-small-latest"
    )
    assert "extra_headers" not in fallback_call


def test_a_fallback_that_does_not_ask_for_it_gets_nothing():
    primary = _Recording(calls=[], fails=True)
    tag_model_identity(primary, name="gpt-6-luna", provider="openai")
    fallback = _Recording(calls=[])
    tag_model_identity(fallback, name="claude-haiku-5-5", provider="claude")
    config = _config(
        _entry("gpt-6-luna", ModelProvider.OPENAI),
        _entry("claude-haiku-5-5", ModelProvider.CLAUDE, routed=False),
    )

    _graph_call(primary, fallback, config)

    assert "prompt_cache_key" in primary.calls[0]
    [fallback_call] = fallback.calls
    assert "prompt_cache_key" not in fallback_call
    assert "extra_headers" not in fallback_call


# --------------------------------------------------------------------------- #
# What the real clients send
# --------------------------------------------------------------------------- #


def _capturing(sent: list) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append((dict(request.headers), json.loads(request.content)))
        body = {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def test_chat_openai_sends_the_key_in_the_body():
    from langchain_openai import ChatOpenAI

    sent: list = []
    model = ChatOpenAI(
        model="gpt-6-luna",
        api_key="test",
        http_client=httpx.Client(transport=_capturing(sent)),
    )

    model.invoke("hi", **routing_settings("openai", "k" * 32))

    headers, body = sent[-1]
    assert body["prompt_cache_key"] == "k" * 32


def test_chat_xai_sends_the_key_as_a_header():
    from langchain_xai import ChatXAI

    sent: list = []
    model = ChatXAI(
        model="grok-4.3-latest",
        xai_api_key="test",
        http_client=httpx.Client(transport=_capturing(sent)),
    )

    model.invoke("hi", **routing_settings("xai", "k" * 32))

    headers, body = sent[-1]
    assert headers["x-grok-conv-id"] == "k" * 32
    assert "prompt_cache_key" not in body
    assert "extra_headers" not in body


def test_chat_mistral_sends_the_key_in_the_body():
    from langchain_mistralai import ChatMistralAI

    sent: list = []
    model = ChatMistralAI(model="mistral-small-latest", api_key="test")
    model.client = httpx.Client(
        base_url="https://api.mistral.test/v1", transport=_capturing(sent)
    )

    model.invoke("hi", **routing_settings("mistral", "k" * 32))

    _headers, body = sent[-1]
    assert body["prompt_cache_key"] == "k" * 32


# --------------------------------------------------------------------------- #
# Where the agent service puts it
# --------------------------------------------------------------------------- #


def _service(config) -> AgentService:
    service = object.__new__(AgentService)
    service.model_factory = SimpleNamespace(config=config)
    return service


def _cost_tracking() -> CostTrackingMiddleware:
    return CostTrackingMiddleware(provider="openai", model_name="gpt-6-luna")


def test_it_goes_inside_the_fallback_and_before_cost_tracking():
    service = _service(_config(_entry("gpt-6-luna", ModelProvider.OPENAI)))
    fallback = CancelAwareModelFallbackMiddleware(_Recording(calls=[]))
    cost_tracking = _cost_tracking()
    middleware = [cost_tracking, fallback]

    service._place_prompt_cache_routing(middleware)
    AgentService._ensure_cost_tracking_last(middleware)

    assert [type(m) for m in middleware] == [
        CancelAwareModelFallbackMiddleware,
        PromptCacheRoutingMiddleware,
        CostTrackingMiddleware,
    ]


def test_without_a_routed_model_nothing_is_added():
    service = _service(
        _config(_entry("gpt-6-luna", ModelProvider.OPENAI, routed=False))
    )
    middleware = [_cost_tracking()]

    service._place_prompt_cache_routing(middleware)

    assert [type(m) for m in middleware] == [CostTrackingMiddleware]


def test_one_inherited_from_a_parent_is_moved_inside_the_subagents_fallback():
    config = _config(_entry("gpt-6-luna", ModelProvider.OPENAI))
    service = _service(config)
    inherited = PromptCacheRoutingMiddleware(config)
    fallback = CancelAwareModelFallbackMiddleware(_Recording(calls=[]))
    middleware = [inherited, fallback]

    service._place_prompt_cache_routing(middleware)

    assert middleware[0] is fallback
    assert isinstance(middleware[1], PromptCacheRoutingMiddleware)
    assert len(middleware) == 2


def test_the_option_is_never_a_parameter_of_the_model(monkeypatch):
    from inference_core.llm.models import LLMModelFactory

    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled", lambda: False
    )
    entry = _entry("gpt-6-luna", ModelProvider.OPENAI)
    llm_config = MagicMock()
    llm_config.enable_caching = False
    llm_config.get_model_config.return_value = entry

    model = LLMModelFactory(llm_config).create_model("gpt-6-luna")

    assert "prompt_cache_routing" not in (model.model_kwargs or {})
    assert not hasattr(model, "prompt_cache_routing")


def test_the_example_routes_only_providers_that_take_a_key(monkeypatch):
    from pathlib import Path

    from inference_core.agents.middleware.prompt_cache_routing import (
        ROUTED_PROVIDERS,
    )
    from inference_core.llm.config import LLMConfig

    example = Path(__file__).parents[4] / "llm_config.example.yaml"
    monkeypatch.setenv("LLM_CONFIG_PATH", str(example))
    config = LLMConfig()

    routed = {
        name: str(getattr(model.provider, "value", model.provider))
        for name, model in config.models.items()
        if model.prompt_cache_routing
    }
    assert routed
    assert set(routed.values()) <= set(ROUTED_PROVIDERS)

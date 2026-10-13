import asyncio
import uuid
from typing import Any, List, Optional
from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.errors import GraphInterrupt

from inference_core.agents.middleware.cost_tracking import CostTrackingMiddleware
from inference_core.llm.model_identity import tag_model_identity

from inference_core.agents.middleware.model_fallback import (
    CancelAwareModelFallbackMiddleware,
    ModelAnswerFailed,
    build_model_fallback_middleware,
    canonicalize_fallback_overrides,
    failed_answer_reason,
    fallback_models_from_mapping,
    normalize_fallback_model_names,
)
from inference_core.services._cancel import AgentCancelled


def test_canonicalize_fallback_models_alias() -> None:
    overrides = canonicalize_fallback_overrides(
        {"fallback_models": ["model-b", "model-c"]}
    )

    assert overrides["fallback"] == ["model-b", "model-c"]
    assert overrides["fallback_models"] == ["model-b", "model-c"]


def test_canonicalize_preserves_empty_list_override() -> None:
    overrides = canonicalize_fallback_overrides({"fallback_models": []})

    assert overrides["fallback"] == []
    assert overrides["fallback_models"] == []


def test_canonicalize_empty_overrides_does_not_create_fallback_keys() -> None:
    assert canonicalize_fallback_overrides({}) == {}


def test_canonicalize_unrelated_overrides_does_not_create_fallback_keys() -> None:
    overrides = canonicalize_fallback_overrides({"temperature": 0.2})

    assert overrides == {"temperature": 0.2}
    assert "fallback" not in overrides
    assert "fallback_models" not in overrides


def test_fallback_prefers_canonical_key() -> None:
    value = fallback_models_from_mapping(
        {"fallback": ["canonical"], "fallback_models": ["alias"]}
    )

    assert value == ["canonical"]


def test_normalize_fallback_model_names_removes_primary_duplicates_and_blanks() -> None:
    assert normalize_fallback_model_names(
        ["model-a", "", "model-b", "model-a", None, " model-c "],
        primary_model="model-b",
    ) == ["model-a", "model-c"]


def test_normalize_accepts_single_string() -> None:
    assert normalize_fallback_model_names("model-b", primary_model="model-a") == [
        "model-b"
    ]


class _FakeModelFactory:
    def __init__(self) -> None:
        self.config = type("Config", (), {"models": {"model-b": object()}})()
        self.calls: list[tuple[str, bool]] = []

    def create_model(self, model_name: str, *, reasoning_output: bool = False):
        self.calls.append((model_name, reasoning_output))
        return FakeListChatModel(responses=["ok"])


def test_build_model_fallback_middleware_uses_model_factory() -> None:
    factory = _FakeModelFactory()

    middleware = build_model_fallback_middleware(
        model_factory=factory,
        fallback_models=["model-b"],
        primary_model="model-a",
        reasoning_output=True,
    )

    assert middleware is not None
    assert factory.calls == [("model-b", True)]


def test_build_model_fallback_middleware_returns_cancel_aware_class() -> None:
    middleware = build_model_fallback_middleware(
        model_factory=_FakeModelFactory(),
        fallback_models=["model-b"],
        primary_model="model-a",
    )

    assert isinstance(middleware, CancelAwareModelFallbackMiddleware)
    assert middleware.cancel_check is None


class _FakeRequest:
    """Minimal stand-in for ModelRequest exposing only ``override``."""

    def __init__(self, model: object | None = None) -> None:
        self.model = model

    def override(self, *, model: object) -> "_FakeRequest":
        return _FakeRequest(model=model)


def _build_middleware(num_fallbacks: int = 2) -> CancelAwareModelFallbackMiddleware:
    models = [FakeListChatModel(responses=["ok"]) for _ in range(num_fallbacks)]
    return CancelAwareModelFallbackMiddleware(models[0], *models[1:])


def test_graph_interrupt_propagates_without_fallback() -> None:
    middleware = _build_middleware()
    calls: list[object] = []

    def handler(request: _FakeRequest) -> object:
        calls.append(request)
        raise GraphInterrupt(())

    with pytest.raises(GraphInterrupt):
        middleware.wrap_model_call(_FakeRequest(), handler)

    assert len(calls) == 1


def test_agent_cancelled_propagates_without_fallback() -> None:
    middleware = _build_middleware()
    calls: list[object] = []

    def handler(request: _FakeRequest) -> object:
        calls.append(request)
        raise AgentCancelled("cancelled")

    with pytest.raises(AgentCancelled):
        middleware.wrap_model_call(_FakeRequest(), handler)

    assert len(calls) == 1


def test_regular_exception_still_triggers_fallback() -> None:
    middleware = _build_middleware(num_fallbacks=1)
    calls: list[object] = []

    def handler(request: _FakeRequest) -> object:
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError("provider down")
        return "fallback-response"

    result = middleware.wrap_model_call(_FakeRequest(), handler)

    assert result == "fallback-response"
    assert len(calls) == 2
    assert calls[1].model is middleware.models[0]


def test_graph_interrupt_from_fallback_attempt_propagates() -> None:
    middleware = _build_middleware(num_fallbacks=2)
    calls: list[object] = []

    def handler(request: _FakeRequest) -> object:
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError("provider down")
        raise GraphInterrupt(())

    with pytest.raises(GraphInterrupt):
        middleware.wrap_model_call(_FakeRequest(), handler)

    # Primary + first fallback only — second fallback never attempted.
    assert len(calls) == 2


def test_cancel_check_blocks_fallback_attempts() -> None:
    middleware = _build_middleware()
    middleware.cancel_check = lambda: True
    calls: list[object] = []

    def handler(request: _FakeRequest) -> object:
        calls.append(request)
        raise RuntimeError("provider down")

    with pytest.raises(AgentCancelled):
        middleware.wrap_model_call(_FakeRequest(), handler)

    # Only the primary attempt ran — no paid fallback call started.
    assert len(calls) == 1


def test_broken_cancel_check_does_not_break_fallback() -> None:
    middleware = _build_middleware(num_fallbacks=1)

    def broken_check() -> bool:
        raise RuntimeError("redis unavailable")

    middleware.cancel_check = broken_check
    calls: list[object] = []

    def handler(request: _FakeRequest) -> object:
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError("provider down")
        return "fallback-response"

    assert middleware.wrap_model_call(_FakeRequest(), handler) == "fallback-response"


def test_async_graph_interrupt_propagates_without_fallback() -> None:
    middleware = _build_middleware()
    calls: list[object] = []

    async def handler(request: _FakeRequest) -> object:
        calls.append(request)
        raise GraphInterrupt(())

    async def run() -> None:
        await middleware.awrap_model_call(_FakeRequest(), handler)

    with pytest.raises(GraphInterrupt):
        asyncio.run(run())

    assert len(calls) == 1


def test_async_cancel_check_blocks_fallback_attempts() -> None:
    middleware = _build_middleware()
    middleware.cancel_check = lambda: True
    calls: list[object] = []

    async def handler(request: _FakeRequest) -> object:
        calls.append(request)
        raise RuntimeError("provider down")

    async def run() -> None:
        await middleware.awrap_model_call(_FakeRequest(), handler)

    with pytest.raises(AgentCancelled):
        asyncio.run(run())

    assert len(calls) == 1


def test_async_regular_exception_still_triggers_fallback() -> None:
    middleware = _build_middleware(num_fallbacks=1)
    calls: list[object] = []

    async def handler(request: _FakeRequest) -> object:
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError("provider down")
        return "fallback-response"

    async def run() -> object:
        return await middleware.awrap_model_call(_FakeRequest(), handler)

    assert asyncio.run(run()) == "fallback-response"
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# A fallback that answers is the model the call is logged under
# ---------------------------------------------------------------------------


class _DownModel(BaseChatModel):
    """A model whose provider is down."""

    @property
    def _llm_type(self) -> str:
        return "down"

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kw: Any
    ) -> ChatResult:
        raise RuntimeError("provider down")


class _ReportingModel(BaseChatModel):
    """A model whose response names a snapshot of it."""

    reports: str

    @property
    def _llm_type(self) -> str:
        return "reporting"

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kw: Any
    ) -> ChatResult:
        message = AIMessage(
            content="answer",
            response_metadata={"model_name": self.reports},
            usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_a_call_answered_by_a_fallback_is_logged_as_the_fallback() -> None:
    """The primary fails and leaves no log; the fallback's call is its own."""
    primary = _DownModel()
    tag_model_identity(primary, name="primary-model", provider="mistral")
    fallback = _ReportingModel(reports="fallback-model-2025-08-07")
    tag_model_identity(fallback, name="fallback-model", provider="openai")

    cost_tracking = CostTrackingMiddleware(
        provider="mistral", model_name="primary-model"
    )
    agent = create_agent(
        primary,
        tools=[],
        middleware=[CancelAwareModelFallbackMiddleware(fallback), cost_tracking],
    )

    with patch(
        "inference_core.agents.middleware.cost_tracking.UsageSession"
    ) as MockSession:
        MockSession.return_value.finalize_sync.return_value = uuid.uuid4()
        state = agent.invoke({"messages": [("user", "hi")]})

    assert state["messages"][-1].content == "answer"
    # One log: the attempt that failed wrote none.
    assert MockSession.call_count == 1
    logged = MockSession.call_args.kwargs
    assert logged["model_name"] == "fallback-model"
    assert logged["provider"] == "openai"
    assert logged["reported_model_name"] == "fallback-model-2025-08-07"
    # And that is what the middleware outside reads from the state.
    assert state["last_request_model_name"] == "fallback-model"
    assert state["last_request_provider"] == "openai"
    assert state["last_request_reported_model_name"] == "fallback-model-2025-08-07"


# ---------------------------------------------------------------------------
# A failure that a provider delivers as an answer
# ---------------------------------------------------------------------------


def _error_answer() -> AIMessage:
    """What a stream that went wrong can end with: nothing, and a reason."""
    return AIMessage(content="", response_metadata={"finish_reason": "error"})


def test_an_answer_that_ended_in_an_error_is_a_failure() -> None:
    assert failed_answer_reason(_error_answer()) is not None
    # A list of empty text chunks holds nothing either.
    assert (
        failed_answer_reason(
            AIMessage(content=["", ""], response_metadata={"finish_reason": "ERROR"})
        )
        is not None
    )


def test_what_arrived_before_the_error_does_not_make_it_an_answer() -> None:
    """The sentence that leads up to a tool call, and the call never sent."""
    lead_in = AIMessage(
        content=[
            {"type": "thinking", "thinking": "All done. Call the final tool."},
            "Everything is ready, here is the proposal:",
        ],
        response_metadata={"finish_reason": "error"},
    )
    broken_call = AIMessage(
        content="",
        invalid_tool_calls=[
            {"name": "search", "args": '{"query": "py', "id": "call-1", "error": "cut"}
        ],
        response_metadata={"finish_reason": "error"},
    )

    assert failed_answer_reason(lead_in) is not None
    assert failed_answer_reason(broken_call) is not None


def test_an_answer_with_a_tool_call_to_run_is_an_answer() -> None:
    call = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {}, "id": "call-1"}],
        response_metadata={"finish_reason": "error"},
    )

    assert failed_answer_reason(call) is None


def test_any_other_finish_reason_is_the_models_own() -> None:
    for reason in ("stop", "length", "tool_calls", None):
        empty = AIMessage(content="", response_metadata={"finish_reason": reason})
        text = AIMessage(content="Done.", response_metadata={"finish_reason": reason})
        assert failed_answer_reason(empty) is None
        assert failed_answer_reason(text) is None
    # Whatever a handler returns that is no message is left alone.
    assert failed_answer_reason("fallback-response") is None


def test_a_failed_answer_is_read_off_a_model_response() -> None:
    from langchain.agents.middleware import ModelResponse

    assert failed_answer_reason(ModelResponse(result=[_error_answer()])) is not None
    assert failed_answer_reason(ModelResponse(result=[AIMessage(content="ok")])) is None


def test_a_failed_answer_of_the_primary_goes_to_the_fallback() -> None:
    middleware = _build_middleware(num_fallbacks=1)
    calls: list[object] = []

    def handler(request: _FakeRequest) -> object:
        calls.append(request)
        if len(calls) == 1:
            return _error_answer()
        return AIMessage(content="answer")

    result = middleware.wrap_model_call(_FakeRequest(), handler)

    assert result.content == "answer"
    assert len(calls) == 2
    assert calls[1].model is middleware.models[0]


def test_when_every_model_answers_with_a_failure_the_call_fails() -> None:
    middleware = _build_middleware(num_fallbacks=2)
    calls: list[object] = []

    def handler(request: _FakeRequest) -> object:
        calls.append(request)
        return _error_answer()

    with pytest.raises(ModelAnswerFailed):
        middleware.wrap_model_call(_FakeRequest(), handler)

    assert len(calls) == 3


def test_async_failed_answer_goes_to_the_fallback() -> None:
    middleware = _build_middleware(num_fallbacks=1)
    calls: list[object] = []

    async def handler(request: _FakeRequest) -> object:
        calls.append(request)
        if len(calls) == 1:
            return _error_answer()
        return AIMessage(content="answer")

    async def run() -> object:
        return await middleware.awrap_model_call(_FakeRequest(), handler)

    assert asyncio.run(run()).content == "answer"
    assert len(calls) == 2


class _ErrorAnswerModel(BaseChatModel):
    """A model whose provider ends the call with an error and no output."""

    @property
    def _llm_type(self) -> str:
        return "error-answer"

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kw: Any
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=_error_answer())])


def test_a_failed_answer_never_becomes_a_turn_of_the_conversation() -> None:
    """The empty message would stay in the history a stricter provider refuses."""
    fallback = _ReportingModel(reports="fallback-model")
    agent = create_agent(
        _ErrorAnswerModel(),
        tools=[],
        middleware=[CancelAwareModelFallbackMiddleware(fallback)],
    )

    state = agent.invoke({"messages": [("user", "hi")]})

    ai_messages = [m for m in state["messages"] if m.type == "ai"]
    assert [m.content for m in ai_messages] == ["answer"]


class _LeadInThenErrorModel(BaseChatModel):
    """A model whose provider fails after the sentence that announces a tool call."""

    @property
    def _llm_type(self) -> str:
        return "lead-in-then-error"

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kw: Any
    ) -> ChatResult:
        message = AIMessage(
            content="Everything is ready, here is the proposal:",
            response_metadata={"finish_reason": "error"},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_a_reply_cut_off_by_an_error_is_made_again_by_the_fallback() -> None:
    """Taken as the reply, the lead-in would end the run where the model meant to act."""
    fallback = _ReportingModel(reports="fallback-model")
    agent = create_agent(
        _LeadInThenErrorModel(),
        tools=[],
        middleware=[CancelAwareModelFallbackMiddleware(fallback)],
    )

    state = agent.invoke({"messages": [("user", "hi")]})

    ai_messages = [m for m in state["messages"] if m.type == "ai"]
    assert [m.content for m in ai_messages] == ["answer"]

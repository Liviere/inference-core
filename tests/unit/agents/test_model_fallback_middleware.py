import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.errors import GraphInterrupt

from inference_core.agents.middleware.model_fallback import (
    CancelAwareModelFallbackMiddleware,
    build_model_fallback_middleware,
    canonicalize_fallback_overrides,
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

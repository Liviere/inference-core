from langchain_core.language_models.fake_chat_models import FakeListChatModel

from inference_core.agents.middleware.model_fallback import (
    build_model_fallback_middleware,
    canonicalize_fallback_overrides,
    fallback_models_from_mapping,
    normalize_fallback_model_names,
)


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

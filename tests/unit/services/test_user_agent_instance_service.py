import pytest

from inference_core.services.user_agent_instance_service import UserAgentInstanceService


def test_normalize_config_overrides_canonicalizes_fallback_alias() -> None:
    overrides = UserAgentInstanceService._normalize_and_validate_config_overrides(
        {"fallback_models": ["model-b"]},
        ["model-a", "model-b"],
    )

    assert overrides == {
        "fallback": ["model-b"],
        "fallback_models": ["model-b"],
    }


def test_normalize_config_overrides_preserves_empty_fallback() -> None:
    overrides = UserAgentInstanceService._normalize_and_validate_config_overrides(
        {"fallback_models": []},
        ["model-a"],
    )

    assert overrides == {"fallback": [], "fallback_models": []}


def test_normalize_config_overrides_preserves_unrelated_overrides() -> None:
    overrides = UserAgentInstanceService._normalize_and_validate_config_overrides(
        {"temperature": 0.2},
        ["model-a"],
    )

    assert overrides == {"temperature": 0.2}
    assert "fallback" not in overrides
    assert "fallback_models" not in overrides


def test_normalize_config_overrides_rejects_unknown_fallback_model() -> None:
    with pytest.raises(ValueError, match="Fallback model"):
        UserAgentInstanceService._normalize_and_validate_config_overrides(
            {"fallback": ["missing-model"]},
            ["model-a"],
        )

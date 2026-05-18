import pytest

from inference_core.schemas.agent_prompt_limits import (
    clear_agent_prompt_limits_runtime_override,
    configure_agent_prompt_limits,
    validate_agent_prompt_lengths,
)
from inference_core.services.user_agent_instance_service import UserAgentInstanceService

TEST_SYSTEM_PROMPT_OVERRIDE_MAX_LENGTH = 20
TEST_SYSTEM_PROMPT_APPEND_MAX_LENGTH = 10


@pytest.fixture(autouse=True)
def reset_agent_prompt_limits():
    clear_agent_prompt_limits_runtime_override()
    yield
    clear_agent_prompt_limits_runtime_override()


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


def test_validate_agent_prompt_lengths_allows_unbounded_values_by_default() -> None:
    validate_agent_prompt_lengths(
        system_prompt_override="b" * 50_000,
        system_prompt_append="a" * 50_000,
    )


def test_validate_agent_prompt_lengths_reads_limits_from_env(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_CORE_AGENT_SYSTEM_PROMPT_APPEND_MAX_LENGTH", "10")
    clear_agent_prompt_limits_runtime_override()

    with pytest.raises(ValueError, match="system_prompt_append exceeds 10 characters"):
        validate_agent_prompt_lengths(system_prompt_append="a" * 11)


def test_runtime_prompt_limits_override_env_values(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_CORE_AGENT_SYSTEM_PROMPT_APPEND_MAX_LENGTH", "10")
    configure_agent_prompt_limits(system_prompt_append=7)

    with pytest.raises(ValueError, match="system_prompt_append exceeds 7 characters"):
        validate_agent_prompt_lengths(system_prompt_append="a" * 8)


@pytest.mark.asyncio
async def test_create_instance_rejects_prompt_append_above_shared_limit() -> None:
    service = UserAgentInstanceService(db=None, llm_config_service=None)
    configure_agent_prompt_limits(
        system_prompt_override=TEST_SYSTEM_PROMPT_OVERRIDE_MAX_LENGTH,
        system_prompt_append=TEST_SYSTEM_PROMPT_APPEND_MAX_LENGTH,
    )

    with pytest.raises(
        ValueError,
        match=f"system_prompt_append exceeds {TEST_SYSTEM_PROMPT_APPEND_MAX_LENGTH} characters",
    ):
        await service.create_instance(
            user_id=None,
            instance_name="test-agent",
            display_name="Test Agent",
            base_agent_name="assistant_agent",
            system_prompt_append="a" * (TEST_SYSTEM_PROMPT_APPEND_MAX_LENGTH + 1),
        )


@pytest.mark.asyncio
async def test_create_instance_rejects_prompt_override_above_shared_limit() -> None:
    service = UserAgentInstanceService(db=None, llm_config_service=None)
    configure_agent_prompt_limits(
        system_prompt_override=TEST_SYSTEM_PROMPT_OVERRIDE_MAX_LENGTH,
        system_prompt_append=TEST_SYSTEM_PROMPT_APPEND_MAX_LENGTH,
    )

    with pytest.raises(
        ValueError,
        match=f"system_prompt_override exceeds {TEST_SYSTEM_PROMPT_OVERRIDE_MAX_LENGTH} characters",
    ):
        await service.create_instance(
            user_id=None,
            instance_name="test-agent",
            display_name="Test Agent",
            base_agent_name="assistant_agent",
            system_prompt_override="b" * (TEST_SYSTEM_PROMPT_OVERRIDE_MAX_LENGTH + 1),
        )


@pytest.mark.asyncio
async def test_update_instance_rejects_prompt_append_above_shared_limit() -> None:
    service = UserAgentInstanceService(db=None, llm_config_service=None)
    configure_agent_prompt_limits(
        system_prompt_override=TEST_SYSTEM_PROMPT_OVERRIDE_MAX_LENGTH,
        system_prompt_append=TEST_SYSTEM_PROMPT_APPEND_MAX_LENGTH,
    )

    with pytest.raises(
        ValueError,
        match=f"system_prompt_append exceeds {TEST_SYSTEM_PROMPT_APPEND_MAX_LENGTH} characters",
    ):
        await service.update_instance(
            user_id=None,
            instance_id=None,
            system_prompt_append="a" * (TEST_SYSTEM_PROMPT_APPEND_MAX_LENGTH + 1),
        )


@pytest.mark.asyncio
async def test_update_instance_rejects_prompt_override_above_shared_limit() -> None:
    service = UserAgentInstanceService(db=None, llm_config_service=None)
    configure_agent_prompt_limits(
        system_prompt_override=TEST_SYSTEM_PROMPT_OVERRIDE_MAX_LENGTH,
        system_prompt_append=TEST_SYSTEM_PROMPT_APPEND_MAX_LENGTH,
    )

    with pytest.raises(
        ValueError,
        match=f"system_prompt_override exceeds {TEST_SYSTEM_PROMPT_OVERRIDE_MAX_LENGTH} characters",
    ):
        await service.update_instance(
            user_id=None,
            instance_id=None,
            system_prompt_override="b" * (TEST_SYSTEM_PROMPT_OVERRIDE_MAX_LENGTH + 1),
        )


# ---------- user_selectable enforcement ----------


class _StubLLMConfigService:
    """Minimal stub exposing only ``get_resolved_config``."""

    def __init__(self, resolved):
        self._resolved = resolved

    async def get_resolved_config(self, user_id):  # noqa: D401 - test stub
        return self._resolved


def _make_resolved_config(agents):
    """Build a SimpleNamespace mirroring ResolvedConfig fields used here."""
    from types import SimpleNamespace

    return SimpleNamespace(
        agents=agents,
        available_models=["model-a"],
    )


def _make_agent_config(*, user_selectable: bool):
    from types import SimpleNamespace

    return SimpleNamespace(
        primary_model="model-a",
        fallback_models=[],
        description="stub",
        allowed_tools=[],
        mcp_profile=None,
        local_tool_providers=[],
        skills=[],
        subagents=[],
        user_selectable=user_selectable,
    )


@pytest.mark.asyncio
async def test_create_instance_rejects_internal_only_base_agent() -> None:
    resolved = _make_resolved_config(
        {"agent_builder_intent": _make_agent_config(user_selectable=False)}
    )
    service = UserAgentInstanceService(
        db=None, llm_config_service=_StubLLMConfigService(resolved)
    )

    with pytest.raises(ValueError, match="not available for user instances"):
        await service.create_instance(
            user_id=None,
            instance_name="x",
            display_name="X",
            base_agent_name="agent_builder_intent",
        )


@pytest.mark.asyncio
async def test_list_templates_filters_internal_only_agents() -> None:
    resolved = _make_resolved_config(
        {
            "assistant_agent": _make_agent_config(user_selectable=True),
            "agent_builder_intent": _make_agent_config(user_selectable=False),
        }
    )
    service = UserAgentInstanceService(
        db=None, llm_config_service=_StubLLMConfigService(resolved)
    )

    templates = await service.list_templates()
    names = [t["agent_name"] for t in templates]
    assert names == ["assistant_agent"]

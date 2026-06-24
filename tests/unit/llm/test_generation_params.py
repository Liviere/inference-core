"""
Unit tests for per-agent generation-param overrides.

Covers:
- GenerationParamsOverride schema (all-optional, exclude_none, bounds, extra="allow")
- generation_params field on AgentConfig
- get_model_for_agent() forwarding generation_params into create_model kwargs
- build_model_fallback_middleware() threading generation_params to fallback models
- Claude extended-thinking strips temperature even when set via generation_params
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from inference_core.agents.middleware.model_fallback import (
    build_model_fallback_middleware,
)
from inference_core.llm.config import (
    AgentConfig,
    GenerationParamsOverride,
    ModelConfig,
    ModelProvider,
)
from inference_core.llm.models import LLMModelFactory


@pytest.fixture(autouse=True)
def _disable_llm_emulation(monkeypatch):
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled",
        lambda: False,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestGenerationParamsOverrideSchema:
    def test_all_fields_optional_and_default_none(self):
        gen = GenerationParamsOverride()
        assert gen.as_overrides() == {}

    def test_as_overrides_excludes_unset_keys(self):
        gen = GenerationParamsOverride(temperature=0.2, max_tokens=4096)
        assert gen.as_overrides() == {"temperature": 0.2, "max_tokens": 4096}

    def test_extra_keys_allowed_and_returned(self):
        gen = GenerationParamsOverride(temperature=0.2, top_k=40)
        assert gen.as_overrides() == {"temperature": 0.2, "top_k": 40}

    def test_temperature_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            GenerationParamsOverride(temperature=3.0)

    def test_top_p_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            GenerationParamsOverride(top_p=1.5)


class TestAgentConfigGenerationParams:
    def test_default_is_none(self):
        cfg = AgentConfig(primary="gpt-5")
        assert cfg.generation_params is None

    def test_parsed_from_dict(self):
        cfg = AgentConfig(
            primary="gpt-5",
            generation_params={"temperature": 0.1, "top_p": 0.9},
        )
        assert isinstance(cfg.generation_params, GenerationParamsOverride)
        assert cfg.generation_params.as_overrides() == {
            "temperature": 0.1,
            "top_p": 0.9,
        }

    def test_field_name_avoids_protected_model_namespace(self):
        # A field named ``model_*`` would trip Pydantic's protected namespace.
        # Confirm we exposed it under ``generation_params`` instead.
        assert "generation_params" in AgentConfig.model_fields
        assert not any(name.startswith("model_") for name in AgentConfig.model_fields)


# ---------------------------------------------------------------------------
# get_model_for_agent forwarding (primary model)
# ---------------------------------------------------------------------------


class TestGetModelForAgentGenerationParams:
    def setup_method(self):
        self.mock_llm_config = MagicMock()
        self.mock_llm_config.enable_caching = False
        self.factory = LLMModelFactory(self.mock_llm_config)

    def test_generation_params_forwarded_from_agent_config(self):
        agent_cfg = AgentConfig(
            primary="gpt-5",
            generation_params={"temperature": 0.2, "max_tokens": 4096},
        )
        self.mock_llm_config.get_specific_agent_config.return_value = agent_cfg
        self.mock_llm_config.get_agent_model.return_value = "gpt-5"

        with patch.object(
            self.factory, "create_model", return_value=MagicMock()
        ) as mock_create:
            self.factory.get_model_for_agent("tuned_agent")
            _, kwargs = mock_create.call_args
            assert kwargs.get("temperature") == 0.2
            assert kwargs.get("max_tokens") == 4096

    def test_no_params_when_agent_has_none(self):
        agent_cfg = AgentConfig(primary="gpt-5")
        self.mock_llm_config.get_specific_agent_config.return_value = agent_cfg
        self.mock_llm_config.get_agent_model.return_value = "gpt-5"

        with patch.object(
            self.factory, "create_model", return_value=MagicMock()
        ) as mock_create:
            self.factory.get_model_for_agent("plain_agent")
            _, kwargs = mock_create.call_args
            assert "temperature" not in kwargs

    def test_explicit_kwarg_wins_over_generation_params(self):
        agent_cfg = AgentConfig(
            primary="gpt-5",
            generation_params={"temperature": 0.2},
        )
        self.mock_llm_config.get_specific_agent_config.return_value = agent_cfg
        self.mock_llm_config.get_agent_model.return_value = "gpt-5"

        with patch.object(
            self.factory, "create_model", return_value=MagicMock()
        ) as mock_create:
            self.factory.get_model_for_agent("tuned_agent", temperature=0.9)
            _, kwargs = mock_create.call_args
            assert kwargs.get("temperature") == 0.9


# ---------------------------------------------------------------------------
# Fallback middleware threading
# ---------------------------------------------------------------------------


class _RecordingFactory:
    def __init__(self) -> None:
        self.config = type("Config", (), {"models": {"model-b": object()}})()
        self.calls: list[dict] = []

    def create_model(self, model_name: str, **kwargs):
        self.calls.append({"model_name": model_name, **kwargs})
        return FakeListChatModel(responses=["ok"])


def test_fallback_middleware_threads_generation_params() -> None:
    factory = _RecordingFactory()

    middleware = build_model_fallback_middleware(
        model_factory=factory,
        fallback_models=["model-b"],
        primary_model="model-a",
        reasoning_output=True,
        generation_params={"temperature": 0.2, "max_tokens": 4096},
    )

    assert middleware is not None
    assert factory.calls == [
        {
            "model_name": "model-b",
            "reasoning_output": True,
            "temperature": 0.2,
            "max_tokens": 4096,
        }
    ]


def test_fallback_middleware_without_generation_params() -> None:
    factory = _RecordingFactory()

    build_model_fallback_middleware(
        model_factory=factory,
        fallback_models=["model-b"],
        primary_model="model-a",
    )

    assert factory.calls == [{"model_name": "model-b", "reasoning_output": False}]


# ---------------------------------------------------------------------------
# Claude extended-thinking interaction (temperature stripped downstream)
# ---------------------------------------------------------------------------


class TestClaudeThinkingStripsGenerationTemperature:
    def setup_method(self):
        self.mock_config = MagicMock()
        self.mock_config.enable_caching = False
        self.factory = LLMModelFactory(self.mock_config)

    @patch("inference_core.llm.models.normalize_params")
    @patch("inference_core.llm.models.ChatAnthropic")
    def test_temperature_stripped_under_extended_thinking(
        self, mock_chat_claude, mock_normalize
    ):
        config = ModelConfig(
            name="claude-opus",
            provider=ModelProvider.CLAUDE,
            api_key="test-key",
            reasoning_config={"thinking": {"type": "enabled", "budget_tokens": 5000}},
        )
        # normalize_params would keep temperature; the Claude-thinking guard
        # must drop it afterwards even though it came from generation_params.
        mock_normalize.return_value = {"temperature": 0.2, "max_tokens": 8000}
        mock_chat_claude.return_value = MagicMock()

        self.factory._create_model_instance(
            config, reasoning_output=True, temperature=0.2
        )

        call_kwargs = mock_chat_claude.call_args.kwargs
        assert "temperature" not in call_kwargs

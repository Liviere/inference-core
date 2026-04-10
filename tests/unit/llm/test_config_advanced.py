"""Tests for advanced LLMConfig methods: availability, fallback, overrides, runtime config.

Covers: is_model_available, get_model_params, get_task_model_with_fallback,
get_agent_model_with_fallback, with_overrides, list_available_models,
list_models_by_task, get_provider_runtime_config, is_development_mode,
get_model_debug_info.
"""

import copy
from unittest.mock import patch

import pytest

from inference_core.llm.config import (
    AgentConfig,
    DimensionPrice,
    LLMConfig,
    ModelConfig,
    ModelProvider,
    PricingConfig,
    ProviderConfig,
    TaskConfig,
)

# ---------------------------------------------------------------------------
# Helpers – build an LLMConfig without touching YAML/filesystem
# ---------------------------------------------------------------------------


def _make_config(
    models: dict | None = None,
    providers: dict | None = None,
    task_models: dict | None = None,
    task_configs: dict | None = None,
    agent_models: dict | None = None,
    agent_configs: dict | None = None,
    yaml_config: dict | None = None,
) -> LLMConfig:
    """Build LLMConfig bypassing _load_config (no YAML read)."""
    with patch.object(LLMConfig, "_load_config"):
        cfg = LLMConfig()

    cfg.providers = providers or {
        "openai": {
            "name": "openai",
            "requires_api_key": True,
            "api_key_env": "OPENAI_API_KEY",
        },
        "ollama": {"name": "ollama", "requires_api_key": False},
        "custom_openai_compatible": {
            "name": "custom_openai_compatible",
            "requires_api_key": False,
        },
        "deepinfra": {
            "name": "deepinfra",
            "requires_api_key": True,
            "api_key_env": "DEEPINFRA_API_TOKEN",
        },
        "gemini": {
            "name": "gemini",
            "requires_api_key": True,
            "api_key_env": "GEMINI_API_KEY",
        },
    }

    cfg.models = models or {
        "gpt-5": ModelConfig(
            name="gpt-5",
            provider=ModelProvider.OPENAI,
            api_key="sk-real-key",
            max_tokens=4096,
            temperature=0.7,
        ),
        "gpt-5-mini": ModelConfig(
            name="gpt-5-mini",
            provider=ModelProvider.OPENAI,
            api_key=None,
            max_tokens=2048,
            temperature=0.5,
        ),
        "local-llama": ModelConfig(
            name="local-llama",
            provider=ModelProvider.OLLAMA,
            base_url="http://localhost:11434",
        ),
        "custom-ep": ModelConfig(
            name="custom-ep",
            provider=ModelProvider.CUSTOM_OPENAI_COMPATIBLE,
            base_url="https://my-endpoint.com/v1",
        ),
        "gemini-pro": ModelConfig(
            name="gemini-pro",
            provider=ModelProvider.GEMINI,
            api_key="gemini-key-123",
        ),
    }

    cfg.task_models = task_models or {"completion": "gpt-5", "chat": "gpt-5-mini"}
    cfg.task_configs = task_configs or {}
    cfg.agent_models = agent_models or {"default": "gpt-5"}
    cfg.agent_configs = agent_configs or {}

    if yaml_config is not None:
        cfg._yaml_config = yaml_config

    return cfg


# ===========================================================================
# is_model_available
# ===========================================================================


class TestIsModelAvailable:
    """Test LLMConfig.is_model_available per-provider logic."""

    def test_openai_with_valid_key(self):
        """OpenAI model with real API key is available."""
        cfg = _make_config()
        assert cfg.is_model_available("gpt-5") is True

    def test_openai_without_key(self):
        """OpenAI model without API key is not available."""
        cfg = _make_config()
        assert cfg.is_model_available("gpt-5-mini") is False

    def test_ollama_with_base_url(self):
        """Ollama model with base_url is available."""
        cfg = _make_config()
        assert cfg.is_model_available("local-llama") is True

    def test_ollama_without_base_url_defaults_available(self):
        """Ollama without base_url defaults to available (local runtime)."""
        cfg = _make_config()
        cfg.models["local-llama"].base_url = None
        assert cfg.is_model_available("local-llama") is True

    def test_custom_openai_with_base_url(self):
        """Custom OpenAI-compatible with base_url is available."""
        cfg = _make_config()
        assert cfg.is_model_available("custom-ep") is True

    def test_custom_openai_without_base_url(self):
        """Custom OpenAI-compatible without base_url is not available."""
        cfg = _make_config()
        cfg.models["custom-ep"].base_url = None
        assert cfg.is_model_available("custom-ep") is False

    def test_gemini_with_key(self):
        """Gemini model with API key is available."""
        cfg = _make_config()
        assert cfg.is_model_available("gemini-pro") is True

    def test_gemini_without_key(self):
        """Gemini model without API key is not available."""
        cfg = _make_config()
        cfg.models["gemini-pro"].api_key = None
        assert cfg.is_model_available("gemini-pro") is False

    def test_unknown_model_returns_false(self):
        """Non-existent model returns False."""
        cfg = _make_config()
        assert cfg.is_model_available("nonexistent-model") is False


# ===========================================================================
# get_model_params
# ===========================================================================


class TestGetModelParams:
    """Test LLMConfig.get_model_params extraction."""

    def test_returns_model_params_for_known_model(self):
        """Known model returns ModelParams with correct values."""
        cfg = _make_config()
        params = cfg.get_model_params("gpt-5")
        assert params is not None
        assert params.max_tokens == 4096
        assert params.temperature == 0.7

    def test_returns_none_for_unknown_model(self):
        """Unknown model returns None."""
        cfg = _make_config()
        assert cfg.get_model_params("nonexistent") is None

    def test_excludes_model_specific_fields(self):
        """ModelParams must not contain provider, api_key, base_url, etc."""
        cfg = _make_config()
        params = cfg.get_model_params("gpt-5")
        assert not hasattr(params, "provider")
        assert not hasattr(params, "api_key")
        assert not hasattr(params, "base_url")


# ===========================================================================
# get_task_model_with_fallback
# ===========================================================================


class TestGetTaskModelWithFallback:
    """Test fallback chain: primary → YAML fallback → any available → primary."""

    def test_primary_available_returns_primary(self):
        """When primary model is available, return it."""
        cfg = _make_config()
        assert cfg.get_task_model_with_fallback("completion") == "gpt-5"

    def test_primary_unavailable_uses_yaml_fallback(self):
        """When primary is unavailable, fall back to YAML fallback list."""
        cfg = _make_config(
            task_models={"completion": "gpt-5-mini"},
            yaml_config={
                "tasks": {"completion": {"fallback": ["local-llama", "gpt-5"]}}
            },
        )
        # gpt-5-mini has no API key → unavailable; local-llama is available
        result = cfg.get_task_model_with_fallback("completion")
        assert result == "local-llama"

    def test_all_unavailable_returns_primary(self):
        """When nothing is available, return primary anyway."""
        cfg = _make_config(
            models={
                "no-key": ModelConfig(
                    name="no-key", provider=ModelProvider.OPENAI, api_key=None
                ),
            },
            task_models={"completion": "no-key"},
        )
        assert cfg.get_task_model_with_fallback("completion") == "no-key"

    def test_last_resort_any_available(self):
        """When primary + YAML fallbacks fail, picks any available model."""
        cfg = _make_config(
            task_models={"completion": "gpt-5-mini"},
            yaml_config={"tasks": {"completion": {"fallback": []}}},
        )
        # gpt-5-mini unavailable, no fallback, but gpt-5 is available
        result = cfg.get_task_model_with_fallback("completion")
        assert result == "gpt-5"  # first available in models dict


# ===========================================================================
# get_agent_model_with_fallback
# ===========================================================================


class TestGetAgentModelWithFallback:
    """Test agent model fallback chain."""

    def test_primary_available(self):
        """Agent primary model available → returned directly."""
        cfg = _make_config(agent_models={"default": "gpt-5"})
        assert cfg.get_agent_model_with_fallback("default") == "gpt-5"

    def test_fallback_via_yaml(self):
        """Agent primary unavailable → YAML fallback used."""
        cfg = _make_config(
            agent_models={"default": "gpt-5-mini"},
            yaml_config={"agents": {"default": {"fallback": ["local-llama"]}}},
        )
        result = cfg.get_agent_model_with_fallback("default")
        assert result == "local-llama"

    def test_unknown_agent_gets_default_model(self):
        """Unknown agent name defaults to gpt-5-mini (hardcoded default)."""
        cfg = _make_config()
        model = cfg.get_agent_model("unknown-agent")
        assert model == "gpt-5-mini"


# ===========================================================================
# with_overrides
# ===========================================================================


class TestWithOverrides:
    """Test LLMConfig.with_overrides immutability and merge logic."""

    def test_model_overrides_applied(self):
        """Model parameter overrides are applied in new config."""
        cfg = _make_config()
        new_cfg = cfg.with_overrides(
            model_overrides={"gpt-5": {"temperature": 0.1, "max_tokens": 100}}
        )
        assert new_cfg.models["gpt-5"].temperature == 0.1
        assert new_cfg.models["gpt-5"].max_tokens == 100

    def test_original_unchanged(self):
        """Original config is not mutated."""
        cfg = _make_config()
        original_temp = cfg.models["gpt-5"].temperature
        cfg.with_overrides(model_overrides={"gpt-5": {"temperature": 0.0}})
        assert cfg.models["gpt-5"].temperature == original_temp

    def test_task_overrides_primary(self):
        """Task override with 'primary' updates task_models."""
        cfg = _make_config()
        new_cfg = cfg.with_overrides(
            task_overrides={"completion": {"primary": "local-llama"}}
        )
        assert new_cfg.task_models["completion"] == "local-llama"

    def test_agent_overrides_primary(self):
        """Agent override with 'primary' updates agent_models."""
        cfg = _make_config(
            agent_configs={"default": AgentConfig(primary="gpt-5")},
        )
        new_cfg = cfg.with_overrides(
            agent_overrides={"default": {"primary": "local-llama"}}
        )
        assert new_cfg.agent_models["default"] == "local-llama"

    def test_global_overrides(self):
        """Global overrides update top-level attributes."""
        cfg = _make_config()
        cfg.default_timeout = 60
        new_cfg = cfg.with_overrides(global_overrides={"default_timeout": 120})
        assert new_cfg.default_timeout == 120
        assert cfg.default_timeout == 60  # original unchanged

    def test_global_overrides_do_not_replace_structural_attrs(self):
        """global_overrides must not replace structural attributes like models."""
        cfg = _make_config()
        original_models = set(cfg.models.keys())
        # Even with a 'models' key in global_overrides, the models dict is preserved
        new_cfg = cfg.with_overrides(global_overrides={"models": {"only-one": {}}})
        assert set(new_cfg.models.keys()) == original_models

    def test_no_overrides_returns_copy(self):
        """Calling with no overrides still returns a new instance."""
        cfg = _make_config()
        new_cfg = cfg.with_overrides()
        assert new_cfg is not cfg


# ===========================================================================
# list_available_models / list_models_by_task
# ===========================================================================


class TestListModels:
    """Test model listing helpers."""

    def test_list_available_models(self):
        """Only models with valid credentials are listed."""
        cfg = _make_config()
        available = cfg.list_available_models()
        assert "gpt-5" in available
        assert "gpt-5-mini" not in available  # no API key
        assert "local-llama" in available

    def test_list_models_by_task(self):
        """Returns primary + fallback models for a task."""
        cfg = _make_config(
            yaml_config={"tasks": {"completion": {"fallback": ["local-llama"]}}},
        )
        models = cfg.list_models_by_task("completion")
        assert "gpt-5" in models
        assert "local-llama" in models


# ===========================================================================
# get_provider_runtime_config
# ===========================================================================


class TestGetProviderRuntimeConfig:
    """Test runtime config resolution with env var fallback."""

    def test_returns_provider_raw_data(self):
        """Basic provider config is returned as-is."""
        cfg = _make_config()
        runtime = cfg.get_provider_runtime_config("openai")
        assert runtime["name"] == "openai"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env-key"})
    def test_resolves_api_key_from_env(self):
        """API key is resolved from environment when not inline."""
        cfg = _make_config()
        # Ensure no inline api_key in providers dict
        cfg.providers["openai"].pop("api_key", None)
        runtime = cfg.get_provider_runtime_config("openai")
        assert runtime["api_key"] == "sk-env-key"

    def test_unknown_provider_returns_empty(self):
        """Unknown provider returns empty dict."""
        cfg = _make_config()
        runtime = cfg.get_provider_runtime_config("nonexistent")
        assert runtime == {}


# ===========================================================================
# is_development_mode / get_model_debug_info
# ===========================================================================


class TestDevelopmentMode:
    """Test development mode detection."""

    def test_dev_mode_when_no_models_available(self):
        """Dev mode is True when no models have valid credentials."""
        cfg = _make_config(
            models={
                "no-key": ModelConfig(
                    name="no-key", provider=ModelProvider.OPENAI, api_key=None
                ),
            },
        )
        assert cfg.is_development_mode() is True

    def test_not_dev_mode_when_model_available(self):
        """Dev mode is False when at least one model is available."""
        cfg = _make_config()
        assert cfg.is_development_mode() is False


class TestGetModelDebugInfo:
    """Test debug info generation."""

    def test_debug_info_for_missing_model(self):
        """Debug info returns error for unknown model."""
        cfg = _make_config()
        info = cfg.get_model_debug_info("nope")
        assert info == {"error": "Model not found"}


# ---------------------------------------------------------------------------
# AgentConfig — memory_tools validator
# ---------------------------------------------------------------------------


class TestAgentConfigMemoryTools:
    """Verify memory_tools field validation on AgentConfig."""

    def test_none_is_valid(self):
        """memory_tools=None is accepted (inherit default)."""
        ac = AgentConfig(primary="gpt-5", memory_tools=None)
        assert ac.memory_tools is None

    def test_empty_list_is_valid(self):
        """memory_tools=[] is accepted (disable all tools)."""
        ac = AgentConfig(primary="gpt-5", memory_tools=[])
        assert ac.memory_tools == []

    def test_valid_tool_names(self):
        """All four valid tool names are accepted."""
        ac = AgentConfig(
            primary="gpt-5",
            memory_tools=[
                "save_memory_store",
                "recall_memories_store",
                "update_memory_store",
                "delete_memory_store",
            ],
        )
        assert len(ac.memory_tools) == 4

    def test_invalid_tool_name_raises(self):
        """Invalid tool name raises ValidationError."""
        with pytest.raises(Exception, match="Invalid memory tool names"):
            AgentConfig(primary="gpt-5", memory_tools=["nonexistent_tool"])

    def test_mixed_valid_invalid_raises(self):
        """Mix of valid and invalid names raises ValidationError."""
        with pytest.raises(Exception, match="Invalid memory tool names"):
            AgentConfig(
                primary="gpt-5",
                memory_tools=["save_memory_store", "bad_tool"],
            )

    def test_session_context_and_instructions_defaults(self):
        """memory_session_context_enabled and memory_tool_instructions_enabled default to None."""
        ac = AgentConfig(primary="gpt-5")
        assert ac.memory_session_context_enabled is None
        assert ac.memory_tool_instructions_enabled is None

    def test_session_context_explicit_false(self):
        """memory_session_context_enabled=False is accepted."""
        ac = AgentConfig(primary="gpt-5", memory_session_context_enabled=False)
        assert ac.memory_session_context_enabled is False

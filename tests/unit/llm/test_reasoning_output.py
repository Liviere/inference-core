"""
Unit tests for reasoning output support.

Covers:
- reasoning_config field on ModelConfig
- reasoning_output field on AgentConfig
- _create_model_instance() merging reasoning_config when reasoning_output=True
- get_model_for_agent() auto-forwarding reasoning_output from AgentConfig
"""

from unittest.mock import MagicMock, patch

import pytest

from inference_core.llm.config import AgentConfig, ModelConfig, ModelProvider
from inference_core.llm.models import LLMModelFactory


@pytest.fixture(autouse=True)
def _disable_llm_emulation(monkeypatch):
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled",
        lambda: False,
    )


# ---------------------------------------------------------------------------
# Config schema tests
# ---------------------------------------------------------------------------


class TestModelConfigReasoningConfig:
    """Test reasoning_config field on ModelConfig."""

    def test_default_is_none(self):
        cfg = ModelConfig(name="test", provider=ModelProvider.OPENAI, api_key="k")
        assert cfg.reasoning_config is None

    def test_openai_style(self):
        cfg = ModelConfig(
            name="gpt-5",
            provider=ModelProvider.OPENAI,
            api_key="k",
            reasoning_config={"reasoning": {"effort": "low", "summary": "auto"}},
        )
        assert cfg.reasoning_config == {
            "reasoning": {"effort": "low", "summary": "auto"}
        }

    def test_claude_style(self):
        cfg = ModelConfig(
            name="claude",
            provider=ModelProvider.CLAUDE,
            api_key="k",
            reasoning_config={"thinking": {"type": "enabled", "budget_tokens": 5000}},
        )
        assert cfg.reasoning_config["thinking"]["budget_tokens"] == 5000

    def test_gemini_style(self):
        cfg = ModelConfig(
            name="gemini",
            provider=ModelProvider.GEMINI,
            api_key="k",
            reasoning_config={"thinking_level": "low", "include_thoughts": True},
        )
        assert cfg.reasoning_config["include_thoughts"] is True

    def test_excluded_from_model_dump_exclude_set(self):
        cfg = ModelConfig(
            name="test",
            provider=ModelProvider.OPENAI,
            api_key="k",
            reasoning_config={"reasoning": {"effort": "low"}},
        )
        exclude = {
            "name",
            "provider",
            "api_key",
            "base_url",
            "pricing",
            "display_name",
            "description",
            "reasoning_config",
        }
        dumped = cfg.model_dump(exclude=exclude)
        assert "reasoning_config" not in dumped

    def test_present_in_model_dump_without_exclude(self):
        cfg = ModelConfig(
            name="test",
            provider=ModelProvider.OPENAI,
            api_key="k",
            reasoning_config={"foo": "bar"},
        )
        dumped = cfg.model_dump()
        assert dumped["reasoning_config"] == {"foo": "bar"}


class TestAgentConfigReasoningOutput:
    """Test reasoning_output field on AgentConfig."""

    def test_default_is_false(self):
        cfg = AgentConfig(primary="gpt-5")
        assert cfg.reasoning_output is False

    def test_set_to_true(self):
        cfg = AgentConfig(primary="gpt-5", reasoning_output=True)
        assert cfg.reasoning_output is True


# ---------------------------------------------------------------------------
# Model factory tests
# ---------------------------------------------------------------------------


class TestCreateModelInstanceReasoning:
    """Test _create_model_instance reasoning merge logic."""

    def setup_method(self):
        self.mock_config = MagicMock()
        self.mock_config.enable_caching = False
        self.factory = LLMModelFactory(self.mock_config)

    @patch("inference_core.llm.models.normalize_params")
    @patch("inference_core.llm.models.ChatOpenAI")
    def test_reasoning_config_merged_when_output_true(
        self, mock_chat_openai, mock_normalize
    ):
        """When reasoning_output=True, reasoning_config kwargs are merged into model_params."""
        config = ModelConfig(
            name="gpt-5",
            provider=ModelProvider.OPENAI,
            api_key="test-key",
            temperature=0.7,
            reasoning_config={"reasoning": {"effort": "low", "summary": "auto"}},
        )
        mock_normalize.return_value = {"temperature": 0.7}
        mock_chat_openai.return_value = MagicMock()

        self.factory._create_model_instance(config, reasoning_output=True)

        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs["reasoning"] == {"effort": "low", "summary": "auto"}

    @patch("inference_core.llm.models.normalize_params")
    @patch("inference_core.llm.models.ChatOpenAI")
    def test_reasoning_config_not_merged_when_output_false(
        self, mock_chat_openai, mock_normalize
    ):
        """When reasoning_output=False (default), reasoning_config is NOT applied."""
        config = ModelConfig(
            name="gpt-5",
            provider=ModelProvider.OPENAI,
            api_key="test-key",
            reasoning_config={"reasoning": {"effort": "low"}},
        )
        mock_normalize.return_value = {"temperature": 0.7}
        mock_chat_openai.return_value = MagicMock()

        self.factory._create_model_instance(config)

        call_kwargs = mock_chat_openai.call_args.kwargs
        assert "reasoning" not in call_kwargs

    @patch("inference_core.llm.models.normalize_params")
    @patch("inference_core.llm.models.ChatOpenAI")
    def test_reasoning_config_not_merged_when_config_is_none(
        self, mock_chat_openai, mock_normalize
    ):
        """When model has no reasoning_config, nothing is merged even with reasoning_output=True."""
        config = ModelConfig(
            name="gpt-5",
            provider=ModelProvider.OPENAI,
            api_key="test-key",
        )
        mock_normalize.return_value = {"temperature": 0.7}
        mock_chat_openai.return_value = MagicMock()

        self.factory._create_model_instance(config, reasoning_output=True)

        call_kwargs = mock_chat_openai.call_args.kwargs
        assert "reasoning" not in call_kwargs
        assert "thinking" not in call_kwargs

    @patch("inference_core.llm.models.normalize_params")
    @patch("inference_core.llm.models.ChatOpenAI")
    def test_kwargs_reasoning_config_overrides_model_config(
        self, mock_chat_openai, mock_normalize
    ):
        """reasoning_config passed via kwargs takes precedence over ModelConfig."""
        config = ModelConfig(
            name="gpt-5",
            provider=ModelProvider.OPENAI,
            api_key="test-key",
            reasoning_config={"reasoning": {"effort": "low"}},
        )
        override = {"reasoning": {"effort": "high", "summary": "detailed"}}
        mock_normalize.return_value = {"temperature": 0.7}
        mock_chat_openai.return_value = MagicMock()

        self.factory._create_model_instance(
            config, reasoning_output=True, reasoning_config=override
        )

        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs["reasoning"] == {"effort": "high", "summary": "detailed"}

    @patch("inference_core.llm.models.normalize_params")
    @patch("inference_core.llm.models.ChatOpenAI")
    def test_reasoning_output_not_passed_to_normalize_params(
        self, mock_chat_openai, mock_normalize
    ):
        """reasoning_output and reasoning_config are popped before normalize_params."""
        config = ModelConfig(
            name="gpt-5",
            provider=ModelProvider.OPENAI,
            api_key="test-key",
            reasoning_config={"reasoning": {"effort": "low"}},
        )
        mock_normalize.return_value = {}
        mock_chat_openai.return_value = MagicMock()

        self.factory._create_model_instance(config, reasoning_output=True)

        raw_params = mock_normalize.call_args[0][1]
        assert "reasoning_output" not in raw_params
        assert "reasoning_config" not in raw_params

    @patch("inference_core.llm.models.normalize_params")
    @patch("inference_core.llm.models.ChatGoogleGenerativeAI")
    def test_gemini_reasoning_config_merged(self, mock_chat_gemini, mock_normalize):
        """Gemini reasoning_config (thinking_level, include_thoughts) is merged."""
        config = ModelConfig(
            name="gemini-2.5-flash",
            provider=ModelProvider.GEMINI,
            api_key="test-key",
            reasoning_config={"thinking_level": "low", "include_thoughts": True},
        )
        mock_normalize.return_value = {"temperature": 0.5}
        mock_chat_gemini.return_value = MagicMock()

        self.factory._create_model_instance(config, reasoning_output=True)

        call_kwargs = mock_chat_gemini.call_args.kwargs
        assert call_kwargs["thinking_level"] == "low"
        assert call_kwargs["include_thoughts"] is True

    @patch("inference_core.llm.models.normalize_params")
    @patch("inference_core.llm.models.ChatAnthropic")
    def test_claude_reasoning_config_merged(self, mock_chat_claude, mock_normalize):
        """Claude reasoning_config (thinking) is merged."""
        config = ModelConfig(
            name="claude-sonnet",
            provider=ModelProvider.CLAUDE,
            api_key="test-key",
            reasoning_config={"thinking": {"type": "enabled", "budget_tokens": 5000}},
        )
        mock_normalize.return_value = {"temperature": 1.0}
        mock_chat_claude.return_value = MagicMock()

        self.factory._create_model_instance(config, reasoning_output=True)

        call_kwargs = mock_chat_claude.call_args.kwargs
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 5000}


class TestGetModelForAgentReasoning:
    """Test get_model_for_agent auto-forwarding reasoning_output."""

    def setup_method(self):
        self.mock_llm_config = MagicMock()
        self.mock_llm_config.enable_caching = False
        self.factory = LLMModelFactory(self.mock_llm_config)

    def test_reasoning_output_forwarded_from_agent_config(self):
        """When AgentConfig.reasoning_output=True, kwargs get reasoning_output=True."""
        agent_cfg = AgentConfig(primary="gpt-5", reasoning_output=True)
        self.mock_llm_config.get_specific_agent_config.return_value = agent_cfg
        self.mock_llm_config.get_agent_model.return_value = "gpt-5"

        with patch.object(
            self.factory, "create_model", return_value=MagicMock()
        ) as mock_create:
            self.factory.get_model_for_agent("reasoning_agent")
            _, kwargs = mock_create.call_args
            assert kwargs.get("reasoning_output") is True

    def test_reasoning_output_not_forwarded_when_false(self):
        """When AgentConfig.reasoning_output=False, kwargs do NOT get reasoning_output."""
        agent_cfg = AgentConfig(primary="gpt-5", reasoning_output=False)
        self.mock_llm_config.get_specific_agent_config.return_value = agent_cfg
        self.mock_llm_config.get_agent_model.return_value = "gpt-5"

        with patch.object(
            self.factory, "create_model", return_value=MagicMock()
        ) as mock_create:
            self.factory.get_model_for_agent("plain_agent")
            _, kwargs = mock_create.call_args
            assert "reasoning_output" not in kwargs

    def test_explicit_kwarg_not_overridden(self):
        """Explicit reasoning_output=False in kwargs is not overridden by AgentConfig."""
        agent_cfg = AgentConfig(primary="gpt-5", reasoning_output=True)
        self.mock_llm_config.get_specific_agent_config.return_value = agent_cfg
        self.mock_llm_config.get_agent_model.return_value = "gpt-5"

        with patch.object(
            self.factory, "create_model", return_value=MagicMock()
        ) as mock_create:
            self.factory.get_model_for_agent("reasoning_agent", reasoning_output=False)
            _, kwargs = mock_create.call_args
            assert kwargs.get("reasoning_output") is False

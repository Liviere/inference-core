"""
Unit tests for ToolBasedModelSwitchMiddleware.

Tests the middleware that enables using different models based on tool calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from inference_core.agents.middleware.tool_model_switch import (
    ToolBasedModelSwitchMiddleware,
    ToolModelOverride,
    ToolModelSwitchConfig,
    create_tool_model_switch_middleware,
)

# === Configuration Tests ===
# Tests for configuration dataclasses and validation.


class TestToolModelOverrideConfig:
    """Tests for ToolModelOverride configuration."""

    def test_valid_after_tool_trigger(self):
        """Test creating override with after_tool trigger."""
        override = ToolModelOverride(
            tool_name="my_tool",
            model="gpt-5",
            trigger="after_tool",
            description="Test override",
        )
        assert override.tool_name == "my_tool"
        assert override.model == "gpt-5"
        assert override.trigger == "after_tool"
        assert override.description == "Test override"

    def test_valid_before_tool_trigger(self):
        """Test creating override with before_tool trigger."""
        override = ToolModelOverride(
            tool_name="my_tool",
            model="gpt-5",
            trigger="before_tool",
        )
        assert override.trigger == "before_tool"

    def test_default_trigger_is_after_tool(self):
        """Test that default trigger is after_tool."""
        override = ToolModelOverride(tool_name="my_tool", model="gpt-5")
        assert override.trigger == "after_tool"

    def test_invalid_trigger_raises_error(self):
        """Test that invalid trigger raises ValueError."""
        with pytest.raises(ValueError, match="trigger must be"):
            ToolModelOverride(
                tool_name="my_tool",
                model="gpt-5",
                trigger="invalid_trigger",
            )


class TestToolModelSwitchConfig:
    """Tests for ToolModelSwitchConfig."""

    def test_empty_config(self):
        """Test creating config with no overrides."""
        config = ToolModelSwitchConfig()
        assert config.overrides == []
        assert config.default_model is None
        assert config.cache_models is True

    def test_config_with_overrides(self):
        """Test creating config with multiple overrides."""
        overrides = [
            ToolModelOverride(tool_name="tool1", model="model1"),
            ToolModelOverride(tool_name="tool2", model="model2", trigger="before_tool"),
        ]
        config = ToolModelSwitchConfig(
            overrides=overrides,
            default_model="default-model",
            cache_models=False,
        )
        assert len(config.overrides) == 2
        assert config.default_model == "default-model"
        assert config.cache_models is False


# === Middleware Initialization Tests ===
# Tests for ToolBasedModelSwitchMiddleware initialization.


class TestToolBasedModelSwitchMiddlewareInit:
    """Tests for middleware initialization."""

    def test_init_with_empty_config(self):
        """Test initializing middleware with empty config."""
        config = ToolModelSwitchConfig()
        middleware = ToolBasedModelSwitchMiddleware(config)

        assert middleware.config == config
        assert middleware._after_tool_overrides == {}
        assert middleware._before_tool_overrides == {}

    def test_init_builds_override_lookups(self):
        """Test that initialization builds lookup dicts correctly."""
        overrides = [
            ToolModelOverride(
                tool_name="after_tool1", model="model1", trigger="after_tool"
            ),
            ToolModelOverride(
                tool_name="after_tool2", model="model2", trigger="after_tool"
            ),
            ToolModelOverride(
                tool_name="before_tool1", model="model3", trigger="before_tool"
            ),
        ]
        config = ToolModelSwitchConfig(overrides=overrides)
        middleware = ToolBasedModelSwitchMiddleware(config)

        assert "after_tool1" in middleware._after_tool_overrides
        assert "after_tool2" in middleware._after_tool_overrides
        assert "before_tool1" in middleware._before_tool_overrides
        assert len(middleware._after_tool_overrides) == 2
        assert len(middleware._before_tool_overrides) == 1


# === Model Retrieval Tests ===
# Tests for model caching and retrieval.


class TestModelRetrieval:
    """Tests for model retrieval and caching."""

    @patch("inference_core.agents.middleware.tool_model_switch.init_chat_model")
    def test_get_model_without_factory(self, mock_init_chat_model):
        """Test getting model when no factory is provided."""
        mock_model = MagicMock()
        mock_init_chat_model.return_value = mock_model

        config = ToolModelSwitchConfig()
        middleware = ToolBasedModelSwitchMiddleware(config, model_factory=None)

        result = middleware._get_model("test-model")

        mock_init_chat_model.assert_called_once_with("test-model")
        assert result == mock_model

    @patch("inference_core.agents.middleware.tool_model_switch.init_chat_model")
    def test_get_model_with_caching(self, mock_init_chat_model):
        """Test that models are cached when cache_models=True."""
        mock_model = MagicMock()
        mock_init_chat_model.return_value = mock_model

        config = ToolModelSwitchConfig(cache_models=True)
        middleware = ToolBasedModelSwitchMiddleware(config)

        # First call
        result1 = middleware._get_model("test-model")
        # Second call
        result2 = middleware._get_model("test-model")

        # Should only call init_chat_model once
        assert mock_init_chat_model.call_count == 1
        assert result1 == result2

    @patch("inference_core.agents.middleware.tool_model_switch.init_chat_model")
    def test_get_model_without_caching(self, mock_init_chat_model):
        """Test that models are not cached when cache_models=False."""
        mock_model = MagicMock()
        mock_init_chat_model.return_value = mock_model

        config = ToolModelSwitchConfig(cache_models=False)
        middleware = ToolBasedModelSwitchMiddleware(config)

        # Multiple calls
        middleware._get_model("test-model")
        middleware._get_model("test-model")

        # Should call init_chat_model each time
        assert mock_init_chat_model.call_count == 2

    def test_get_model_with_factory(self):
        """Test getting model when factory is provided."""
        mock_model = MagicMock()
        mock_factory = MagicMock()
        mock_factory.create_model.return_value = mock_model

        config = ToolModelSwitchConfig()
        middleware = ToolBasedModelSwitchMiddleware(config, model_factory=mock_factory)

        result = middleware._get_model("test-model")

        mock_factory.create_model.assert_called_once_with("test-model")
        assert result == mock_model


# === Tool Message Detection Tests ===
# Tests for finding the last tool message in conversation.


class TestFindLastToolMessage:
    """Tests for _find_last_tool_message method."""

    def test_no_tool_messages(self):
        """Test when there are no tool messages."""
        config = ToolModelSwitchConfig()
        middleware = ToolBasedModelSwitchMiddleware(config)

        messages = [
            MagicMock(spec=[]),  # Not a ToolMessage
            MagicMock(spec=[]),
        ]
        result = middleware._find_last_tool_message(messages)
        assert result is None

    def test_single_tool_message(self):
        """Test finding a single tool message."""
        from langchain.messages import ToolMessage

        config = ToolModelSwitchConfig()
        middleware = ToolBasedModelSwitchMiddleware(config)

        tool_msg = ToolMessage(content="result", tool_call_id="123", name="my_tool")
        messages = [MagicMock(spec=[]), tool_msg]

        result = middleware._find_last_tool_message(messages)
        assert result == "my_tool"

    def test_multiple_tool_messages_returns_last(self):
        """Test that the last tool message is returned."""
        from langchain.messages import ToolMessage

        config = ToolModelSwitchConfig()
        middleware = ToolBasedModelSwitchMiddleware(config)

        tool_msg1 = ToolMessage(content="result1", tool_call_id="1", name="first_tool")
        tool_msg2 = ToolMessage(content="result2", tool_call_id="2", name="second_tool")
        messages = [tool_msg1, MagicMock(spec=[]), tool_msg2]

        result = middleware._find_last_tool_message(messages)
        assert result == "second_tool"

    def test_dict_style_tool_message(self):
        """Test finding tool message in dict format."""
        config = ToolModelSwitchConfig()
        middleware = ToolBasedModelSwitchMiddleware(config)

        messages = [
            {"type": "human", "content": "hello"},
            {"type": "tool", "name": "dict_tool", "content": "result"},
        ]

        result = middleware._find_last_tool_message(messages)
        assert result == "dict_tool"


# === Wrap Model Call Tests ===
# Tests for the wrap_model_call hook.


class TestWrapModelCall:
    """Tests for wrap_model_call middleware hook."""

    @patch("inference_core.agents.middleware.tool_model_switch.init_chat_model")
    def test_no_override_passes_through(self, mock_init_chat_model):
        """Test that requests without matching override pass through unchanged."""
        config = ToolModelSwitchConfig(
            overrides=[
                ToolModelOverride(tool_name="other_tool", model="other_model"),
            ]
        )
        middleware = ToolBasedModelSwitchMiddleware(config)

        # Create mock request and handler
        mock_request = MagicMock()
        mock_request.messages = []  # No tool messages
        mock_response = MagicMock()
        mock_handler = MagicMock(return_value=mock_response)

        result = middleware.wrap_model_call(mock_request, mock_handler)

        # Handler should be called with original request
        mock_handler.assert_called_once_with(mock_request)
        assert result == mock_response
        # init_chat_model should not be called
        mock_init_chat_model.assert_not_called()

    @patch("inference_core.agents.middleware.tool_model_switch.init_chat_model")
    def test_after_tool_override_switches_model(self, mock_init_chat_model):
        """Test that after_tool override switches the model."""
        from langchain.messages import ToolMessage

        mock_new_model = MagicMock()
        mock_init_chat_model.return_value = mock_new_model

        config = ToolModelSwitchConfig(
            overrides=[
                ToolModelOverride(
                    tool_name="trigger_tool",
                    model="better_model",
                    trigger="after_tool",
                ),
            ]
        )
        middleware = ToolBasedModelSwitchMiddleware(config)

        # Create request with tool message
        tool_msg = ToolMessage(
            content="result", tool_call_id="123", name="trigger_tool"
        )
        mock_request = MagicMock()
        mock_request.messages = [tool_msg]

        # Track the request passed to handler
        captured_request = None

        def capture_handler(req):
            nonlocal captured_request
            captured_request = req
            return MagicMock()

        mock_handler = MagicMock(side_effect=capture_handler)

        middleware.wrap_model_call(mock_request, mock_handler)

        # Verify model was fetched
        mock_init_chat_model.assert_called_once_with("better_model")
        # Verify request.override was called with new model
        mock_request.override.assert_called_once()


# === Factory Function Tests ===
# Tests for create_tool_model_switch_middleware factory.


class TestFactoryFunction:
    """Tests for create_tool_model_switch_middleware factory function."""

    def test_create_from_dict_config(self):
        """Test creating middleware from dict configuration."""
        overrides = [
            {
                "tool_name": "tool1",
                "model": "model1",
                "trigger": "after_tool",
                "description": "Test description",
            },
            {
                "tool_name": "tool2",
                "model": "model2",
            },  # Uses default trigger
        ]

        middleware = create_tool_model_switch_middleware(
            overrides=overrides,
            default_model="default-model",
            cache_models=False,
        )

        assert isinstance(middleware, ToolBasedModelSwitchMiddleware)
        assert len(middleware.config.overrides) == 2
        assert middleware.config.default_model == "default-model"
        assert middleware.config.cache_models is False

    def test_create_with_model_factory(self):
        """Test creating middleware with model factory."""
        mock_factory = MagicMock()

        middleware = create_tool_model_switch_middleware(
            overrides=[{"tool_name": "t", "model": "m"}],
            model_factory=mock_factory,
        )

        assert middleware._model_factory == mock_factory

    def test_create_empty_overrides(self):
        """Test creating middleware with empty overrides."""
        middleware = create_tool_model_switch_middleware(overrides=[])

        assert len(middleware.config.overrides) == 0


# === Integration with AgentConfig ===
# Tests for integration with LLM config parsing.


class TestAgentConfigIntegration:
    """Tests for AgentConfig tool_model_overrides parsing."""

    def test_agent_config_with_overrides(self):
        """Test AgentConfig parses tool_model_overrides correctly."""
        from inference_core.llm.config import AgentConfig, ToolModelOverrideConfig

        config = AgentConfig(
            primary="gpt-5-mini",
            tool_model_overrides=[
                ToolModelOverrideConfig(
                    tool_name="test_tool",
                    model="gpt-5",
                    trigger="after_tool",
                    description="Test",
                ),
            ],
        )

        assert len(config.tool_model_overrides) == 1
        assert config.tool_model_overrides[0].tool_name == "test_tool"
        assert config.tool_model_overrides[0].model == "gpt-5"

    def test_agent_config_without_overrides(self):
        """Test AgentConfig works without tool_model_overrides."""
        from inference_core.llm.config import AgentConfig

        config = AgentConfig(primary="gpt-5-mini")

        assert config.tool_model_overrides is None

    def test_tool_model_override_config_validation(self):
        """Test ToolModelOverrideConfig validates trigger."""
        from inference_core.llm.config import ToolModelOverrideConfig

        # Valid triggers
        config1 = ToolModelOverrideConfig(
            tool_name="t", model="m", trigger="after_tool"
        )
        assert config1.trigger == "after_tool"

        config2 = ToolModelOverrideConfig(
            tool_name="t", model="m", trigger="before_tool"
        )
        assert config2.trigger == "before_tool"

        # Invalid trigger
        with pytest.raises(ValueError, match="trigger must be one of"):
            ToolModelOverrideConfig(tool_name="t", model="m", trigger="invalid")

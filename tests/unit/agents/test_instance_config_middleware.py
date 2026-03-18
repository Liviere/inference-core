"""Tests for InstanceConfigMiddleware — runtime model/prompt overrides on Agent Server.

Covers:
- Model swap via wrap_model_call when primary_model is in configurable
- System prompt override via wrap_model_call
- System prompt append via wrap_model_call
- No-op pass-through when no overrides are present
- Model caching across calls
- before_agent populates context vars from get_config().configurable
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import SystemMessage

from inference_core.agents.middleware._runtime_context import clear
from inference_core.agents.middleware.instance_config import InstanceConfigMiddleware


@pytest.fixture(autouse=True)
def _clean_context():
    clear()
    yield
    clear()


def _make_request(system_message=None):
    """Build a minimal ModelRequest-like mock."""
    req = MagicMock()
    req.system_message = system_message
    # override returns a new request with replaced fields
    req.override = MagicMock(side_effect=lambda **kw: _apply_override(req, **kw))
    return req


def _apply_override(original, **kwargs):
    """Simulate ModelRequest.override() — returns a new mock with updated attrs."""
    new = MagicMock()
    new.system_message = kwargs.get("system_message", original.system_message)
    new.model = kwargs.get("model", getattr(original, "model", None))
    new.override = MagicMock(side_effect=lambda **kw: _apply_override(new, **kw))
    return new


def _patch_configurable(configurable: dict):
    """Patch get_config() to return a RunnableConfig with given configurable."""
    return patch(
        "inference_core.agents.middleware.instance_config.get_config",
        return_value={"configurable": configurable},
    )


class TestWrapModelCallModelSwap:
    def test_swaps_model_when_primary_model_set(self):
        mock_factory = MagicMock()
        mock_target = MagicMock(name="claude-model")
        mock_factory.create_model.return_value = mock_target

        mw = InstanceConfigMiddleware(model_factory=mock_factory)

        request = _make_request()
        handler = MagicMock(return_value="response")

        with _patch_configurable({"primary_model": "claude-haiku-4-5-20251001"}):
            mw.wrap_model_call(request, handler)

        handler.assert_called_once()
        called_request = handler.call_args[0][0]
        assert called_request.model is mock_target
        mock_factory.create_model.assert_called_once_with("claude-haiku-4-5-20251001")

    def test_no_model_swap_when_no_override(self):
        mw = InstanceConfigMiddleware(model_factory=MagicMock())

        request = _make_request()
        handler = MagicMock(return_value="response")

        with _patch_configurable({}):
            mw.wrap_model_call(request, handler)

        handler.assert_called_once_with(request)

    def test_model_caching(self):
        mock_factory = MagicMock()
        mock_target = MagicMock()
        mock_factory.create_model.return_value = mock_target

        mw = InstanceConfigMiddleware(model_factory=mock_factory)

        handler = MagicMock(return_value="response")

        with _patch_configurable({"primary_model": "claude-haiku-4-5-20251001"}):
            mw.wrap_model_call(_make_request(), handler)
            mw.wrap_model_call(_make_request(), handler)

        # create_model should be called only once due to caching
        mock_factory.create_model.assert_called_once()


class TestWrapModelCallSystemPrompt:
    def test_override_replaces_system_message(self):
        mw = InstanceConfigMiddleware()

        request = _make_request(system_message=SystemMessage(content="Old prompt"))
        handler = MagicMock(return_value="response")

        with _patch_configurable({"system_prompt_override": "New prompt"}):
            mw.wrap_model_call(request, handler)

        called_request = handler.call_args[0][0]
        assert isinstance(called_request.system_message, SystemMessage)
        assert called_request.system_message.content == "New prompt"

    def test_append_concatenates_to_system_message(self):
        mw = InstanceConfigMiddleware()

        request = _make_request(system_message=SystemMessage(content="Base prompt"))
        handler = MagicMock(return_value="response")

        with _patch_configurable({"system_prompt_append": "Extra instructions"}):
            mw.wrap_model_call(request, handler)

        called_request = handler.call_args[0][0]
        assert isinstance(called_request.system_message, SystemMessage)
        assert "Base prompt" in called_request.system_message.content
        assert "Extra instructions" in called_request.system_message.content

    def test_append_alone_when_no_base(self):
        mw = InstanceConfigMiddleware()

        request = _make_request(system_message=None)
        handler = MagicMock(return_value="response")

        with _patch_configurable({"system_prompt_append": "Extra"}):
            mw.wrap_model_call(request, handler)

        called_request = handler.call_args[0][0]
        assert isinstance(called_request.system_message, SystemMessage)
        assert called_request.system_message.content == "Extra"

    def test_override_takes_precedence_over_append(self):
        mw = InstanceConfigMiddleware()

        request = _make_request(system_message=SystemMessage(content="Base"))
        handler = MagicMock(return_value="response")

        with _patch_configurable(
            {
                "system_prompt_override": "Override wins",
                "system_prompt_append": "Should be ignored",
            }
        ):
            mw.wrap_model_call(request, handler)

        called_request = handler.call_args[0][0]
        assert isinstance(called_request.system_message, SystemMessage)
        assert called_request.system_message.content == "Override wins"

    def test_empty_string_override_is_ignored(self):
        """Empty string should NOT trigger override — prevents type corruption."""
        mw = InstanceConfigMiddleware()

        original_sm = SystemMessage(content="Keep me")
        request = _make_request(system_message=original_sm)
        handler = MagicMock(return_value="response")

        with _patch_configurable({"system_prompt_override": ""}):
            mw.wrap_model_call(request, handler)

        # No override applied — original request passed through
        handler.assert_called_once_with(request)

    def test_empty_string_append_is_ignored(self):
        """Empty string should NOT trigger append."""
        mw = InstanceConfigMiddleware()

        original_sm = SystemMessage(content="Keep me")
        request = _make_request(system_message=original_sm)
        handler = MagicMock(return_value="response")

        with _patch_configurable({"system_prompt_append": ""}):
            mw.wrap_model_call(request, handler)

        handler.assert_called_once_with(request)


class TestWrapModelCallCombined:
    def test_model_and_prompt_override_together(self):
        mock_factory = MagicMock()
        mock_target = MagicMock(name="target-model")
        mock_factory.create_model.return_value = mock_target

        mw = InstanceConfigMiddleware(model_factory=mock_factory)

        request = _make_request(system_message=SystemMessage(content="Default prompt"))
        handler = MagicMock(return_value="response")

        with _patch_configurable(
            {
                "primary_model": "claude-haiku-4-5-20251001",
                "system_prompt_override": "Custom prompt",
            }
        ):
            mw.wrap_model_call(request, handler)

        called_request = handler.call_args[0][0]
        assert called_request.model is mock_target
        assert isinstance(called_request.system_message, SystemMessage)
        assert called_request.system_message.content == "Custom prompt"


class TestBeforeAgent:
    def test_populates_context_from_configurable(self):
        mw = InstanceConfigMiddleware()

        runtime = MagicMock()
        state = {}

        with _patch_configurable(
            {
                "primary_model": "gpt-5",
                "session_id": "sess-abc",
            }
        ):
            result = mw.before_agent(state, runtime)

        assert result is None

        from inference_core.agents.middleware._runtime_context import (
            get_primary_model,
            get_session_id,
        )

        assert get_primary_model() == "gpt-5"
        assert get_session_id() == "sess-abc"

    def test_handles_missing_configurable(self):
        mw = InstanceConfigMiddleware()
        runtime = MagicMock(spec=[])
        state = {}

        # get_config raises RuntimeError when called outside runnable context
        with patch(
            "inference_core.agents.middleware.instance_config.get_config",
            side_effect=RuntimeError("no runnable context"),
        ):
            result = mw.before_agent(state, runtime)

        assert result is None


class TestAsyncWrapModelCall:
    async def test_async_swaps_model(self):
        mock_factory = MagicMock()
        mock_target = MagicMock()
        mock_factory.create_model.return_value = mock_target

        mw = InstanceConfigMiddleware(model_factory=mock_factory)

        from unittest.mock import AsyncMock

        request = _make_request()
        handler = AsyncMock(return_value="response")

        with _patch_configurable({"primary_model": "gpt-5"}):
            await mw.awrap_model_call(request, handler)

        handler.assert_awaited_once()
        called_request = handler.call_args[0][0]
        assert called_request.model is mock_target

"""Tests for SubagentConfigMiddleware — runtime subagent-specific overrides.

Covers:
- Model swap when subagent_configs contains matching agent_name
- System prompt override via subagent_configs
- System prompt append via subagent_configs
- No-op pass-through when no matching config
- No-op when subagent_configs is empty or missing
- Model caching across calls
- Override takes precedence: prompt_override over prompt_append
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import SystemMessage

from inference_core.agents.middleware.subagent_config import SubagentConfigMiddleware


def _make_request(system_message=None):
    """Build a minimal ModelRequest-like mock."""
    req = MagicMock()
    req.system_message = system_message
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
        "inference_core.agents.middleware.subagent_config.get_config",
        return_value={"configurable": configurable},
    )


class TestSubagentModelSwap:
    def test_swaps_model_when_matching_config_present(self):
        mock_factory = MagicMock()
        mock_target = MagicMock(name="gemini-model")
        mock_factory.create_model.return_value = mock_target

        mw = SubagentConfigMiddleware(
            agent_name="weather_agent", model_factory=mock_factory
        )

        request = _make_request()
        handler = MagicMock(return_value="response")

        with _patch_configurable(
            {
                "subagent_configs": {
                    "weather_agent": {"primary_model": "gemini-2.5-flash"},
                }
            }
        ):
            mw.wrap_model_call(request, handler)

        handler.assert_called_once()
        called_request = handler.call_args[0][0]
        assert called_request.model is mock_target
        mock_factory.create_model.assert_called_once_with("gemini-2.5-flash")

    def test_no_swap_when_no_matching_agent(self):
        mw = SubagentConfigMiddleware(
            agent_name="weather_agent", model_factory=MagicMock()
        )

        request = _make_request()
        handler = MagicMock(return_value="response")

        with _patch_configurable(
            {
                "subagent_configs": {
                    "other_agent": {"primary_model": "gemini-2.5-flash"},
                }
            }
        ):
            mw.wrap_model_call(request, handler)

        # Original request passed through unchanged
        handler.assert_called_once_with(request)

    def test_no_swap_when_subagent_configs_empty(self):
        mw = SubagentConfigMiddleware(
            agent_name="weather_agent", model_factory=MagicMock()
        )

        request = _make_request()
        handler = MagicMock(return_value="response")

        with _patch_configurable({"subagent_configs": {}}):
            mw.wrap_model_call(request, handler)

        handler.assert_called_once_with(request)

    def test_no_swap_when_subagent_configs_missing(self):
        mw = SubagentConfigMiddleware(
            agent_name="weather_agent", model_factory=MagicMock()
        )

        request = _make_request()
        handler = MagicMock(return_value="response")

        with _patch_configurable({}):
            mw.wrap_model_call(request, handler)

        handler.assert_called_once_with(request)

    def test_model_caching(self):
        mock_factory = MagicMock()
        mock_target = MagicMock()
        mock_factory.create_model.return_value = mock_target

        mw = SubagentConfigMiddleware(
            agent_name="weather_agent", model_factory=mock_factory
        )

        handler = MagicMock(return_value="response")
        configurable = {
            "subagent_configs": {
                "weather_agent": {"primary_model": "gemini-2.5-flash"},
            }
        }

        with _patch_configurable(configurable):
            mw.wrap_model_call(_make_request(), handler)
            mw.wrap_model_call(_make_request(), handler)

        # create_model called only once due to caching
        mock_factory.create_model.assert_called_once()

    def test_ignores_parent_primary_model(self):
        """SubagentConfigMiddleware must NOT read top-level primary_model."""
        mock_factory = MagicMock()
        mw = SubagentConfigMiddleware(
            agent_name="weather_agent", model_factory=mock_factory
        )

        request = _make_request()
        handler = MagicMock(return_value="response")

        with _patch_configurable(
            {
                "primary_model": "claude-haiku-4-5-20251001",
                "subagent_configs": {},
            }
        ):
            mw.wrap_model_call(request, handler)

        # No model swap — parent's primary_model must be ignored
        handler.assert_called_once_with(request)
        mock_factory.create_model.assert_not_called()


class TestSubagentSystemPrompt:
    def test_override_replaces_system_message(self):
        mw = SubagentConfigMiddleware(agent_name="weather_agent")

        request = _make_request(system_message=SystemMessage(content="Old prompt"))
        handler = MagicMock(return_value="response")

        with _patch_configurable(
            {
                "subagent_configs": {
                    "weather_agent": {"system_prompt_override": "New prompt"},
                }
            }
        ):
            mw.wrap_model_call(request, handler)

        called_request = handler.call_args[0][0]
        assert isinstance(called_request.system_message, SystemMessage)
        assert called_request.system_message.content == "New prompt"

    def test_append_concatenates_to_system_message(self):
        mw = SubagentConfigMiddleware(agent_name="weather_agent")

        request = _make_request(system_message=SystemMessage(content="Base prompt"))
        handler = MagicMock(return_value="response")

        with _patch_configurable(
            {
                "subagent_configs": {
                    "weather_agent": {"system_prompt_append": "Extra instructions"},
                }
            }
        ):
            mw.wrap_model_call(request, handler)

        called_request = handler.call_args[0][0]
        assert isinstance(called_request.system_message, SystemMessage)
        assert "Base prompt" in called_request.system_message.content
        assert "Extra instructions" in called_request.system_message.content

    def test_append_alone_when_no_base(self):
        mw = SubagentConfigMiddleware(agent_name="weather_agent")

        request = _make_request(system_message=None)
        handler = MagicMock(return_value="response")

        with _patch_configurable(
            {
                "subagent_configs": {
                    "weather_agent": {"system_prompt_append": "Extra"},
                }
            }
        ):
            mw.wrap_model_call(request, handler)

        called_request = handler.call_args[0][0]
        assert isinstance(called_request.system_message, SystemMessage)
        assert called_request.system_message.content == "Extra"

    def test_override_takes_precedence_over_append(self):
        mw = SubagentConfigMiddleware(agent_name="weather_agent")

        request = _make_request(system_message=SystemMessage(content="Base"))
        handler = MagicMock(return_value="response")

        with _patch_configurable(
            {
                "subagent_configs": {
                    "weather_agent": {
                        "system_prompt_override": "Override wins",
                        "system_prompt_append": "Should be ignored",
                    },
                }
            }
        ):
            mw.wrap_model_call(request, handler)

        called_request = handler.call_args[0][0]
        assert isinstance(called_request.system_message, SystemMessage)
        assert called_request.system_message.content == "Override wins"


class TestSubagentCombined:
    def test_model_and_prompt_override_together(self):
        mock_factory = MagicMock()
        mock_target = MagicMock(name="target-model")
        mock_factory.create_model.return_value = mock_target

        mw = SubagentConfigMiddleware(
            agent_name="weather_agent", model_factory=mock_factory
        )

        request = _make_request(system_message=SystemMessage(content="Default"))
        handler = MagicMock(return_value="response")

        with _patch_configurable(
            {
                "subagent_configs": {
                    "weather_agent": {
                        "primary_model": "gemini-2.5-flash",
                        "system_prompt_append": "Be concise.",
                    },
                }
            }
        ):
            mw.wrap_model_call(request, handler)

        called_request = handler.call_args[0][0]
        assert called_request.model is mock_target
        assert "Be concise." in called_request.system_message.content


class TestSubagentAsyncHook:
    async def test_async_applies_overrides(self):
        mock_factory = MagicMock()
        mock_target = MagicMock(name="gemini-model")
        mock_factory.create_model.return_value = mock_target

        mw = SubagentConfigMiddleware(
            agent_name="weather_agent", model_factory=mock_factory
        )

        request = _make_request()

        async def async_handler(req):
            return "async_response"

        with _patch_configurable(
            {
                "subagent_configs": {
                    "weather_agent": {"primary_model": "gemini-2.5-flash"},
                }
            }
        ):
            result = await mw.awrap_model_call(request, async_handler)

        assert result == "async_response"
        mock_factory.create_model.assert_called_once_with("gemini-2.5-flash")

    async def test_async_no_override_when_missing(self):
        mw = SubagentConfigMiddleware(
            agent_name="weather_agent", model_factory=MagicMock()
        )

        request = _make_request()
        call_args = []

        async def async_handler(req):
            call_args.append(req)
            return "async_response"

        with _patch_configurable({}):
            await mw.awrap_model_call(request, async_handler)

        # Original request passed through
        assert call_args[0] is request


class TestGetConfigUnavailable:
    def test_graceful_fallback_when_get_config_raises(self):
        """When get_config() raises RuntimeError, middleware falls through."""
        mw = SubagentConfigMiddleware(
            agent_name="weather_agent", model_factory=MagicMock()
        )

        request = _make_request()
        handler = MagicMock(return_value="response")

        with patch(
            "inference_core.agents.middleware.subagent_config.get_config",
            side_effect=RuntimeError("No config"),
        ):
            mw.wrap_model_call(request, handler)

        handler.assert_called_once_with(request)

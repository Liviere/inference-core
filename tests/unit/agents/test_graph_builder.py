"""Tests for graph_builder.py — Agent Server graph construction with middleware.

Covers:
- build_agent_graph creates a compiled graph with middleware
- _build_server_middleware constructs CostTracking and ToolModelSwitch
- Middleware is omitted gracefully when config is absent
"""

from unittest.mock import MagicMock, patch

import pytest

from inference_core.agents.graph_builder import (
    _build_server_middleware,
    build_agent_graph,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_agent_config(**overrides):
    defaults = {
        "description": "Test agent",
        "local_tool_providers": [],
        "allowed_tools": None,
        "tool_model_overrides": None,
        "mcp_profile": None,
        "skills": None,
        "subagents": None,
        "interrupt_on": None,
        "execution_mode": "remote",
        "remote_graph_id": None,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _mock_factory(model_name="gpt-5-mini", agent_config=None):
    factory = MagicMock()
    factory.get_agent_model_name.return_value = model_name
    factory.get_model_for_agent.return_value = MagicMock()
    factory.config.get_specific_agent_config.return_value = (
        agent_config or _mock_agent_config()
    )
    return factory


def _mock_llm_config(model_name="gpt-5-mini", pricing=None, provider="openai"):
    cfg = MagicMock()
    model_cfg = MagicMock()
    model_cfg.pricing = pricing
    model_cfg.provider = provider
    cfg.models = {model_name: model_cfg}
    return cfg


# ---------------------------------------------------------------------------
# _build_server_middleware
# ---------------------------------------------------------------------------


class TestBuildServerMiddleware:
    def test_always_includes_instance_config_first(self):
        from inference_core.agents.middleware.instance_config import (
            InstanceConfigMiddleware,
        )

        factory = _mock_factory()
        agent_config = _mock_agent_config()

        with patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ):
            mw = _build_server_middleware("test_agent", agent_config, factory)

        assert len(mw) >= 2
        assert isinstance(mw[0], InstanceConfigMiddleware)

    def test_always_includes_cost_tracking(self):
        from inference_core.agents.middleware.cost_tracking import (
            CostTrackingMiddleware,
        )

        factory = _mock_factory()
        agent_config = _mock_agent_config()

        with patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ):
            mw = _build_server_middleware("test_agent", agent_config, factory)

        assert len(mw) >= 2
        assert isinstance(mw[1], CostTrackingMiddleware)

    def test_cost_tracking_has_no_user_id(self):
        """Agent Server middleware is built without per-request user context."""
        from inference_core.agents.middleware.cost_tracking import (
            CostTrackingMiddleware,
        )

        factory = _mock_factory()
        agent_config = _mock_agent_config()

        with patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ):
            mw = _build_server_middleware("test_agent", agent_config, factory)

        ct = mw[1]
        assert isinstance(ct, CostTrackingMiddleware)
        assert ct.user_id is None
        assert ct.session_id is None

    def test_includes_tool_model_switch_when_overrides_present(self):
        from inference_core.agents.middleware.tool_model_switch import (
            ToolBasedModelSwitchMiddleware,
        )

        override = MagicMock(
            tool_name="check_weather",
            model="gpt-5",
            trigger="after_tool",
            description="switch after weather",
        )
        agent_config = _mock_agent_config(tool_model_overrides=[override])
        factory = _mock_factory(agent_config=agent_config)

        with patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ):
            mw = _build_server_middleware("test_agent", agent_config, factory)

        assert len(mw) == 3
        assert isinstance(mw[2], ToolBasedModelSwitchMiddleware)

    def test_no_tool_model_switch_without_overrides(self):
        factory = _mock_factory()
        agent_config = _mock_agent_config(tool_model_overrides=None)

        with patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ):
            mw = _build_server_middleware("test_agent", agent_config, factory)

        assert len(mw) == 2  # InstanceConfig + CostTracking

    def test_pricing_config_passed_to_cost_tracking(self):
        pricing = MagicMock()
        factory = _mock_factory()
        agent_config = _mock_agent_config()

        with patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(pricing=pricing, provider="deepinfra"),
        ):
            mw = _build_server_middleware("test_agent", agent_config, factory)

        ct = mw[1]
        assert ct.pricing_config is pricing
        assert ct._provider == "deepinfra"

    def test_graceful_on_pricing_load_failure(self):
        factory = _mock_factory()
        agent_config = _mock_agent_config()

        with patch(
            "inference_core.agents.graph_builder.get_llm_config",
            side_effect=RuntimeError("config unavailable"),
        ):
            mw = _build_server_middleware("test_agent", agent_config, factory)

        assert len(mw) >= 2
        # mw[0] = InstanceConfigMiddleware, mw[1] = CostTrackingMiddleware
        assert mw[1].pricing_config is None


# ---------------------------------------------------------------------------
# build_agent_graph
# ---------------------------------------------------------------------------


class TestBuildAgentGraph:
    def test_passes_middleware_to_create_agent(self):
        factory = _mock_factory()

        with patch(
            "inference_core.agents.graph_builder.get_model_factory",
            return_value=factory,
        ), patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ), patch(
            "inference_core.agents.graph_builder.get_registered_providers",
            return_value=[],
        ), patch(
            "inference_core.agents.graph_builder.create_agent"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            build_agent_graph("test_agent")

        call_kwargs = mock_create.call_args[1]
        assert "middleware" in call_kwargs
        assert call_kwargs["middleware"] is not None
        assert len(call_kwargs["middleware"]) >= 1

    def test_graph_returned_is_create_agent_result(self):
        factory = _mock_factory()
        sentinel = object()

        with patch(
            "inference_core.agents.graph_builder.get_model_factory",
            return_value=factory,
        ), patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ), patch(
            "inference_core.agents.graph_builder.get_registered_providers",
            return_value=[],
        ), patch(
            "inference_core.agents.graph_builder.create_agent",
            return_value=sentinel,
        ):
            result = build_agent_graph("test_agent")

        assert result is sentinel

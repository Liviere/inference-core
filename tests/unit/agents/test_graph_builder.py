"""Tests for graph_builder.py — Agent Server graph construction with middleware.

Covers:
- build_agent_graph creates a compiled graph with middleware
- _build_server_middleware constructs CostTracking and ToolModelSwitch
- Middleware is omitted gracefully when config is absent
- _build_deep_agent_middleware adds SubAgentMiddleware + SkillsMiddleware
- _build_server_subagents compiles YAML subagents as CompiledSubAgent
- _build_skills_backend creates FilesystemBackend for SkillsMiddleware
"""

from pathlib import Path
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

    def test_excludes_instance_config_when_flag_false(self):
        """include_instance_config=False suppresses InstanceConfigMiddleware."""
        from inference_core.agents.middleware.cost_tracking import (
            CostTrackingMiddleware,
        )
        from inference_core.agents.middleware.instance_config import (
            InstanceConfigMiddleware,
        )

        factory = _mock_factory()
        agent_config = _mock_agent_config()

        with patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ):
            mw = _build_server_middleware(
                "test_agent",
                agent_config,
                factory,
                include_instance_config=False,
            )

        assert not any(isinstance(m, InstanceConfigMiddleware) for m in mw)
        assert any(isinstance(m, CostTrackingMiddleware) for m in mw)
        # CostTracking should be first when InstanceConfig is excluded
        assert isinstance(mw[0], CostTrackingMiddleware)


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


# ---------------------------------------------------------------------------
# _build_deep_agent_middleware
# ---------------------------------------------------------------------------


class TestBuildDeepAgentMiddleware:
    """Tests for deep-agent middleware injection (SubAgentMiddleware + SkillsMiddleware)."""

    def test_returns_none_when_no_subagents_or_skills(self):
        from inference_core.agents.graph_builder import _build_deep_agent_middleware

        config = _mock_agent_config(subagents=None, skills=None)
        factory = _mock_factory()
        middleware = []

        result = _build_deep_agent_middleware(config, factory, middleware)

        assert result is None
        assert middleware == []

    def test_returns_none_for_empty_lists(self):
        from inference_core.agents.graph_builder import _build_deep_agent_middleware

        config = _mock_agent_config(subagents=[], skills=[])
        factory = _mock_factory()
        middleware = []

        result = _build_deep_agent_middleware(config, factory, middleware)

        assert result is None
        assert middleware == []

    def test_adds_subagent_middleware(self):
        from deepagents.middleware import SubAgentMiddleware

        from inference_core.agents.graph_builder import _build_deep_agent_middleware

        # Build a factory that can resolve the subagent config + model
        sub_config = _mock_agent_config(
            description="Weather helper",
            subagents=None,
            skills=None,
        )
        factory = _mock_factory()
        factory.config.get_specific_agent_config.return_value = sub_config
        factory.get_model_for_agent.return_value = MagicMock()

        parent_config = _mock_agent_config(
            subagents=["weather_agent"],
            skills=None,
        )
        middleware = []

        with patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ), patch(
            "inference_core.agents.graph_builder.get_registered_providers",
            return_value={},
        ), patch(
            "inference_core.agents.graph_builder.create_agent",
            return_value=MagicMock(),
        ):
            result = _build_deep_agent_middleware(parent_config, factory, middleware)

        assert result is None  # No skills → no store
        assert any(isinstance(m, SubAgentMiddleware) for m in middleware)

    def test_adds_skills_middleware_with_backend(self, tmp_path):
        from deepagents.middleware import SkillsMiddleware

        from inference_core.agents.graph_builder import _build_deep_agent_middleware

        # Create a temp skill file
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n\n# Test Skill\n"
        )

        parent_config = _mock_agent_config(
            subagents=None,
            skills=[str(skill_file.relative_to(tmp_path))],
        )
        factory = _mock_factory()
        middleware = []

        with patch("inference_core.agents.graph_builder._PROJECT_ROOT", tmp_path):
            _build_deep_agent_middleware(parent_config, factory, middleware)

        assert any(isinstance(m, SkillsMiddleware) for m in middleware)

    def test_recursion_guard_prevents_circular_subagents(self):
        from inference_core.agents.graph_builder import _build_deep_agent_middleware

        # Agent A lists agent B as subagent, but B is already visited
        parent_config = _mock_agent_config(
            subagents=["already_visited_agent"],
            skills=None,
        )
        factory = _mock_factory()
        middleware = []
        visited = {"already_visited_agent"}

        with patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ), patch(
            "inference_core.agents.graph_builder.get_registered_providers",
            return_value={},
        ):
            _build_deep_agent_middleware(
                parent_config, factory, middleware, visited=visited
            )

        # No SubAgentMiddleware added because subagent was filtered out
        from deepagents.middleware import SubAgentMiddleware

        assert not any(isinstance(m, SubAgentMiddleware) for m in middleware)


# ---------------------------------------------------------------------------
# _build_server_subagents
# ---------------------------------------------------------------------------


class TestBuildServerSubagents:
    def test_compiles_subagent_as_compiled_subagent(self):
        from deepagents import CompiledSubAgent

        from inference_core.agents.graph_builder import _build_server_subagents

        sub_config = _mock_agent_config(
            description="Weather subagent",
            subagents=None,
            skills=None,
            system_prompt="You are weather.",
        )
        factory = _mock_factory()
        factory.config.get_specific_agent_config.return_value = sub_config

        sentinel_graph = MagicMock()

        with patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ), patch(
            "inference_core.agents.graph_builder.get_registered_providers",
            return_value={},
        ), patch(
            "inference_core.agents.graph_builder.create_agent",
            return_value=sentinel_graph,
        ):
            specs = _build_server_subagents(["weather_agent"], factory)

        assert len(specs) == 1
        spec = specs[0]
        assert spec["name"] == "weather_agent"
        assert spec["description"] == "Weather subagent"
        assert spec["runnable"] is sentinel_graph

    def test_skips_visited_subagents(self):
        from inference_core.agents.graph_builder import _build_server_subagents

        factory = _mock_factory()
        visited = {"weather_agent"}

        specs = _build_server_subagents(["weather_agent"], factory, visited=visited)

        assert specs == []

    def test_skips_missing_config(self):
        from inference_core.agents.graph_builder import _build_server_subagents

        factory = _mock_factory()
        factory.config.get_specific_agent_config.return_value = None

        with patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ):
            specs = _build_server_subagents(["nonexistent_agent"], factory)

        assert specs == []

    def test_skips_missing_model(self):
        from inference_core.agents.graph_builder import _build_server_subagents

        sub_config = _mock_agent_config(subagents=None, skills=None)
        factory = _mock_factory()
        factory.config.get_specific_agent_config.return_value = sub_config
        factory.get_model_for_agent.return_value = None

        with patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ):
            specs = _build_server_subagents(["broken_agent"], factory)

        assert specs == []

    def test_subagent_gets_own_server_middleware(self):
        from inference_core.agents.graph_builder import _build_server_subagents

        sub_config = _mock_agent_config(description="Sub", subagents=None, skills=None)
        factory = _mock_factory()
        factory.config.get_specific_agent_config.return_value = sub_config

        with patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ), patch(
            "inference_core.agents.graph_builder.get_registered_providers",
            return_value={},
        ), patch(
            "inference_core.agents.graph_builder.create_agent"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            _build_server_subagents(["sub_agent"], factory)

        # create_agent was called with middleware
        call_kwargs = mock_create.call_args[1]
        assert "middleware" in call_kwargs
        mw_list = call_kwargs["middleware"]
        assert mw_list is not None
        assert len(mw_list) >= 1  # CostTracking (no InstanceConfig)

    def test_subagent_excludes_instance_config_middleware(self):
        """Subagent middleware must NOT include InstanceConfigMiddleware.

        LangGraph propagates the parent's configurable into subgraph
        invocations — if the subagent also had InstanceConfigMiddleware,
        it would read the parent's primary_model / prompt overrides
        and erroneously swap its own model.
        """
        from inference_core.agents.graph_builder import _build_server_subagents
        from inference_core.agents.middleware.cost_tracking import (
            CostTrackingMiddleware,
        )
        from inference_core.agents.middleware.instance_config import (
            InstanceConfigMiddleware,
        )

        sub_config = _mock_agent_config(description="Sub", subagents=None, skills=None)
        factory = _mock_factory()
        factory.config.get_specific_agent_config.return_value = sub_config

        with patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ), patch(
            "inference_core.agents.graph_builder.get_registered_providers",
            return_value={},
        ), patch(
            "inference_core.agents.graph_builder.create_agent"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            _build_server_subagents(["sub_agent"], factory)

        call_kwargs = mock_create.call_args[1]
        mw_list = call_kwargs["middleware"]

        # CostTrackingMiddleware IS included
        assert any(isinstance(m, CostTrackingMiddleware) for m in mw_list)
        # InstanceConfigMiddleware is NOT included
        assert not any(isinstance(m, InstanceConfigMiddleware) for m in mw_list)


# ---------------------------------------------------------------------------
# _build_skills_backend
# ---------------------------------------------------------------------------


class TestBuildSkillsBackend:
    def test_creates_filesystem_backend(self, tmp_path):
        from deepagents.backends.filesystem import FilesystemBackend

        from inference_core.agents.graph_builder import _build_skills_backend

        skill_dir = tmp_path / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: A skill\n---\n\n# My Skill\n"
        )

        with patch("inference_core.agents.graph_builder._PROJECT_ROOT", tmp_path):
            backend, sources = _build_skills_backend(["skills/my-skill/SKILL.md"])

        assert isinstance(backend, FilesystemBackend)
        assert sources == ["skills/"]

    def test_derives_unique_sources(self):
        from inference_core.agents.graph_builder import _build_skills_backend

        with patch("inference_core.agents.graph_builder._PROJECT_ROOT", Path("/fake")):
            _, sources = _build_skills_backend(
                [
                    "skills/skill-a/SKILL.md",
                    "skills/skill-b/SKILL.md",
                    "other/skill-c/SKILL.md",
                ]
            )

        assert "skills/" in sources
        assert "other/" in sources
        assert len(sources) == 2

    def test_backend_can_read_skill_file(self, tmp_path):
        from inference_core.agents.graph_builder import _build_skills_backend

        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Test")

        with patch("inference_core.agents.graph_builder._PROJECT_ROOT", tmp_path):
            backend, _ = _build_skills_backend(["skills/test-skill/SKILL.md"])

        # Verify backend can list and read files
        items = backend.ls_info("skills/")
        assert len(items) >= 1


# ---------------------------------------------------------------------------
# build_agent_graph — deep agent integration
# ---------------------------------------------------------------------------


class TestBuildAgentGraphDeep:
    """Tests that build_agent_graph correctly handles deep agents."""

    def test_passes_no_store_when_skills_present(self, tmp_path):
        """Skills use FilesystemBackend — no store passed to create_agent."""
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: Test\n---\n\n"
        )

        agent_config = _mock_agent_config(
            subagents=None,
            skills=["skills/test-skill/SKILL.md"],
        )
        factory = _mock_factory(agent_config=agent_config)

        with patch(
            "inference_core.agents.graph_builder.get_model_factory",
            return_value=factory,
        ), patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ), patch(
            "inference_core.agents.graph_builder.get_registered_providers",
            return_value={},
        ), patch(
            "inference_core.agents.graph_builder._PROJECT_ROOT", tmp_path
        ), patch(
            "inference_core.agents.graph_builder.create_agent"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            build_agent_graph("deep_agent")

        call_kwargs = mock_create.call_args[1]
        assert "store" not in call_kwargs

    def test_no_store_without_skills(self):
        agent_config = _mock_agent_config(subagents=None, skills=None)
        factory = _mock_factory(agent_config=agent_config)

        with patch(
            "inference_core.agents.graph_builder.get_model_factory",
            return_value=factory,
        ), patch(
            "inference_core.agents.graph_builder.get_llm_config",
            return_value=_mock_llm_config(),
        ), patch(
            "inference_core.agents.graph_builder.get_registered_providers",
            return_value={},
        ), patch(
            "inference_core.agents.graph_builder.create_agent"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            build_agent_graph("test_agent")

        call_kwargs = mock_create.call_args[1]
        assert "store" not in call_kwargs

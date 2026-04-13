"""
Unit tests for ToolCallLimitMiddleware config models and builder.

Tests the Pydantic config models (ToolCallLimitEntry, ToolCallLimitsConfig),
the build_tool_call_limit_middleware factory, and YAML round-trip parsing.
"""

import pytest
from inference_core.agents.middleware.tool_call_limits import (
    build_tool_call_limit_middleware,
    generate_tool_call_limits_instructions,
)
from inference_core.llm.config import (
    AgentConfig,
    ToolCallLimitEntry,
    ToolCallLimitsConfig,
)
from pydantic import ValidationError

# ═══════════════════════════════════════════════════════════════════
# Config model tests
# ═══════════════════════════════════════════════════════════════════


class TestToolCallLimitEntry:
    """Tests for ToolCallLimitEntry Pydantic model."""

    def test_run_limit_only(self):
        entry = ToolCallLimitEntry(run_limit=10)
        assert entry.run_limit == 10
        assert entry.thread_limit is None
        assert entry.exit_behavior == "continue"
        assert entry.tool_name is None

    def test_thread_limit_only(self):
        entry = ToolCallLimitEntry(thread_limit=50)
        assert entry.thread_limit == 50
        assert entry.run_limit is None

    def test_both_limits(self):
        entry = ToolCallLimitEntry(run_limit=10, thread_limit=50)
        assert entry.run_limit == 10
        assert entry.thread_limit == 50

    def test_with_tool_name(self):
        entry = ToolCallLimitEntry(tool_name="browser_get_page_content", run_limit=5)
        assert entry.tool_name == "browser_get_page_content"

    def test_exit_behavior_error(self):
        entry = ToolCallLimitEntry(run_limit=5, exit_behavior="error")
        assert entry.exit_behavior == "error"

    def test_no_limits_raises(self):
        with pytest.raises(ValidationError, match="(?i)at least one"):
            ToolCallLimitEntry()

    def test_invalid_exit_behavior_raises(self):
        with pytest.raises(ValidationError, match="exit_behavior"):
            ToolCallLimitEntry(run_limit=5, exit_behavior="end")

    def test_run_limit_zero_raises(self):
        with pytest.raises(ValidationError):
            ToolCallLimitEntry(run_limit=0)

    def test_negative_thread_limit_raises(self):
        with pytest.raises(ValidationError):
            ToolCallLimitEntry(thread_limit=-1)


class TestToolCallLimitsConfig:
    """Tests for ToolCallLimitsConfig Pydantic model."""

    def test_global_only(self):
        config = ToolCallLimitsConfig(global_limit=ToolCallLimitEntry(run_limit=30))
        assert config.global_limit is not None
        assert config.per_tool == []

    def test_per_tool_only(self):
        config = ToolCallLimitsConfig(
            per_tool=[
                ToolCallLimitEntry(tool_name="tool_a", run_limit=5),
                ToolCallLimitEntry(tool_name="tool_b", thread_limit=20),
            ]
        )
        assert config.global_limit is None
        assert len(config.per_tool) == 2

    def test_global_and_per_tool(self):
        config = ToolCallLimitsConfig(
            global_limit=ToolCallLimitEntry(run_limit=30, thread_limit=100),
            per_tool=[
                ToolCallLimitEntry(tool_name="heavy_tool", run_limit=3),
            ],
        )
        assert config.global_limit.run_limit == 30
        assert len(config.per_tool) == 1

    def test_per_tool_without_name_raises(self):
        with pytest.raises(ValidationError, match="tool_name"):
            ToolCallLimitsConfig(per_tool=[ToolCallLimitEntry(run_limit=5)])

    def test_empty_config_valid(self):
        config = ToolCallLimitsConfig()
        assert config.global_limit is None
        assert config.per_tool == []


class TestAgentConfigToolCallLimits:
    """Tests for tool_call_limits field on AgentConfig."""

    def test_no_limits_defaults_to_none(self):
        agent = AgentConfig(primary="gpt-5-mini")
        assert agent.tool_call_limits is None

    def test_limits_parsed(self):
        agent = AgentConfig(
            primary="gpt-5-mini",
            tool_call_limits=ToolCallLimitsConfig(
                global_limit=ToolCallLimitEntry(run_limit=20),
            ),
        )
        assert agent.tool_call_limits is not None
        assert agent.tool_call_limits.global_limit.run_limit == 20

    def test_limits_from_dict(self):
        """Simulate YAML-style nested dict parsing."""
        agent = AgentConfig(
            primary="gpt-5-mini",
            tool_call_limits={
                "global_limit": {"run_limit": 15, "thread_limit": 60},
                "per_tool": [
                    {"tool_name": "read_file", "run_limit": 3},
                ],
            },
        )
        assert agent.tool_call_limits.global_limit.run_limit == 15
        assert agent.tool_call_limits.global_limit.thread_limit == 60
        assert len(agent.tool_call_limits.per_tool) == 1
        assert agent.tool_call_limits.per_tool[0].tool_name == "read_file"


# ═══════════════════════════════════════════════════════════════════
# Builder / factory tests
# ═══════════════════════════════════════════════════════════════════


class TestBuildToolCallLimitMiddleware:
    """Tests for build_tool_call_limit_middleware factory."""

    def test_none_config_returns_empty(self):
        assert build_tool_call_limit_middleware(None) == []

    def test_global_only_returns_one(self):
        config = ToolCallLimitsConfig(
            global_limit=ToolCallLimitEntry(run_limit=30, thread_limit=100),
        )
        mws = build_tool_call_limit_middleware(config)
        assert len(mws) == 1
        assert mws[0].tool_name is None

    def test_per_tool_only_returns_correct_count(self):
        config = ToolCallLimitsConfig(
            per_tool=[
                ToolCallLimitEntry(tool_name="a", run_limit=5),
                ToolCallLimitEntry(tool_name="b", run_limit=3),
            ],
        )
        mws = build_tool_call_limit_middleware(config)
        assert len(mws) == 2
        assert mws[0].tool_name == "a"
        assert mws[1].tool_name == "b"

    def test_global_plus_per_tool(self):
        config = ToolCallLimitsConfig(
            global_limit=ToolCallLimitEntry(run_limit=30),
            per_tool=[
                ToolCallLimitEntry(tool_name="heavy", run_limit=5),
                ToolCallLimitEntry(tool_name="heavier", run_limit=2),
            ],
        )
        mws = build_tool_call_limit_middleware(config)
        assert len(mws) == 3
        # Global first, then per-tool
        assert mws[0].tool_name is None
        assert mws[1].tool_name == "heavy"
        assert mws[2].tool_name == "heavier"

    def test_empty_config_returns_empty(self):
        config = ToolCallLimitsConfig()
        assert build_tool_call_limit_middleware(config) == []

    def test_middleware_params_match_config(self):
        config = ToolCallLimitsConfig(
            global_limit=ToolCallLimitEntry(
                run_limit=10,
                thread_limit=40,
                exit_behavior="error",
            ),
        )
        mws = build_tool_call_limit_middleware(config)
        mw = mws[0]
        assert mw.run_limit == 10
        assert mw.thread_limit == 40
        assert mw.exit_behavior == "error"


# ═══════════════════════════════════════════════════════════════════
# generate_tool_call_limits_instructions tests
# ═══════════════════════════════════════════════════════════════════


class TestGenerateToolCallLimitsInstructions:
    """Tests for generate_tool_call_limits_instructions()."""

    def test_none_config_returns_none(self):
        assert generate_tool_call_limits_instructions(None) is None

    def test_empty_config_returns_none(self):
        config = ToolCallLimitsConfig()
        assert generate_tool_call_limits_instructions(config) is None

    def test_global_limit_returns_instructions(self):
        config = ToolCallLimitsConfig(
            global_limit=ToolCallLimitEntry(run_limit=30),
        )
        result = generate_tool_call_limits_instructions(config)
        assert result is not None
        assert isinstance(result, str)

    def test_per_tool_only_returns_instructions(self):
        config = ToolCallLimitsConfig(
            per_tool=[ToolCallLimitEntry(tool_name="my_tool", run_limit=5)],
        )
        result = generate_tool_call_limits_instructions(config)
        assert result is not None

    def test_both_limits_returns_instructions(self):
        config = ToolCallLimitsConfig(
            global_limit=ToolCallLimitEntry(run_limit=30),
            per_tool=[ToolCallLimitEntry(tool_name="my_tool", run_limit=5)],
        )
        result = generate_tool_call_limits_instructions(config)
        assert result is not None

    def test_instructions_contain_key_phrases(self):
        config = ToolCallLimitsConfig(
            global_limit=ToolCallLimitEntry(run_limit=30),
        )
        result = generate_tool_call_limits_instructions(config)
        assert "Summarize progress" in result
        assert "Report blockers" in result
        assert "Plan next steps" in result

    def test_instructions_explain_run_vs_thread_limits(self):
        config = ToolCallLimitsConfig(
            global_limit=ToolCallLimitEntry(run_limit=30),
        )
        result = generate_tool_call_limits_instructions(config)
        assert "Run limit" in result
        assert "Thread limit" in result
        assert "resets" in result.lower()

    def test_instructions_allow_continuation(self):
        config = ToolCallLimitsConfig(
            global_limit=ToolCallLimitEntry(run_limit=30),
        )
        result = generate_tool_call_limits_instructions(config)
        assert "continue" in result.lower()
        assert "do not refuse" in result.lower()

    def test_instructions_contain_policy_header(self):
        config = ToolCallLimitsConfig(
            global_limit=ToolCallLimitEntry(run_limit=10),
        )
        result = generate_tool_call_limits_instructions(config)
        assert "Tool-Call Limit Policy" in result

    def test_instructions_show_global_run_limit(self):
        config = ToolCallLimitsConfig(
            global_limit=ToolCallLimitEntry(run_limit=30, thread_limit=120),
        )
        result = generate_tool_call_limits_instructions(config)
        assert "Your current limits" in result
        assert "30 calls per invocation" in result
        assert "120 calls per conversation" in result

    def test_instructions_show_per_tool_limits(self):
        config = ToolCallLimitsConfig(
            per_tool=[
                ToolCallLimitEntry(tool_name="heavy_tool", run_limit=5, thread_limit=20),
                ToolCallLimitEntry(tool_name="light_tool", run_limit=10),
            ],
        )
        result = generate_tool_call_limits_instructions(config)
        assert "`heavy_tool`: run=5, thread=20" in result
        assert "`light_tool`: run=10" in result

    def test_instructions_global_run_only(self):
        config = ToolCallLimitsConfig(
            global_limit=ToolCallLimitEntry(run_limit=15),
        )
        result = generate_tool_call_limits_instructions(config)
        assert "15 calls per invocation" in result
        assert "calls per conversation" not in result

    def test_instructions_global_thread_only(self):
        config = ToolCallLimitsConfig(
            global_limit=ToolCallLimitEntry(thread_limit=50),
        )
        result = generate_tool_call_limits_instructions(config)
        assert "50 calls per conversation" in result
        assert "calls per invocation" not in result

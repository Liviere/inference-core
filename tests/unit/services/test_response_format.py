"""Tests for structured-output (``response_format``) plumbing.

Covers:
  - ``AgentConfig.response_format`` validator (accept/reject shapes).
  - ``_RunAccumulator`` captures ``structured_response`` from updates chunks.
  - ``UserAgentInstanceService._normalize_and_validate_config_overrides``
    validates ``config_overrides.response_format``.
  - ``AgentService.build_config_for_instance`` forwards ``response_format``
    from DB overrides into ``agent_overrides``.
"""

from unittest.mock import MagicMock, patch

import pytest

from inference_core.llm.config import AgentConfig
from inference_core.services.agents_service import (
    AgentService,
    _RunAccumulator,
)

# ---------------------------------------------------------------------------
# AgentConfig.response_format validator
# ---------------------------------------------------------------------------


class TestAgentConfigResponseFormat:
    def test_accepts_json_schema_with_type(self):
        cfg = AgentConfig(
            primary="gpt-5-mini",
            response_format={
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        )
        assert cfg.response_format == {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

    def test_accepts_ref_shape(self):
        cfg = AgentConfig(
            primary="gpt-5-mini",
            response_format={"$ref": "#/definitions/MySchema"},
        )
        assert cfg.response_format["$ref"] == "#/definitions/MySchema"

    @pytest.mark.parametrize("variant", ["oneOf", "anyOf", "allOf"])
    def test_accepts_composition_keywords(self, variant):
        cfg = AgentConfig(
            primary="gpt-5-mini",
            response_format={variant: [{"type": "string"}, {"type": "integer"}]},
        )
        assert variant in cfg.response_format

    def test_defaults_to_none(self):
        cfg = AgentConfig(primary="gpt-5-mini")
        assert cfg.response_format is None

    def test_rejects_empty_dict(self):
        with pytest.raises(ValueError, match="response_format"):
            AgentConfig(primary="gpt-5-mini", response_format={})

    def test_rejects_non_dict(self):
        with pytest.raises(ValueError):
            AgentConfig(primary="gpt-5-mini", response_format="not-a-dict")

    def test_rejects_dict_without_schema_markers(self):
        with pytest.raises(ValueError, match="JSON Schema"):
            AgentConfig(
                primary="gpt-5-mini",
                response_format={"properties": {"x": {"type": "string"}}},
            )


# ---------------------------------------------------------------------------
# _RunAccumulator.process_updates_chunk captures structured_response
# ---------------------------------------------------------------------------


class TestAccumulatorStructuredResponse:
    def test_captures_structured_response_from_updates(self):
        acc = _RunAccumulator()
        payload = {"name": "Ada", "email": "ada@example.com"}
        acc.process_updates_chunk(
            {"final": {"structured_response": payload, "messages": []}},
            on_step=None,
        )
        assert acc.structured_response == payload

    def test_keeps_latest_when_multiple_chunks(self):
        acc = _RunAccumulator()
        acc.process_updates_chunk(
            {"step1": {"structured_response": {"v": 1}}}, on_step=None
        )
        acc.process_updates_chunk(
            {"step2": {"structured_response": {"v": 2}}}, on_step=None
        )
        assert acc.structured_response == {"v": 2}

    def test_remains_none_when_no_structured_response(self):
        acc = _RunAccumulator()
        acc.process_updates_chunk({"step": {"messages": [MagicMock()]}}, on_step=None)
        assert acc.structured_response is None

    def test_build_response_propagates_structured_response(self):
        from datetime import UTC, datetime

        acc = _RunAccumulator()
        acc.structured_response = {"answer": 42}
        resp = acc.build_response(
            model_name="gpt-5-mini",
            start_time=datetime.now(UTC),
            enable_cost_tracking=False,
        )
        assert resp.structured_response == {"answer": 42}


# ---------------------------------------------------------------------------
# build_config_for_instance forwards response_format override
# ---------------------------------------------------------------------------


class TestBuildConfigResponseFormatOverride:
    def test_response_format_forwarded_as_agent_override(self):
        """response_format in config_overrides becomes an agent-level override."""
        mock_base = MagicMock()
        mock_base.agent_models = {"my_agent": "gpt-4o"}
        mock_base.with_overrides.return_value = mock_base

        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        AgentService.build_config_for_instance(
            {
                "base_agent_name": "my_agent",
                "config_overrides": {"response_format": schema},
            },
            base_config=mock_base,
        )

        mock_base.with_overrides.assert_called_once()
        kwargs = mock_base.with_overrides.call_args.kwargs
        agent_overrides = kwargs.get("agent_overrides") or {}
        assert "my_agent" in agent_overrides
        assert agent_overrides["my_agent"].get("response_format") == schema


# ---------------------------------------------------------------------------
# UserAgentInstanceService config_overrides validation
# ---------------------------------------------------------------------------


class TestUserAgentInstanceResponseFormatValidation:
    def _validate(self, overrides):
        from inference_core.services.user_agent_instance_service import (
            UserAgentInstanceService,
        )

        return UserAgentInstanceService._normalize_and_validate_config_overrides(
            overrides, available_models=[]
        )

    def test_accepts_valid_response_format(self):
        out = self._validate({"response_format": {"type": "object"}})
        assert out["response_format"] == {"type": "object"}

    def test_rejects_empty_dict(self):
        with pytest.raises(ValueError, match="response_format"):
            self._validate({"response_format": {}})

    def test_rejects_non_dict(self):
        with pytest.raises(ValueError, match="response_format"):
            self._validate({"response_format": "nope"})

    def test_rejects_dict_without_schema_markers(self):
        with pytest.raises(ValueError, match="JSON Schema"):
            self._validate(
                {"response_format": {"properties": {"x": {"type": "string"}}}}
            )

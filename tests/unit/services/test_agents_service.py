"""Tests for AgentService.

Covers testable methods in isolation: _sync_connection_string, _build_provider_user_context,
_build_system_prompt, _build_middleware, run_agent_steps, and Pydantic models.

Heavy constructor side-effects (model loading, embeddings) are bypassed by mocking
the constructor's dependencies or by testing individual methods on a pre-built instance.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from inference_core.services.agents_service import (
    AgentCostMetrics,
    AgentMetadata,
    AgentResponse,
    AgentService,
    DeepAgentService,
)

# ---------------------------------------------------------------------------
# Fixtures – build AgentService with all external deps mocked
# ---------------------------------------------------------------------------


def _make_agent_service(**overrides):
    """Create AgentService with all heavy constructor deps mocked.

    Patches get_model_factory, get_llm_config, get_settings so the constructor
    does not touch real YAML files, embeddings, or database connections.
    """
    mock_model_factory = MagicMock()
    mock_model = MagicMock()
    mock_model_factory.get_agent_model_name.return_value = "gpt-4o"
    mock_model_factory.get_model_for_agent.return_value = mock_model
    mock_model_factory.config.get_specific_agent_config.return_value = MagicMock(
        local_tool_providers=[],
        mcp_profile=None,
        allowed_tools=None,
        tool_model_overrides=None,
        description="Test agent",
        skills=None,
        subagents=None,
        interrupt_on=None,
    )

    defaults = {
        "agent_name": "test_agent",
        "tools": None,
        "use_checkpoints": False,
        "use_memory": False,
        "checkpoint_config": None,
        "enable_cost_tracking": True,
        "user_id": uuid.uuid4(),
        "session_id": "sess-1",
        "request_id": "req-1",
    }
    defaults.update(overrides)

    with patch(
        "inference_core.services.agents_service.get_model_factory",
        return_value=mock_model_factory,
    ), patch(
        "inference_core.services.agents_service.get_llm_config",
        return_value=MagicMock(),
    ), patch(
        "inference_core.services.agents_service.get_settings",
        return_value=MagicMock(
            database_url="sqlite+aiosqlite:///test.db",
            agent_memory_enabled=False,
        ),
    ):
        svc = AgentService(**defaults)

    # Attach mock references for assertions
    svc._mock_model_factory = mock_model_factory
    svc._mock_model = mock_model
    return svc


@pytest.fixture
def agent_service():
    return _make_agent_service()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestPydanticModels:
    """Verify AgentMetadata, AgentCostMetrics, AgentResponse schemas."""

    def test_agent_metadata_fields(self):
        """AgentMetadata stores model name, tools, timestamps."""
        now = datetime.now(UTC)
        meta = AgentMetadata(
            model_name="gpt-4o",
            tools_used=["search", "calculator"],
            start_time=now,
            end_time=now,
        )
        assert meta.model_name == "gpt-4o"
        assert len(meta.tools_used) == 2

    def test_agent_cost_metrics_defaults(self):
        """AgentCostMetrics defaults all counters to zero."""
        metrics = AgentCostMetrics()
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.total_tokens == 0
        assert metrics.extra_tokens == {}
        assert metrics.model_call_count == 0

    def test_agent_response_optional_cost(self):
        """AgentResponse allows cost_metrics to be None."""
        now = datetime.now(UTC)
        resp = AgentResponse(
            result={"messages": []},
            steps=[],
            metadata=AgentMetadata(
                model_name="gpt-4o", tools_used=[], start_time=now
            ),
            cost_metrics=None,
        )
        assert resp.cost_metrics is None


# ---------------------------------------------------------------------------
# _sync_connection_string
# ---------------------------------------------------------------------------


class TestSyncConnectionString:
    """Verify async driver → sync driver mapping."""

    @pytest.mark.parametrize(
        "async_url, expected",
        [
            ("sqlite+aiosqlite:///test.db", "sqlite:///test.db"),
            ("postgresql+asyncpg://user:pass@host/db", "postgresql://user:pass@host/db"),
            ("mysql+aiomysql://user:pass@host/db", "mysql://user:pass@host/db"),
            ("sqlite:///already_sync.db", "sqlite:///already_sync.db"),
        ],
    )
    def test_maps_async_to_sync(self, async_url, expected):
        """Async SQLAlchemy drivers are replaced with their sync counterparts."""
        with patch(
            "inference_core.services.agents_service.get_settings"
        ) as mock_settings:
            mock_settings.return_value.database_url = async_url
            result = AgentService._sync_connection_string()
        assert result == expected


# ---------------------------------------------------------------------------
# _build_provider_user_context
# ---------------------------------------------------------------------------


class TestBuildProviderUserContext:
    """Verify context dict merging for tool providers."""

    def test_adds_user_id_session_request(self, agent_service):
        """Internal IDs are added when not already in user_context."""
        ctx = agent_service._build_provider_user_context(None)
        assert "user_id" in ctx
        assert "session_id" in ctx
        assert "request_id" in ctx

    def test_preserves_existing_keys(self, agent_service):
        """Pre-existing keys in user_context are not overwritten."""
        ctx = agent_service._build_provider_user_context(
            {"user_id": "override", "custom": "value"}
        )
        assert ctx["user_id"] == "override"
        assert ctx["custom"] == "value"

    def test_empty_when_no_ids(self):
        """Returns empty dict when no context and no IDs set."""
        svc = _make_agent_service(user_id=None, session_id=None, request_id=None)
        ctx = svc._build_provider_user_context(None)
        assert ctx == {}


# ---------------------------------------------------------------------------
# _build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """Verify system prompt composition with optional memory instructions."""

    def test_returns_base_prompt_when_no_memory(self, agent_service):
        """Without memory, base prompt is returned unchanged."""
        result = agent_service._build_system_prompt("You are helpful.")
        assert result == "You are helpful."

    def test_returns_none_when_no_prompt_no_memory(self, agent_service):
        """Without prompt or memory, returns None."""
        result = agent_service._build_system_prompt(None)
        assert result is None

    def test_appends_memory_instructions(self):
        """When memory is enabled, instructions are appended to base prompt."""
        svc = _make_agent_service()
        svc._memory_service = MagicMock()
        svc.use_memory = True

        with patch(
            "inference_core.services.agents_service.generate_memory_tools_system_instructions",
            return_value="[MEMORY INSTRUCTIONS]",
        ):
            result = svc._build_system_prompt("Base prompt.")

        assert "Base prompt." in result
        assert "[MEMORY INSTRUCTIONS]" in result

    def test_memory_instructions_only_when_no_base(self):
        """When memory enabled but no base prompt, returns only memory instructions."""
        svc = _make_agent_service()
        svc._memory_service = MagicMock()
        svc.use_memory = True

        with patch(
            "inference_core.services.agents_service.generate_memory_tools_system_instructions",
            return_value="[MEMORY INSTRUCTIONS]",
        ):
            result = svc._build_system_prompt(None)

        assert result == "[MEMORY INSTRUCTIONS]"


# ---------------------------------------------------------------------------
# _build_middleware
# ---------------------------------------------------------------------------


class TestBuildMiddleware:
    """Verify middleware list construction."""

    def test_adds_cost_tracking_when_enabled(self, agent_service):
        """CostTrackingMiddleware is auto-added when enable_cost_tracking=True."""
        from inference_core.agents.middleware import CostTrackingMiddleware

        with patch(
            "inference_core.services.agents_service.get_llm_config"
        ) as mock_cfg:
            mock_cfg.return_value.models = {}

            middleware = agent_service._build_middleware()

        has_cost = any(isinstance(m, CostTrackingMiddleware) for m in middleware)
        assert has_cost

    def test_no_cost_tracking_when_disabled(self):
        """CostTrackingMiddleware is NOT added when enable_cost_tracking=False."""
        from inference_core.agents.middleware import CostTrackingMiddleware

        svc = _make_agent_service(enable_cost_tracking=False)
        middleware = svc._build_middleware()

        has_cost = any(isinstance(m, CostTrackingMiddleware) for m in middleware)
        assert not has_cost

    def test_no_duplicate_cost_tracking(self, agent_service):
        """If CostTrackingMiddleware is pre-added, no duplicate is inserted."""
        from inference_core.agents.middleware import CostTrackingMiddleware

        existing_ct = CostTrackingMiddleware()
        agent_service._middleware = [existing_ct]

        with patch(
            "inference_core.services.agents_service.get_llm_config"
        ) as mock_cfg:
            mock_cfg.return_value.models = {}
            middleware = agent_service._build_middleware()

        ct_count = sum(1 for m in middleware if isinstance(m, CostTrackingMiddleware))
        assert ct_count == 1


# ---------------------------------------------------------------------------
# run_agent_steps
# ---------------------------------------------------------------------------


class TestRunAgentSteps:
    """Verify agent execution pipeline and response construction."""

    def test_returns_agent_response(self, agent_service):
        """run_agent_steps returns AgentResponse with result, steps, metadata."""
        # Mock agent.stream to yield a single chunk
        mock_agent = MagicMock()
        ai_msg = MagicMock()
        ai_msg.content = "Hello!"

        mock_agent.stream.return_value = [
            {"agent": {"messages": [ai_msg]}},
        ]
        agent_service.agent = mock_agent

        response = agent_service.run_agent_steps("Hi")

        assert isinstance(response, AgentResponse)
        assert response.metadata.model_name == "gpt-4o"
        assert len(response.steps) == 1
        assert response.result["messages"] == [ai_msg]

    def test_captures_cost_metrics(self, agent_service):
        """Cost metrics are extracted from accumulated state updates."""
        mock_agent = MagicMock()
        mock_agent.stream.return_value = [
            {
                "agent": {
                    "messages": [MagicMock(content="ok")],
                    "accumulated_input_tokens": 100,
                    "accumulated_output_tokens": 50,
                    "accumulated_total_tokens": 150,
                    "accumulated_extra_tokens": {},
                    "model_call_count": 1,
                    "tool_call_count": 0,
                    "usage_session_id": "u1",
                }
            },
        ]
        agent_service.agent = mock_agent

        response = agent_service.run_agent_steps("Hello")

        assert response.cost_metrics is not None
        assert response.cost_metrics.input_tokens == 100
        assert response.cost_metrics.output_tokens == 50
        assert response.cost_metrics.model_call_count == 1

    def test_no_cost_metrics_when_disabled(self):
        """cost_metrics is None when cost tracking is disabled."""
        svc = _make_agent_service(enable_cost_tracking=False)
        mock_agent = MagicMock()
        mock_agent.stream.return_value = [
            {"agent": {"messages": [MagicMock(content="ok")]}},
        ]
        svc.agent = mock_agent

        response = svc.run_agent_steps("Hello")
        assert response.cost_metrics is None

    def test_empty_stream_returns_empty_result(self, agent_service):
        """Empty stream yields empty result dict."""
        mock_agent = MagicMock()
        mock_agent.stream.return_value = []
        agent_service.agent = mock_agent

        response = agent_service.run_agent_steps("Hello")
        assert response.result == {}
        assert response.steps == []

    def test_interrupt_captured_in_result(self, agent_service):
        """__interrupt__ step data is captured in result."""
        mock_agent = MagicMock()
        interrupt_data = [MagicMock()]
        mock_agent.stream.return_value = [
            {"__interrupt__": interrupt_data},
        ]
        agent_service.agent = mock_agent

        response = agent_service.run_agent_steps("Hello")
        assert "__interrupt__" in response.result


# ---------------------------------------------------------------------------
# close / context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    """Verify resource cleanup via context manager."""

    def test_close_calls_exit_stack(self, agent_service):
        """close() properly cleans up the exit stack."""
        agent_service._exit_stack = MagicMock()
        agent_service.close()
        agent_service._exit_stack.close.assert_called_once()

    def test_context_manager_protocol(self, agent_service):
        """AgentService works with `with` statement."""
        agent_service._exit_stack = MagicMock()
        with agent_service as svc:
            assert svc is agent_service
        agent_service._exit_stack.close.assert_called_once()


# ---------------------------------------------------------------------------
# DeepAgentService._collect_spec_names
# ---------------------------------------------------------------------------


class TestDeepAgentServiceHelpers:
    """Verify DeepAgentService utility methods."""

    def test_collect_spec_names_from_dicts(self):
        """Extracts 'name' from SubAgent dicts."""
        specs = [
            {"name": "agent_a", "description": "A"},
            {"name": "agent_b", "description": "B"},
        ]
        names = DeepAgentService._collect_spec_names(specs)
        assert names == {"agent_a", "agent_b"}

    def test_collect_spec_names_from_objects(self):
        """Extracts 'name' from objects with name attribute."""
        obj1 = MagicMock()
        obj1.name = "obj_agent"
        specs = [obj1]
        names = DeepAgentService._collect_spec_names(specs)
        assert names == {"obj_agent"}

    def test_collect_spec_names_mixed(self):
        """Handles mix of dicts and objects."""
        obj = MagicMock()
        obj.name = "obj_agent"
        specs = [{"name": "dict_agent"}, obj]
        names = DeepAgentService._collect_spec_names(specs)
        assert names == {"dict_agent", "obj_agent"}

    def test_collect_spec_names_empty(self):
        """Empty list returns empty set."""
        assert DeepAgentService._collect_spec_names([]) == set()

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
    InstanceContext,
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
        memory_tools=None,
        memory_session_context_enabled=None,
        memory_tool_instructions_enabled=None,
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
        "instance_context": None,
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
            metadata=AgentMetadata(model_name="gpt-4o", tools_used=[], start_time=now),
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
            (
                "postgresql+asyncpg://user:pass@host/db",
                "postgresql://user:pass@host/db",
            ),
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

        with patch("inference_core.services.agents_service.get_llm_config") as mock_cfg:
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

        with patch("inference_core.services.agents_service.get_llm_config") as mock_cfg:
            mock_cfg.return_value.models = {}
            middleware = agent_service._build_middleware()

        ct_count = sum(1 for m in middleware if isinstance(m, CostTrackingMiddleware))
        assert ct_count == 1


# ---------------------------------------------------------------------------
# _build_stream_config — sync vs async callback selection
# ---------------------------------------------------------------------------


class TestBuildStreamConfigSyncMode:
    """Verify that _build_stream_config selects the right callback type."""

    def test_sync_true_uses_sync_callback(self, agent_service):
        from inference_core.services.stream_utils import SyncStreamCancelCallback

        _, cancel_cb, _ = agent_service._build_stream_config(
            {},
            on_token=None,
            on_custom=None,
            graceful_cancel=True,
            sync=True,
        )
        assert isinstance(cancel_cb, SyncStreamCancelCallback)

    def test_sync_false_uses_async_callback(self, agent_service):
        from inference_core.services.stream_utils import StreamCancelCallback

        _, cancel_cb, _ = agent_service._build_stream_config(
            {},
            on_token=None,
            on_custom=None,
            graceful_cancel=True,
            sync=False,
        )
        assert isinstance(cancel_cb, StreamCancelCallback)

    def test_default_uses_async_callback(self, agent_service):
        from inference_core.services.stream_utils import StreamCancelCallback

        _, cancel_cb, _ = agent_service._build_stream_config(
            {},
            on_token=None,
            on_custom=None,
            graceful_cancel=True,
        )
        assert isinstance(cancel_cb, StreamCancelCallback)

    def test_no_callback_when_graceful_cancel_false(self, agent_service):
        _, cancel_cb, _ = agent_service._build_stream_config(
            {},
            on_token=None,
            on_custom=None,
            graceful_cancel=False,
            sync=True,
        )
        assert cancel_cb is None


# ---------------------------------------------------------------------------
# run_agent_steps
# ---------------------------------------------------------------------------


class TestRunAgentSteps:
    """Verify agent execution pipeline and response construction."""

    def test_returns_agent_response(self, agent_service):
        """run_agent_steps returns AgentResponse with result, steps, metadata."""
        # Mock agent.stream to yield a single v2 StreamPart chunk
        mock_agent = MagicMock()
        ai_msg = MagicMock()
        ai_msg.content = "Hello!"

        mock_agent.stream.return_value = [
            {"type": "updates", "data": {"agent": {"messages": [ai_msg]}}, "ns": []},
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
                "type": "updates",
                "ns": [],
                "data": {
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
            {
                "type": "updates",
                "data": {"agent": {"messages": [MagicMock(content="ok")]}},
                "ns": [],
            },
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
        """__interrupt__ step data is captured in result messages."""
        mock_agent = MagicMock()
        interrupt_data = [MagicMock()]
        mock_agent.stream.return_value = [
            {"type": "updates", "data": {"__interrupt__": interrupt_data}, "ns": []},
        ]
        agent_service.agent = mock_agent

        response = agent_service.run_agent_steps("Hello")
        assert "messages" in response.result
        assert response.result["messages"][0] == {"__interrupt__": interrupt_data}


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


# ---------------------------------------------------------------------------
# InstanceContext
# ---------------------------------------------------------------------------


class TestInstanceContext:
    """Verify InstanceContext dataclass and display_name property."""

    def test_frozen_dataclass(self):
        """InstanceContext fields are immutable."""
        ctx = InstanceContext(
            instance_id=uuid.uuid4(),
            instance_name="my-writer",
            base_agent_name="assistant_agent",
        )
        with pytest.raises(AttributeError):
            ctx.instance_name = "other"

    def test_display_name_with_instance(self):
        """display_name returns instance_name when instance_context is set."""
        ctx = InstanceContext(
            instance_id=uuid.uuid4(),
            instance_name="creative-writer",
            base_agent_name="assistant_agent",
        )
        svc = _make_agent_service(instance_context=ctx)
        assert svc.display_name == "creative-writer"
        assert svc.agent_name == "test_agent"

    def test_display_name_without_instance(self, agent_service):
        """display_name falls back to agent_name when no instance_context."""
        assert agent_service.display_name == "test_agent"
        assert agent_service.instance_context is None

    def test_instance_context_in_provider_context(self):
        """Instance fields are included in _build_provider_user_context."""
        ctx = InstanceContext(
            instance_id=uuid.uuid4(),
            instance_name="creative-writer",
            base_agent_name="assistant_agent",
        )
        svc = _make_agent_service(instance_context=ctx)
        provider_ctx = svc._build_provider_user_context(None)
        assert provider_ctx["instance_name"] == "creative-writer"
        assert provider_ctx["instance_id"] == str(ctx.instance_id)

    def test_instance_context_not_in_provider_context_when_absent(self, agent_service):
        """Instance fields are absent from provider context when no instance_context."""
        ctx = agent_service._build_provider_user_context(None)
        assert "instance_name" not in ctx
        assert "instance_id" not in ctx


# ---------------------------------------------------------------------------
# build_config_for_instance
# ---------------------------------------------------------------------------


class TestBuildConfigForInstance:
    """Verify DB →  LLMConfig override translation."""

    def test_uses_provided_base_config(self):
        """Custom base_config is used instead of raw YAML."""
        mock_base = MagicMock()
        mock_base.agent_models = {"test_agent": "gpt-4o"}
        mock_base.with_overrides.return_value = mock_base

        result = AgentService.build_config_for_instance(
            {"base_agent_name": "test_agent", "primary_model": "gpt-5"},
            base_config=mock_base,
        )
        mock_base.with_overrides.assert_called_once()
        assert result is mock_base

    def test_falls_back_to_yaml_when_no_base(self):
        """When no base_config, raw YAML config is used."""
        with patch("inference_core.services.agents_service.get_llm_config") as mock_get:
            mock_cfg = MagicMock()
            mock_cfg.agent_models = {"test_agent": "gpt-4o"}
            mock_cfg.with_overrides.return_value = mock_cfg
            mock_get.return_value = mock_cfg

            AgentService.build_config_for_instance(
                {"base_agent_name": "test_agent", "primary_model": "gpt-5"},
            )
            mock_get.assert_called_once()

    def test_accepts_orm_object(self):
        """ORM objects with to_dict() are accepted and converted."""
        mock_orm = MagicMock()
        mock_orm.to_dict.return_value = {
            "base_agent_name": "test_agent",
            "primary_model": "gpt-5",
        }
        mock_base = MagicMock()
        mock_base.agent_models = {"test_agent": "gpt-4o"}
        mock_base.with_overrides.return_value = mock_base

        AgentService.build_config_for_instance(mock_orm, base_config=mock_base)
        mock_orm.to_dict.assert_called_once()
        mock_base.with_overrides.assert_called_once()

    def test_no_overrides_when_no_changes(self):
        """Empty instance config returns base config unchanged without calling with_overrides."""
        mock_base = MagicMock()
        mock_base.agent_models = {"test_agent": "gpt-4o"}

        result = AgentService.build_config_for_instance(
            {"base_agent_name": "test_agent"},
            base_config=mock_base,
        )
        # No primary_model, no config_overrides → base config returned as-is
        mock_base.with_overrides.assert_not_called()
        assert result is mock_base


# ---------------------------------------------------------------------------
# CostTrackingMiddleware instance context
# ---------------------------------------------------------------------------


class TestCostTrackingInstanceContext:
    """Verify CostTrackingMiddleware carries instance fields."""

    def test_middleware_stores_instance_fields(self):
        from inference_core.agents.middleware import CostTrackingMiddleware

        iid = uuid.uuid4()
        m = CostTrackingMiddleware(instance_id=iid, instance_name="my-writer")
        assert m.instance_id == iid
        assert m.instance_name == "my-writer"

    def test_middleware_none_when_no_instance(self):
        from inference_core.agents.middleware import CostTrackingMiddleware

        m = CostTrackingMiddleware()
        assert m.instance_id is None
        assert m.instance_name is None

    def test_cost_tracking_gets_instance_from_agent_service(self):
        """_build_middleware passes instance context to CostTrackingMiddleware."""
        from inference_core.agents.middleware import CostTrackingMiddleware

        ctx = InstanceContext(
            instance_id=uuid.uuid4(),
            instance_name="creative-writer",
            base_agent_name="assistant_agent",
        )
        svc = _make_agent_service(instance_context=ctx)

        with patch("inference_core.services.agents_service.get_llm_config") as mock_cfg:
            mock_cfg.return_value.models = {}
            middleware = svc._build_middleware()

        ct = [m for m in middleware if isinstance(m, CostTrackingMiddleware)]
        assert len(ct) == 1
        assert ct[0].instance_id == ctx.instance_id
        assert ct[0].instance_name == "creative-writer"


# ---------------------------------------------------------------------------
# AgentService.from_user_instance
# ---------------------------------------------------------------------------


def _make_mock_instance(**overrides):
    """Build a minimal UserAgentInstance-like mock."""
    instance = MagicMock()
    instance.id = uuid.uuid4()
    instance.base_agent_name = "test_agent"
    instance.instance_name = "my-instance"
    instance.primary_model = None
    instance.system_prompt_override = None
    instance.system_prompt_append = None
    instance.description = None
    instance.config_overrides = {}
    for k, v in overrides.items():
        setattr(instance, k, v)
    return instance


def _make_from_user_instance_patches():
    """Context-manager building all patches needed for from_user_instance tests.

    WHY: from_user_instance passes config=resolved_config so __init__ calls
    LLMModelFactory(config) instead of get_model_factory() — both need mocking.
    """
    import contextlib

    mock_factory = MagicMock()
    mock_factory.get_agent_model_name.return_value = "gpt-4o"
    mock_factory.get_model_for_agent.return_value = MagicMock()
    mock_factory.config.get_specific_agent_config.return_value = MagicMock(
        local_tool_providers=[],
        mcp_profile=None,
        allowed_tools=None,
        tool_model_overrides=None,
        description="Test",
        skills=None,
        subagents=None,
        interrupt_on=None,
    )

    mock_base = MagicMock()
    mock_base.agent_models = {"test_agent": "gpt-4o"}
    # No overrides means build_config_for_instance returns config unchanged
    # (no with_overrides call), so we don't need to configure it.

    @contextlib.contextmanager
    def patches():
        with patch(
            "inference_core.services.agents_service.get_model_factory",
            return_value=mock_factory,
        ), patch(
            "inference_core.services.agents_service.LLMModelFactory",
            return_value=mock_factory,
        ), patch(
            "inference_core.services.agents_service.get_llm_config",
            return_value=mock_base,
        ), patch(
            "inference_core.services.agents_service.get_settings",
            return_value=MagicMock(
                database_url="sqlite+aiosqlite:///test.db",
                agent_memory_enabled=False,
            ),
        ):
            yield mock_base

    return patches


class TestAgentServiceFromUserInstance:
    """Verify AgentService.from_user_instance factory."""

    def test_returns_agent_service(self):
        """from_user_instance returns an AgentService instance."""
        patches = _make_from_user_instance_patches()
        instance = _make_mock_instance()
        with patches() as mock_base:
            svc = AgentService.from_user_instance(
                instance, user_id=uuid.uuid4(), base_config=mock_base
            )
        assert isinstance(svc, AgentService)

    def test_sets_instance_context(self):
        """from_user_instance wires InstanceContext from ORM fields."""
        patches = _make_from_user_instance_patches()
        iid = uuid.uuid4()
        instance = _make_mock_instance()
        instance.id = iid
        with patches() as mock_base:
            svc = AgentService.from_user_instance(instance, base_config=mock_base)
        assert svc.instance_context is not None
        assert svc.instance_context.instance_id == iid
        assert svc.instance_context.instance_name == "my-instance"
        assert svc.instance_context.base_agent_name == "test_agent"

    def test_stores_prompt_override(self):
        """system_prompt_override is stored for use in create_agent."""
        patches = _make_from_user_instance_patches()
        instance = _make_mock_instance(system_prompt_override="Custom prompt")
        with patches() as mock_base:
            svc = AgentService.from_user_instance(instance, base_config=mock_base)
        assert svc._system_prompt_override == "Custom prompt"
        assert svc._system_prompt_append is None

    def test_stores_prompt_append(self):
        """system_prompt_append is stored for use in create_agent."""
        patches = _make_from_user_instance_patches()
        instance = _make_mock_instance(system_prompt_append="Extra instructions")
        with patches() as mock_base:
            svc = AgentService.from_user_instance(instance, base_config=mock_base)
        assert svc._system_prompt_override is None
        assert svc._system_prompt_append == "Extra instructions"


# ---------------------------------------------------------------------------
# _apply_prompt_overrides
# ---------------------------------------------------------------------------


class TestApplyPromptOverrides:
    """Verify prompt override precedence logic in AgentService."""

    def test_override_replaces_base(self, agent_service):
        """system_prompt_override fully replaces the base prompt."""
        agent_service._system_prompt_override = "OVERRIDE"
        agent_service._system_prompt_append = None
        result = agent_service._apply_prompt_overrides("original")
        assert result == "OVERRIDE"

    def test_append_concatenates(self, agent_service):
        """system_prompt_append is appended to the base prompt."""
        agent_service._system_prompt_override = None
        agent_service._system_prompt_append = "EXTRA"
        result = agent_service._apply_prompt_overrides("base")
        assert result == "base\n\nEXTRA"

    def test_append_alone_when_no_base(self, agent_service):
        """system_prompt_append is used alone when base prompt is None."""
        agent_service._system_prompt_override = None
        agent_service._system_prompt_append = "EXTRA"
        result = agent_service._apply_prompt_overrides(None)
        assert result == "EXTRA"

    def test_passthrough_when_no_overrides(self, agent_service):
        """Base prompt is returned unchanged when no overrides are set."""
        agent_service._system_prompt_override = None
        agent_service._system_prompt_append = None
        result = agent_service._apply_prompt_overrides("base")
        assert result == "base"

    def test_none_passthrough(self, agent_service):
        """None base prompt is returned as None when no overrides are set."""
        agent_service._system_prompt_override = None
        agent_service._system_prompt_append = None
        result = agent_service._apply_prompt_overrides(None)
        assert result is None

    def test_override_takes_precedence_over_append(self, agent_service):
        """system_prompt_override wins over system_prompt_append."""
        agent_service._system_prompt_override = "OVERRIDE"
        agent_service._system_prompt_append = "EXTRA"
        result = agent_service._apply_prompt_overrides("base")
        assert result == "OVERRIDE"


# ---------------------------------------------------------------------------
# Memory surface resolution helpers
# ---------------------------------------------------------------------------


class TestResolveMemoryTools:
    """Verify _resolve_memory_tools precedence: runtime → AgentConfig → None."""

    def test_returns_none_by_default(self):
        """No override, no AgentConfig → None (all tools)."""
        svc = _make_agent_service()
        assert svc._resolve_memory_tools() is None

    def test_runtime_override_wins(self):
        """Runtime override takes precedence over AgentConfig."""
        svc = _make_agent_service(
            memory_tools=["save_memory_store"],
        )
        svc.agent_config = MagicMock(
            memory_tools=["recall_memories_store", "delete_memory_store"],
        )
        assert svc._resolve_memory_tools() == ["save_memory_store"]

    def test_agent_config_fallback(self):
        """AgentConfig is used when no runtime override."""
        svc = _make_agent_service()
        svc.agent_config = MagicMock(
            memory_tools=["save_memory_store", "recall_memories_store"],
        )
        assert svc._resolve_memory_tools() == [
            "save_memory_store",
            "recall_memories_store",
        ]

    def test_empty_list_runtime(self):
        """Runtime override with empty list disables all tools."""
        svc = _make_agent_service(memory_tools=[])
        assert svc._resolve_memory_tools() == []

    def test_agent_config_none_returns_none(self):
        """AgentConfig.memory_tools=None → returns None."""
        svc = _make_agent_service()
        svc.agent_config = MagicMock(memory_tools=None)
        assert svc._resolve_memory_tools() is None


class TestResolveMemorySessionContextEnabled:
    """Verify _resolve_memory_session_context_enabled precedence."""

    def test_defaults_to_settings_auto_recall(self):
        """Falls back to global agent_memory_auto_recall when no overrides."""
        svc = _make_agent_service()
        with patch(
            "inference_core.services.agents_service.get_settings",
        ) as mock_settings:
            mock_settings.return_value.agent_memory_auto_recall = True
            assert svc._resolve_memory_session_context_enabled() is True

    def test_runtime_override_wins(self):
        """Runtime override takes precedence."""
        svc = _make_agent_service(memory_session_context_enabled=False)
        svc.agent_config = MagicMock(memory_session_context_enabled=True)
        assert svc._resolve_memory_session_context_enabled() is False

    def test_agent_config_fallback(self):
        """AgentConfig is used when no runtime override."""
        svc = _make_agent_service()
        svc.agent_config = MagicMock(memory_session_context_enabled=False)
        assert svc._resolve_memory_session_context_enabled() is False


class TestResolveMemoryToolInstructionsEnabled:
    """Verify _resolve_memory_tool_instructions_enabled precedence."""

    def test_default_true_when_all_tools(self):
        """Default is True when memory_tools is None (all tools active)."""
        svc = _make_agent_service()
        assert svc._resolve_memory_tool_instructions_enabled() is True

    def test_default_false_when_tools_empty(self):
        """Default is False when memory_tools=[] (no tools active)."""
        svc = _make_agent_service(memory_tools=[])
        assert svc._resolve_memory_tool_instructions_enabled() is False

    def test_runtime_override_wins(self):
        """Runtime override takes precedence."""
        svc = _make_agent_service(memory_tool_instructions_enabled=False)
        assert svc._resolve_memory_tool_instructions_enabled() is False

    def test_agent_config_fallback(self):
        """AgentConfig is used when no runtime override."""
        svc = _make_agent_service()
        svc.agent_config = MagicMock(memory_tool_instructions_enabled=False)
        assert svc._resolve_memory_tool_instructions_enabled() is False


class TestBuildSystemPromptMemoryConfig:
    """Verify _build_system_prompt respects memory_tool_instructions_enabled."""

    def test_instructions_omitted_when_disabled(self):
        """No memory instructions when tool_instructions_enabled=False."""
        svc = _make_agent_service(memory_tool_instructions_enabled=False)
        svc._memory_service = MagicMock()
        svc.use_memory = True
        result = svc._build_system_prompt("Base prompt.")
        assert result == "Base prompt."

    def test_instructions_included_when_enabled(self):
        """Memory instructions appended when tool_instructions_enabled=True."""
        svc = _make_agent_service(memory_tool_instructions_enabled=True)
        svc._memory_service = MagicMock()
        svc.use_memory = True

        with patch(
            "inference_core.services.agents_service.generate_memory_tools_system_instructions",
            return_value="[MEMORY INSTRUCTIONS]",
        ):
            result = svc._build_system_prompt("Base prompt.")

        assert "[MEMORY INSTRUCTIONS]" in result

    def test_instructions_pass_resolved_tools(self):
        """generate_memory_tools_system_instructions receives resolved tool names."""
        svc = _make_agent_service(
            memory_tools=["save_memory_store"],
            memory_tool_instructions_enabled=True,
        )
        svc._memory_service = MagicMock()
        svc.use_memory = True

        with patch(
            "inference_core.services.agents_service.generate_memory_tools_system_instructions",
            return_value="[INSTRUCTIONS]",
        ) as mock_gen:
            svc._build_system_prompt("Base.")
            mock_gen.assert_called_once_with(
                active_tool_names=["save_memory_store"],
            )


class TestBuildMiddlewareMemoryConfig:
    """Verify _build_middleware respects memory_session_context_enabled."""

    def test_memory_middleware_auto_recall_follows_resolution(self):
        """MemoryMiddleware auto_recall uses _resolve_memory_session_context_enabled."""
        from inference_core.agents.middleware.memory import MemoryMiddleware

        svc = _make_agent_service(
            memory_session_context_enabled=False,
        )
        svc._memory_service = MagicMock()
        svc._user_id = uuid.uuid4()
        svc.use_memory = True

        with patch(
            "inference_core.services.agents_service.get_settings",
        ) as mock_settings, patch(
            "inference_core.services.agents_service.get_llm_config",
        ) as mock_cfg:
            mock_settings.return_value.agent_memory_auto_recall = True
            mock_settings.return_value.agent_memory_max_results = 5
            mock_settings.return_value.agent_memory_postrun_analysis_enabled = True
            mock_settings.return_value.agent_memory_postrun_analysis_model = None
            mock_cfg.return_value.models = {}
            middleware = svc._build_middleware()

        mem_mws = [m for m in middleware if isinstance(m, MemoryMiddleware)]
        assert len(mem_mws) == 1
        assert mem_mws[0].auto_recall is False


class TestBuildRemoteMetadataMemoryOverrides:
    """Verify _build_remote_metadata forwards memory overrides."""

    def test_no_memory_keys_when_no_overrides(self):
        """No memory_session_context_enabled key when no runtime override."""
        svc = _make_agent_service()
        metadata = svc._build_remote_metadata()
        assert "memory_session_context_enabled" not in metadata
        assert "memory_tool_instructions_enabled" not in metadata

    def test_forwards_session_context_override(self):
        """Forwards memory_session_context_enabled when runtime override set."""
        svc = _make_agent_service(memory_session_context_enabled=False)
        metadata = svc._build_remote_metadata()
        assert "memory_session_context_enabled" in metadata
        assert metadata["memory_session_context_enabled"] is False

    def test_forwards_tool_instructions_override(self):
        """Forwards memory_tool_instructions_enabled when runtime override set."""
        svc = _make_agent_service(memory_tool_instructions_enabled=False)
        metadata = svc._build_remote_metadata()
        assert "memory_tool_instructions_enabled" in metadata
        assert metadata["memory_tool_instructions_enabled"] is False

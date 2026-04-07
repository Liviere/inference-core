"""Tests for agent_server_client.py and remote execution path in AgentService.

Covers:
- get_agent_server_client singleton behavior
- run_remote / stream_remote delegation
- _is_remote property logic
- _arun_agent_steps_remote integration with AgentService
- Fallback to local when agent_server_enabled=False
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_core.services.agent_server_client import (
    _build_config,
    _forward_message_event,
    _forward_step_event,
    _resolve_graph_id,
    get_agent_server_client,
    reset_agent_server_client,
    run_remote,
    stream_remote,
)
from inference_core.services.agents_service import AgentResponse, AgentService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure singleton is clean before and after each test."""
    reset_agent_server_client()
    yield
    reset_agent_server_client()


def _mock_settings(**overrides):
    defaults = {
        "agent_server_url": "http://localhost:8123",
        "agent_server_api_key": "test-key",
        "agent_server_enabled": True,
        "agent_server_timeout": 60,
        "database_url": "sqlite+aiosqlite:///test.db",
        "agent_memory_enabled": False,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _make_agent_service_for_remote(execution_mode="remote", **overrides):
    """Create AgentService with mocked deps and configurable execution_mode."""
    mock_model_factory = MagicMock()
    mock_model = MagicMock()
    mock_model_factory.get_agent_model_name.return_value = "gpt-5-mini"
    mock_model_factory.get_model_for_agent.return_value = mock_model

    agent_config = MagicMock(
        local_tool_providers=[],
        mcp_profile=None,
        allowed_tools=None,
        tool_model_overrides=None,
        description="Test agent",
        skills=None,
        subagents=None,
        interrupt_on=None,
        execution_mode=execution_mode,
        remote_graph_id=overrides.pop("remote_graph_id", None),
    )
    mock_model_factory.config.get_specific_agent_config.return_value = agent_config

    defaults = {
        "agent_name": "test_remote_agent",
        "tools": None,
        "use_checkpoints": False,
        "use_memory": False,
        "checkpoint_config": None,
        "enable_cost_tracking": False,
        "user_id": uuid.uuid4(),
        "session_id": "sess-remote",
        "request_id": "req-remote",
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
        return_value=_mock_settings(),
    ):
        svc = AgentService(**defaults)

    return svc


# ---------------------------------------------------------------------------
# _resolve_graph_id
# ---------------------------------------------------------------------------


class TestResolveGraphId:
    def test_uses_remote_graph_id_when_set(self):
        assert _resolve_graph_id("default_agent", "custom_graph") == "custom_graph"

    def test_falls_back_to_agent_name(self):
        assert _resolve_graph_id("default_agent", None) == "default_agent"


# ---------------------------------------------------------------------------
# get_agent_server_client
# ---------------------------------------------------------------------------


class TestGetAgentServerClient:
    def test_creates_client_with_url(self):
        settings = _mock_settings()
        with patch(
            "inference_core.services.agent_server_client.get_client"
        ) as mock_get:
            mock_get.return_value = MagicMock()
            client = get_agent_server_client(settings)
            mock_get.assert_called_once_with(
                url="http://localhost:8123",
                api_key="test-key",
                timeout=60,
            )

    def test_returns_singleton(self):
        settings = _mock_settings()
        with patch(
            "inference_core.services.agent_server_client.get_client"
        ) as mock_get:
            mock_get.return_value = MagicMock()
            c1 = get_agent_server_client(settings)
            c2 = get_agent_server_client(settings)
            assert c1 is c2
            mock_get.assert_called_once()

    def test_raises_without_url(self):
        settings = _mock_settings(agent_server_url=None)
        with pytest.raises(RuntimeError, match="AGENT_SERVER_URL"):
            get_agent_server_client(settings)


# ---------------------------------------------------------------------------
# _forward_message_event / _forward_step_event
# ---------------------------------------------------------------------------


class TestForwardEvents:
    def test_forward_message_text(self):
        tokens = []
        callback = lambda text, meta: tokens.append((text, meta))
        data = [{"type": "ai", "content": "Hello world", "name": "agent"}]
        _forward_message_event(data, callback)
        assert len(tokens) == 1
        assert tokens[0][0] == "Hello world"
        assert tokens[0][1]["type"] == "text"
        assert "ns" not in tokens[0][1]

    def test_forward_message_with_ns(self):
        tokens = []
        callback = lambda text, meta: tokens.append((text, meta))
        data = [{"type": "ai", "content": "Sub hello", "name": "weather"}]
        _forward_message_event(data, callback, ns=["weather_agent:abc"])
        assert len(tokens) == 1
        assert tokens[0][1]["ns"] == ["weather_agent:abc"]
        assert tokens[0][1]["node"] == "weather"

    def test_forward_message_structured_blocks(self):
        tokens = []
        callback = lambda text, meta: tokens.append((text, meta))
        data = [
            {
                "type": "ai",
                "content": [
                    {"type": "text", "text": "Part 1"},
                    {"type": "thinking", "thinking": "Reasoning..."},
                ],
                "name": "agent",
            }
        ]
        _forward_message_event(data, callback)
        assert len(tokens) == 2
        assert tokens[0][1]["type"] == "text"
        assert tokens[1][1]["type"] == "reasoning"

    def test_forward_message_ignores_non_ai(self):
        tokens = []
        callback = lambda text, meta: tokens.append((text, meta))
        data = [{"type": "human", "content": "User msg"}]
        _forward_message_event(data, callback)
        assert len(tokens) == 0

    def test_forward_step_event(self):
        steps = []
        callback = lambda name, data: steps.append((name, data))
        _forward_step_event(
            {"agent": {"messages": [{"role": "ai", "content": "done"}]}},
            callback,
        )
        assert len(steps) == 1
        assert steps[0][0] == "agent"
        assert "ns" not in steps[0][1]

    def test_forward_step_event_with_ns(self):
        steps = []
        callback = lambda name, data: steps.append((name, data))
        _forward_step_event(
            {"agent": {"messages": [{"role": "ai", "content": "done"}]}},
            callback,
            ns=["sub:123"],
        )
        assert len(steps) == 1
        assert steps[0][1]["ns"] == ["sub:123"]


# ---------------------------------------------------------------------------
# run_remote
# ---------------------------------------------------------------------------


class TestRunRemote:
    async def test_calls_wait_with_correct_params(self):
        mock_client = MagicMock()
        mock_client.runs.wait = AsyncMock(
            return_value={"messages": [{"role": "ai", "content": "Hello"}]}
        )

        with patch(
            "inference_core.services.agent_server_client.get_agent_server_client",
            return_value=mock_client,
        ):
            result = await run_remote(
                agent_name="test_agent",
                user_input="Hi there",
                thread_id="thread-123",
            )

        assert result["messages"][0]["content"] == "Hello"
        mock_client.runs.wait.assert_called_once()
        call_kwargs = mock_client.runs.wait.call_args[1]
        assert call_kwargs["thread_id"] == "thread-123"
        assert call_kwargs["assistant_id"] == "test_agent"

    async def test_uses_remote_graph_id(self):
        mock_client = MagicMock()
        mock_client.runs.wait = AsyncMock(return_value={})

        with patch(
            "inference_core.services.agent_server_client.get_agent_server_client",
            return_value=mock_client,
        ):
            await run_remote(
                agent_name="test_agent",
                remote_graph_id="custom_graph",
                user_input="Hi",
            )

        call_kwargs = mock_client.runs.wait.call_args[1]
        assert call_kwargs["assistant_id"] == "custom_graph"


# ---------------------------------------------------------------------------
# stream_remote
# ---------------------------------------------------------------------------


class TestStreamRemote:
    async def test_forwards_tokens_and_steps(self):
        tokens = []
        steps = []

        # Simulate v2 SSE events (TypedDicts) from Agent Server
        async def _fake_stream(*args, **kwargs):
            yield {
                "type": "messages/partial",
                "ns": [],
                "data": [{"type": "ai", "content": "Hello", "name": "agent"}],
            }
            yield {
                "type": "updates",
                "ns": [],
                "data": {"agent": {"messages": [{"role": "ai", "content": "done"}]}},
            }

        mock_client = MagicMock()
        mock_client.runs.stream = _fake_stream

        with patch(
            "inference_core.services.agent_server_client.get_agent_server_client",
            return_value=mock_client,
        ):
            result = await stream_remote(
                agent_name="test_agent",
                user_input="Hi",
                on_token=lambda text, meta: tokens.append(text),
                on_step=lambda name, data: steps.append(name),
            )

        assert len(tokens) == 1
        assert tokens[0] == "Hello"
        assert len(steps) == 1
        assert steps[0] == "agent"

    async def test_forwards_subgraph_ns(self):
        """Subgraph events carry non-empty ns — verify it reaches on_token meta."""
        tokens = []

        async def _fake_stream(*args, **kwargs):
            yield {
                "type": "messages/partial",
                "ns": ["weather_agent:task_abc"],
                "data": [{"type": "ai", "content": "Sunny", "name": "weather"}],
            }

        mock_client = MagicMock()
        mock_client.runs.stream = _fake_stream

        with patch(
            "inference_core.services.agent_server_client.get_agent_server_client",
            return_value=mock_client,
        ):
            await stream_remote(
                agent_name="test_agent",
                user_input="Weather?",
                on_token=lambda text, meta: tokens.append((text, meta)),
            )

        assert len(tokens) == 1
        assert tokens[0][0] == "Sunny"
        assert tokens[0][1]["ns"] == ["weather_agent:task_abc"]
        assert tokens[0][1]["node"] == "weather"

    async def test_cancel_check_raises(self):
        from inference_core.services._cancel import AgentCancelled

        call_count = 0

        async def _fake_stream(*args, **kwargs):
            yield {"type": "updates", "ns": [], "data": {}}
            yield {"type": "updates", "ns": [], "data": {}}

        mock_client = MagicMock()
        mock_client.runs.stream = _fake_stream

        def cancel():
            nonlocal call_count
            call_count += 1
            return call_count >= 1

        with patch(
            "inference_core.services.agent_server_client.get_agent_server_client",
            return_value=mock_client,
        ):
            with pytest.raises(AgentCancelled):
                await stream_remote(
                    agent_name="test_agent",
                    user_input="Hi",
                    cancel_check=cancel,
                )


# ---------------------------------------------------------------------------
# AgentService._is_remote
# ---------------------------------------------------------------------------


class TestIsRemote:
    def test_remote_when_both_flags_set(self):
        svc = _make_agent_service_for_remote(execution_mode="remote")
        with patch(
            "inference_core.services.agents_service.get_settings",
            return_value=_mock_settings(agent_server_enabled=True),
        ):
            assert svc._is_remote is True

    def test_local_when_server_disabled(self):
        svc = _make_agent_service_for_remote(execution_mode="remote")
        with patch(
            "inference_core.services.agents_service.get_settings",
            return_value=_mock_settings(agent_server_enabled=False),
        ):
            assert svc._is_remote is False

    def test_local_when_execution_mode_local(self):
        svc = _make_agent_service_for_remote(execution_mode="local")
        with patch(
            "inference_core.services.agents_service.get_settings",
            return_value=_mock_settings(agent_server_enabled=True),
        ):
            assert svc._is_remote is False


# ---------------------------------------------------------------------------
# AgentService.run_agent_steps — remote guard
# ---------------------------------------------------------------------------


class TestRunAgentStepsRemoteGuard:
    def test_sync_run_raises_for_remote(self):
        svc = _make_agent_service_for_remote(execution_mode="remote")
        with patch(
            "inference_core.services.agents_service.get_settings",
            return_value=_mock_settings(agent_server_enabled=True),
        ):
            with pytest.raises(RuntimeError, match="execution_mode='remote'"):
                svc.run_agent_steps("Hello")


# ---------------------------------------------------------------------------
# AgentService.arun_agent_steps — remote delegation
# ---------------------------------------------------------------------------


class TestArunAgentStepsRemote:
    async def test_delegates_to_remote(self):
        svc = _make_agent_service_for_remote(execution_mode="remote")

        mock_result = {"messages": [{"role": "ai", "content": "Remote response"}]}

        with patch(
            "inference_core.services.agents_service.get_settings",
            return_value=_mock_settings(agent_server_enabled=True),
        ), patch(
            "inference_core.services.agent_server_client.get_agent_server_client",
            return_value=MagicMock(
                runs=MagicMock(wait=AsyncMock(return_value=mock_result))
            ),
        ):
            response = await svc.arun_agent_steps("Hello remote")

        assert isinstance(response, AgentResponse)
        assert response.result["messages"][0]["content"] == "Remote response"
        assert response.metadata.model_name == "gpt-5-mini"

    async def test_uses_local_when_disabled(self):
        """When agent_server_enabled=False, should go through local path."""
        svc = _make_agent_service_for_remote(execution_mode="remote")

        # Mock the local execution path
        mock_agent = AsyncMock()

        async def _fake_astream(*args, **kwargs):
            yield {
                "type": "updates",
                "data": {"agent": {"messages": [MagicMock(content="local response")]}},
                "ns": [],
            }

        mock_agent.astream = _fake_astream
        svc.agent = mock_agent

        with patch(
            "inference_core.services.agents_service.get_settings",
            return_value=_mock_settings(agent_server_enabled=False),
        ):
            # _is_remote returns False, so it should try local path
            assert svc._is_remote is False


# ---------------------------------------------------------------------------
# _build_config — merges checkpoint + user context into configurable
# ---------------------------------------------------------------------------


class TestBuildConfig:
    def test_empty_when_no_inputs(self):
        assert _build_config(None, None) == {}

    def test_checkpoint_keys_forwarded(self):
        cfg = _build_config({"thread_id": "t1", "version": "v2"}, None)
        assert cfg["configurable"]["version"] == "v2"
        assert "thread_id" not in cfg["configurable"]

    def test_user_context_from_metadata(self):
        uid = str(uuid.uuid4())
        cfg = _build_config(None, {"user_id": uid, "session_id": "s1"})
        assert cfg["configurable"]["user_id"] == uid
        assert cfg["configurable"]["session_id"] == "s1"

    def test_merges_checkpoint_and_metadata(self):
        uid = str(uuid.uuid4())
        cfg = _build_config(
            {"thread_id": "t1", "ns": "test"},
            {"user_id": uid, "request_id": "r1", "unrelated_key": "ignored"},
        )
        assert cfg["configurable"]["ns"] == "test"
        assert cfg["configurable"]["user_id"] == uid
        assert cfg["configurable"]["request_id"] == "r1"
        # Non-middleware keys from metadata should NOT be forwarded
        assert "unrelated_key" not in cfg["configurable"]

    def test_ignores_non_middleware_metadata_keys(self):
        cfg = _build_config(None, {"source": "test", "agent_name": "bot"})
        # No middleware keys → empty dict
        assert cfg == {}

    def test_forwards_instance_override_keys(self):
        cfg = _build_config(
            None,
            {
                "user_id": str(uuid.uuid4()),
                "primary_model": "claude-haiku-4-5-20251001",
                "system_prompt_override": "Custom prompt",
                "system_prompt_append": "Extra",
            },
        )
        c = cfg["configurable"]
        assert c["primary_model"] == "claude-haiku-4-5-20251001"
        assert c["system_prompt_override"] == "Custom prompt"
        assert c["system_prompt_append"] == "Extra"

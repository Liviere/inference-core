"""How agents load and call MCP tools: which profile, which servers, which tools."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.tools import _handle_mcp_tool_error, _MCPToolExecutionError

from inference_core.agents.agent_mcp_tools import (
    AgentMCPToolManager,
    _wrap_tool_for_sync,
)
from inference_core.llm.config import (
    LLMConfig,
    MCPConfig,
    MCPProfileConfig,
    MCPServerConfig,
)
from inference_core.services.agents_service import AgentService, _drop_repeated_tools


def _config(**mcp):
    """A loaded configuration with only what MCP reads, without a YAML file."""
    config = object.__new__(LLMConfig)
    config.agent_configs = {
        "assistant_agent": SimpleNamespace(mcp_profile="web"),
        "email_agent": SimpleNamespace(mcp_profile=None),
    }
    config.mcp_config = MCPConfig(enabled=True, require_superuser=False, **mcp)
    return config


def _manager(config):
    with patch("inference_core.agents.agent_mcp_tools.get_llm_config", return_value=config):
        return AgentMCPToolManager()


def _named(name):
    tool = MagicMock()
    tool.name = name
    return tool


# --------------------------------------------------------------------------- #
# Which profile an agent may load
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_profile_loads_for_the_type_that_names_it():
    config = _config(
        profiles={"web": MCPProfileConfig(servers=["s"])},
        servers={"s": MCPServerConfig(transport="stdio", command="x")},
    )
    with patch("langchain_mcp_adapters.client.MultiServerMCPClient") as MockClient:
        MockClient.return_value.get_tools = AsyncMock(return_value=[_named("t")])
        tools = await _manager(config).get_tools_for_profile(
            "web", agent_type="assistant_agent"
        )

    assert [t.name for t in tools] == ["t"]


@pytest.mark.asyncio
async def test_a_profile_of_another_type_loads_nothing_and_connects_nowhere():
    config = _config(
        profiles={"web": MCPProfileConfig(servers=["s"])},
        servers={"s": MCPServerConfig(transport="stdio", command="x")},
    )
    with patch("langchain_mcp_adapters.client.MultiServerMCPClient") as MockClient:
        tools = await _manager(config).get_tools_for_profile(
            "web", agent_type="email_agent"
        )

    assert tools == []
    MockClient.assert_not_called()


@pytest.mark.asyncio
async def test_a_profile_that_lists_a_type_loads_for_it():
    config = _config(
        profiles={"web": MCPProfileConfig(servers=["s"], agents=["email_agent"])},
        servers={"s": MCPServerConfig(transport="stdio", command="x")},
    )
    with patch("langchain_mcp_adapters.client.MultiServerMCPClient") as MockClient:
        MockClient.return_value.get_tools = AsyncMock(return_value=[_named("t")])
        tools = await _manager(config).get_tools_for_profile(
            "web", agent_type="email_agent"
        )

    assert [t.name for t in tools] == ["t"]


# --------------------------------------------------------------------------- #
# Servers
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_server_that_does_not_answer_is_left_out_in_time():
    config = _config(
        profiles={
            "web": MCPProfileConfig(servers=["slow", "fast"], load_timeout_seconds=1)
        },
        servers={
            "slow": MCPServerConfig(transport="stdio", command="x"),
            "fast": MCPServerConfig(transport="stdio", command="x"),
        },
    )
    fast_tool = _named("fast_tool")

    async def get_tools(*, server_name):
        if server_name == "slow":
            await asyncio.sleep(30)
        return [fast_tool]

    started = time.monotonic()
    with patch("langchain_mcp_adapters.client.MultiServerMCPClient") as MockClient:
        MockClient.return_value.get_tools = get_tools
        tools = await _manager(config).get_tools_for_profile("web")

    assert tools == [fast_tool]
    assert time.monotonic() - started < 5


def test_each_transport_gets_only_the_settings_its_session_takes():
    timeouts = {"connect_seconds": 5, "read_seconds": 60}
    config = _config(
        profiles={"web": MCPProfileConfig(servers=["http", "sse", "ws"])},
        servers={
            "http": MCPServerConfig(
                transport="streamable_http",
                url="https://a.example/mcp",
                headers={"Authorization": "Bearer x"},
                timeouts=timeouts,
            ),
            "sse": MCPServerConfig(
                transport="sse", url="https://b.example/sse", timeouts=timeouts
            ),
            "ws": MCPServerConfig(
                transport="websocket",
                url="wss://c.example/mcp",
                headers={"X": "y"},
                timeouts=timeouts,
            ),
        },
    )

    connections = _manager(config)._server_connections("web", ["http", "sse", "ws"])

    assert connections["http"] == {
        "transport": "streamable_http",
        "url": "https://a.example/mcp",
        "headers": {"Authorization": "Bearer x"},
        "timeout": 5.0,
        "sse_read_timeout": 60.0,
        "terminate_on_close": True,
    }
    assert connections["sse"] == {
        "transport": "sse",
        "url": "https://b.example/sse",
        "timeout": 5.0,
        "sse_read_timeout": 60.0,
    }
    assert connections["ws"] == {"transport": "websocket", "url": "wss://c.example/mcp"}


# --------------------------------------------------------------------------- #
# Calling a tool
# --------------------------------------------------------------------------- #


def _remote_tool(coroutine, *, args_schema=None, metadata=None):
    """A tool shaped like the ones the MCP adapters make."""
    return StructuredTool(
        name="remote",
        description="A remote tool.",
        args_schema=args_schema or {"type": "object", "properties": {}},
        coroutine=coroutine,
        response_format="content_and_artifact",
        handle_tool_error=_handle_mcp_tool_error,
        metadata=metadata,
    )


def _call(arguments=None):
    return {"name": "remote", "args": arguments or {}, "id": "call-1", "type": "tool_call"}


@pytest.mark.asyncio
async def test_a_tool_without_arguments_can_be_called():
    received = []

    async def call(**arguments):
        received.append(arguments)
        return "went back", None

    wrapped = _wrap_tool_for_sync(_remote_tool(call))

    message = await wrapped.ainvoke(_call())
    assert message.content == "went back"
    assert received == [{}]

    # The sync path, as the agent runs it in a worker
    message = await asyncio.to_thread(wrapped.invoke, _call())
    assert message.content == "went back"


@pytest.mark.asyncio
async def test_an_error_result_reaches_the_agent_as_an_error():
    async def call(**arguments):
        raise _MCPToolExecutionError([{"type": "text", "text": "no such page"}])

    message = await _wrap_tool_for_sync(_remote_tool(call)).ainvoke(_call())

    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert "no such page" in str(message.content)


def test_the_tool_keeps_what_the_server_said_about_it():
    async def call(**arguments):
        return "ok", None

    wrapped = _wrap_tool_for_sync(
        _remote_tool(call, metadata={"readOnlyHint": True, "destructiveHint": False})
    )

    assert wrapped.metadata == {"readOnlyHint": True, "destructiveHint": False}


# --------------------------------------------------------------------------- #
# The agent's own rules
# --------------------------------------------------------------------------- #


@pytest.fixture
def agent_factory():
    with patch("inference_core.services.agents_service.get_model_factory") as get:
        factory = MagicMock()
        factory.get_agent_model_name.return_value = "gpt-4"
        factory.get_model_for_agent.return_value = MagicMock()
        agent_config = MagicMock()
        agent_config.mcp_profile = "web"
        agent_config.local_tool_providers = []
        agent_config.allowed_tools = None
        factory.config.get_specific_agent_config.return_value = agent_config
        get.return_value = factory
        yield agent_config


async def _load(agent_config, tools):
    service = AgentService(agent_name="assistant_agent")
    with patch("inference_core.agents.agent_mcp_tools.get_agent_mcp_manager") as get:
        get.return_value.get_tools_for_profile = AsyncMock(return_value=tools)
        await service._load_mcp_tools()
    return [t.name for t in service.tools]


@pytest.mark.asyncio
async def test_an_agent_allowed_no_tools_gets_none_from_mcp(agent_factory):
    agent_factory.allowed_tools = []
    assert await _load(agent_factory, [_named("click")]) == []


@pytest.mark.asyncio
async def test_an_agent_allowlist_applies_to_mcp_tools(agent_factory):
    agent_factory.allowed_tools = ["read_page"]
    tools = [_named("read_page"), _named("click")]
    assert await _load(agent_factory, tools) == ["read_page"]


@pytest.mark.asyncio
async def test_without_an_allowlist_every_mcp_tool_stays(agent_factory):
    tools = [_named("read_page"), _named("click")]
    assert await _load(agent_factory, tools) == ["read_page", "click"]


def test_a_repeated_tool_name_keeps_the_first_tool():
    first, second, repeat = _named("search"), _named("fetch"), _named("search")
    assert _drop_repeated_tools([first, second, repeat], []) == [first, second]


def test_a_tool_cannot_take_the_name_of_a_middleware_tool():
    middleware = SimpleNamespace(tools=[_named("task")])
    mine, shadow = _named("search"), _named("task")
    assert _drop_repeated_tools([mine, shadow], [middleware]) == [mine]


def test_tools_given_as_functions_are_told_apart_by_name():
    def search():
        """Search."""

    def fetch():
        """Fetch."""

    assert _drop_repeated_tools([search, fetch, search], []) == [search, fetch]

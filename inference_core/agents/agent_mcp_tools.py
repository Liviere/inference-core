import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.structured import StructuredTool

from inference_core.celery.async_utils import run_async_safely
from inference_core.llm.config import get_llm_config

logger = logging.getLogger(__name__)


def _call_payload(args: tuple, kwargs: dict) -> Any:
    """What the wrapped tool is invoked with, from the wrapper's own arguments."""
    if kwargs:
        return kwargs
    if len(args) == 1:
        return args[0]
    if not args:
        # A tool that takes no arguments: an empty list is no valid tool input.
        return {}
    return list(args)


def _wrap_tool_for_sync(tool: BaseTool) -> BaseTool:
    """Wrap async-only StructuredTool to provide sync _run.

    Uses run_async_safely() to reuse the Celery worker loop when available,
    avoiding creation of conflicting event loops in nested tool calls.

    A failed call is handled by the wrapper, not by the wrapped tool: the
    wrapped tool is invoked without a tool call id, so the error output it
    would make for itself reaches the agent as an ordinary result. Raised to
    the wrapper instead, the agent's tool message keeps ``status="error"``.
    """

    if not isinstance(tool, StructuredTool):
        return tool

    error_handler = tool.handle_tool_error
    tool.handle_tool_error = False

    def _normalize_result(result: Any, response_format: str) -> Any:
        """Ensure output matches declared response format to avoid runtime errors."""

        if response_format == "content_and_artifact":
            if isinstance(result, tuple) and len(result) == 2:
                return result
            return (result, result)
        return result

    class SyncWrapper(BaseTool):
        name: str = tool.name
        description: str = getattr(tool, "description", "") or ""
        args_schema: Any = getattr(tool, "args_schema", None)
        return_direct: bool = getattr(tool, "return_direct", False)
        response_format: str = getattr(tool, "response_format", "content")
        extras: Any = getattr(tool, "extras", None)
        metadata: Any = getattr(tool, "metadata", None)
        handle_tool_error: Any = error_handler

        def _run(self, *args, **kwargs):
            payload = _call_payload(args, kwargs)
            result = run_async_safely(tool.ainvoke(payload))
            return _normalize_result(result, self.response_format)

        async def _arun(self, *args, **kwargs):
            payload = _call_payload(args, kwargs)
            result = await tool.ainvoke(payload)
            return _normalize_result(result, self.response_format)

    return SyncWrapper()


class AgentMCPToolManager:
    """
    Manages MCP tools for Agents using the new LangChain MCP adapters.

    No client is kept between calls: a client holds only the connections of a
    profile, and every tool opens a session of its own when it is called. A
    client per call keeps whatever a caller passes to one from reaching any
    other caller.
    """

    def __init__(self):
        self.config = get_llm_config()

    def __filter_tools(self, tools: List[Any], profile, profile_name: str) -> List[Any]:
        """
        Filter tools based on profile configuration.
        Args:
            tools: List of LangChain tools.
            profile: MCP profile configuration.
            profile_name: The name of the MCP profile.
        Returns:
            Filtered list of LangChain tools.
        """
        # Filter tools if include_tools is configured
        if profile.include_tools is not None:
            original_count = len(tools)
            tools = [t for t in tools if t.name in profile.include_tools]
            logger.debug(
                f"Filtered MCP tools for profile '{profile_name}': {original_count} -> {len(tools)}"
            )
        # Ensure sync compatibility (StructuredTool lacks _run)
        wrapped = []
        for t in tools:
            try:
                wrapped.append(_wrap_tool_for_sync(t))
            except Exception as e:
                logger.warning(
                    f"Failed to wrap tool '{getattr(t, 'name', '<unknown>')}' for sync: {e}"
                )
                wrapped.append(t)
        return wrapped

    async def get_tools_for_profile(
        self, profile_name: str, *, agent_type: Optional[str] = None
    ) -> List[Any]:
        """
        Get LangChain-compatible tools for a specific MCP profile.

        Args:
            profile_name: The name of the MCP profile to load tools for.
            agent_type: The agent type the tools are for. When given, a profile
                that is not available to that type loads nothing. Who may
                select a profile is checked when it is saved, not here.

        Returns:
            List of LangChain tools.
        """
        if not self.config.mcp_config.enabled:
            logger.debug("MCP is disabled globally.")
            return []

        profile = self.config.mcp_config.get_profile(profile_name)
        if not profile:
            logger.warning(f"MCP profile '{profile_name}' not found.")
            return []

        if agent_type is not None:
            problem = self.config.mcp_profile_problem(
                profile_name, agent_type, is_superuser=True
            )
            if problem:
                logger.warning(
                    "Not loading MCP tools for agent type '%s': %s",
                    agent_type,
                    problem,
                )
                return []

        connections = self._server_connections(profile_name, profile.servers)
        if not connections:
            logger.warning(
                f"No valid server configurations found for profile '{profile_name}'"
            )
            return []
        client = self._new_client(profile_name, connections)
        if not client:
            return []

        # Each server is listed on its own, so one that fails or hangs does
        # not take the others' tools with it.
        per_server = await asyncio.gather(
            *(
                self._load_server_tools(
                    client, server_name, profile_name, profile.load_timeout_seconds
                )
                for server_name in connections
            )
        )
        tools = [tool for server_tools in per_server for tool in server_tools]
        tools = self.__filter_tools(tools, profile, profile_name)
        logger.info(f"Loaded {len(tools)} MCP tools for profile '{profile_name}'")
        return tools

    async def _load_server_tools(
        self, client: Any, server_name: str, profile_name: str, timeout: float
    ) -> List[Any]:
        try:
            return await asyncio.wait_for(
                client.get_tools(server_name=server_name), timeout=timeout
            )
        except Exception as e:
            logger.error(
                "Error getting tools from MCP server '%s' (profile '%s'): %s",
                server_name,
                profile_name,
                e,
            )
            return []

    def _server_connections(
        self, profile_name: str, server_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """The adapter's connection settings for each usable server of a profile."""
        server_configs: Dict[str, Dict[str, Any]] = {}

        for server_name in server_names:
            server_config = self.config.mcp_config.servers.get(server_name)
            if not server_config:
                logger.warning(
                    f"Server '{server_name}' referenced in profile '{profile_name}' not found in config."
                )
                continue

            config_dict: Dict[str, Any] = {
                "transport": server_config.transport,
            }

            if server_config.transport == "stdio":
                if not server_config.command:
                    continue
                config_dict["command"] = server_config.command
                config_dict["args"] = server_config.args or []
                if server_config.env:
                    config_dict["env"] = server_config.env
                if server_config.cwd:
                    config_dict["cwd"] = server_config.cwd

            elif server_config.transport == "websocket":
                # A websocket session takes the address and nothing else.
                if not server_config.url:
                    continue
                config_dict["url"] = server_config.url

            elif server_config.transport in ["streamable_http", "sse"]:
                if not server_config.url:
                    continue
                config_dict["url"] = server_config.url
                if server_config.headers:
                    config_dict["headers"] = server_config.headers
                if server_config.timeouts:
                    config_dict["timeout"] = float(
                        server_config.timeouts.connect_seconds
                    )
                    config_dict["sse_read_timeout"] = float(
                        server_config.timeouts.read_seconds
                    )
                # Only a streamable HTTP session ends the server's session on close.
                if (
                    server_config.transport == "streamable_http"
                    and server_config.terminate_on_close
                ):
                    config_dict["terminate_on_close"] = True

            server_configs[server_name] = config_dict

        return server_configs

    def _new_client(
        self, profile_name: str, connections: Dict[str, Dict[str, Any]]
    ) -> Optional[Any]:
        """A MultiServerMCPClient for these connections."""
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            logger.error(
                "langchain-mcp-adapters not installed. Run: pip install langchain-mcp-adapters mcp"
            )
            return None

        try:
            return MultiServerMCPClient(connections)
        except Exception as e:
            logger.error(
                f"Failed to initialize MCP client for profile '{profile_name}': {e}"
            )
            return None

    async def close(self):
        """Release what the manager holds. No client is kept, so nothing is."""


# Global instance
_agent_mcp_manager = None


def get_agent_mcp_manager() -> AgentMCPToolManager:
    global _agent_mcp_manager
    if _agent_mcp_manager is None:
        _agent_mcp_manager = AgentMCPToolManager()
    return _agent_mcp_manager


async def close_agent_mcp_manager() -> None:
    """Close the global Agent MCP manager if it was initialized.

    WHY: Application and worker shutdown should not instantiate MCP just to
    close it, but any existing manager must release transport resources.
    """
    global _agent_mcp_manager
    if _agent_mcp_manager is None:
        return
    await _agent_mcp_manager.close()
    _agent_mcp_manager = None

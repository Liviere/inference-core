import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.structured import StructuredTool

from inference_core.celery.async_utils import run_async_safely
from inference_core.llm.config import get_llm_config

logger = logging.getLogger(__name__)


def _wrap_tool_for_sync(tool: BaseTool) -> BaseTool:
    """Wrap async-only StructuredTool to provide sync _run.

    Uses run_async_safely() to reuse the Celery worker loop when available,
    avoiding creation of conflicting event loops in nested tool calls.
    """

    if not isinstance(tool, StructuredTool):
        return tool

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

        def _run(self, *args, **kwargs):
            payload = kwargs if kwargs else (args[0] if len(args) == 1 else list(args))
            result = run_async_safely(tool.ainvoke(payload))
            return _normalize_result(result, self.response_format)

        async def _arun(self, *args, **kwargs):
            payload = kwargs if kwargs else (args[0] if len(args) == 1 else list(args))
            result = await tool.ainvoke(payload)
            return _normalize_result(result, self.response_format)

    return SyncWrapper()


class AgentMCPToolManager:
    """
    Manages MCP tools for Agents using the new LangChain MCP adapters.
    """

    def __init__(self):
        self.config = get_llm_config()
        self._clients: Dict[str, Any] = {}  # Cache clients by profile name

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

    async def get_tools_for_profile(self, profile_name: str) -> List[Any]:
        """
        Get LangChain-compatible tools for a specific MCP profile.

        Args:
            profile_name: The name of the MCP profile to load tools for.

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

        # Check if we already have a client for this profile
        if profile_name in self._clients:
            try:
                # In the new adapter, we might need to ensure connection is active
                # But typically get_tools() handles it or we assume it's persistent/stateless enough
                tools = await self._clients[profile_name].get_tools()
                tools = self.__filter_tools(tools, profile, profile_name)
                logger.info(
                    f"Loaded {len(tools)} MCP tools from cache for profile '{profile_name}'"
                )
                return tools
            except Exception as e:
                logger.error(
                    f"Error getting tools from cached client for profile '{profile_name}': {e}"
                )
                # If cached client fails, try to recreate it
                del self._clients[profile_name]

        # Create new client
        client = await self._create_client_for_profile(profile_name, profile.servers)
        if not client:
            return []

        self._clients[profile_name] = client

        try:
            tools = await client.get_tools()
            tools = self.__filter_tools(tools, profile, profile_name)
            logger.info(f"Loaded {len(tools)} MCP tools for profile '{profile_name}'")
            return tools
        except Exception as e:
            logger.error(f"Error getting tools for profile '{profile_name}': {e}")
            return []

    async def _create_client_for_profile(
        self, profile_name: str, server_names: List[str]
    ) -> Optional[Any]:
        """Create a MultiServerMCPClient for a specific set of servers."""
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            logger.error(
                "langchain-mcp-adapters not installed. Run: pip install langchain-mcp-adapters mcp"
            )
            return None

        server_configs = {}

        for server_name in server_names:
            server_config = self.config.mcp_config.servers.get(server_name)
            if not server_config:
                logger.warning(
                    f"Server '{server_name}' referenced in profile '{profile_name}' not found in config."
                )
                continue

            config_dict = {
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

            elif server_config.transport in ["streamable_http", "sse", "websocket"]:
                if not server_config.url:
                    continue
                config_dict["url"] = server_config.url
                if server_config.headers:
                    config_dict["headers"] = server_config.headers

            server_configs[server_name] = config_dict

        if not server_configs:
            logger.warning(
                f"No valid server configurations found for profile '{profile_name}'"
            )
            return None

        try:
            # Initialize the multi-server client
            client = MultiServerMCPClient(server_configs)
            return client
        except Exception as e:
            logger.error(
                f"Failed to initialize MCP client for profile '{profile_name}': {e}"
            )
            return None

    async def close(self):
        """Close all managed clients."""
        self._clients.clear()


# Global instance
_agent_mcp_manager = None


def get_agent_mcp_manager() -> AgentMCPToolManager:
    global _agent_mcp_manager
    if _agent_mcp_manager is None:
        _agent_mcp_manager = AgentMCPToolManager()
    return _agent_mcp_manager

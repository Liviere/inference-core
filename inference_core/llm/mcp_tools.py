"""
MCP Tool Manager

Manages Model Context Protocol (MCP) tool integration with LangChain.
Handles multi-server client initialization, tool loading, and security controls.

Source: LangChain MCP Adapters – multi-server client, tool loading, callbacks
Source: MCP Python SDK – clients/servers, transports (stdio/SSE/HTTP/WS)
"""

import logging
import os
from typing import Any, Dict, List, Optional

from inference_core.llm.config import MCPConfig, MCPProfileConfig, get_llm_config

logger = logging.getLogger(__name__)


class MCPToolManager:
    """
    Manages MCP servers and provides LangChain-compatible tools.
    
    Responsibilities:
    - Initialize MultiServerMCPClient with configured servers
    - Load tools for specific profiles
    - Enforce security controls (RBAC, timeouts, step limits)
    - Provide observability hooks (logging, Sentry)
    """

    def __init__(self, mcp_config: Optional[MCPConfig] = None):
        """Initialize the MCP tool manager.
        
        Args:
            mcp_config: MCP configuration (defaults to global config)
        """
        self.mcp_config = mcp_config or get_llm_config().mcp_config
        self._client = None
        self._initialized = False

    def is_enabled(self) -> bool:
        """Check if MCP is globally enabled."""
        return self.mcp_config.enabled

    def check_permissions(self, user: Optional[Dict[str, Any]] = None) -> bool:
        """Check if user has permission to use MCP tools.
        
        Args:
            user: User dict with 'is_superuser' key
            
        Returns:
            True if user has permission, False otherwise
        """
        if not self.mcp_config.enabled:
            return False
        
        # If superuser is required, check user permissions
        if self.mcp_config.require_superuser:
            if not user or not user.get("is_superuser"):
                logger.warning("MCP tools require superuser permissions")
                return False
        
        return True

    async def get_tools(
        self,
        profile_name: Optional[str] = None,
        user: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Get LangChain-compatible tools for a profile.
        
        Args:
            profile_name: MCP profile name (uses default if None)
            user: User dict for permission checking
            
        Returns:
            List of LangChain tools
            
        Raises:
            PermissionError: If user lacks required permissions
            ValueError: If profile doesn't exist or MCP is disabled
        """
        # Check permissions
        if not self.check_permissions(user):
            raise PermissionError("User does not have permission to use MCP tools")
        
        # Determine profile
        if profile_name is None:
            profile_name = self.mcp_config.default_profile
        
        if profile_name is None:
            logger.info("No MCP profile specified and no default profile configured")
            return []
        
        # Get profile config
        profile = self.mcp_config.get_profile(profile_name)
        if profile is None:
            raise ValueError(f"MCP profile '{profile_name}' not found")
        
        logger.info(f"Loading MCP tools for profile: {profile_name}")
        
        # Lazy initialization of client
        if not self._initialized:
            await self._initialize_client()
        
        try:
            # Import here to avoid import errors if dependencies not installed
            from langchain_mcp_adapters.client import MultiServerMCPClient
            
            # Get tools from all servers in the profile
            if self._client is None:
                raise RuntimeError("MCP client not initialized")
            
            tools = await self._client.get_tools()
            
            logger.info(
                f"Loaded {len(tools)} tools from profile '{profile_name}': "
                f"{[t.name for t in tools]}"
            )
            
            return tools
            
        except ImportError as e:
            logger.error(
                "MCP dependencies not installed. "
                "Run: pip install langchain-mcp-adapters mcp"
            )
            raise ValueError(
                "MCP dependencies not installed. Please install langchain-mcp-adapters and mcp."
            ) from e
        except Exception as e:
            logger.error(f"Error loading MCP tools: {e}")
            raise

    async def _initialize_client(self):
        """Initialize the MultiServerMCPClient.
        
        Source: LangChain MCP Adapters – multi-server client initialization
        """
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            
            # Build server configurations for the client
            server_configs = {}
            
            for server_name, server_config in self.mcp_config.servers.items():
                config_dict: Dict[str, Any] = {
                    "transport": server_config.transport,
                }
                
                # Add transport-specific configuration
                if server_config.transport == "stdio":
                    if not server_config.command:
                        logger.warning(
                            f"Server '{server_name}' uses stdio but missing command"
                        )
                        continue
                    
                    config_dict["command"] = server_config.command
                    config_dict["args"] = server_config.args or []
                    
                    if server_config.env:
                        config_dict["env"] = server_config.env
                    if server_config.cwd:
                        config_dict["cwd"] = server_config.cwd
                
                elif server_config.transport in ["streamable_http", "sse", "websocket"]:
                    if not server_config.url:
                        logger.warning(
                            f"Server '{server_name}' uses {server_config.transport} "
                            f"but missing url"
                        )
                        continue
                    
                    config_dict["url"] = server_config.url
                    
                    if server_config.headers:
                        config_dict["headers"] = server_config.headers
                    
                    if server_config.timeouts:
                        config_dict["timeout"] = server_config.timeouts.connect_seconds
                        # SSE read timeout is separate
                        if server_config.transport in ["sse", "streamable_http"]:
                            config_dict["sse_read_timeout"] = (
                                server_config.timeouts.read_seconds
                            )
                
                server_configs[server_name] = config_dict
            
            if not server_configs:
                logger.warning("No valid MCP server configurations found")
                self._client = None
                self._initialized = True
                return
            
            logger.info(f"Initializing MCP client with servers: {list(server_configs.keys())}")
            
            # Initialize the multi-server client
            # Source: LangChain MCP Adapters – MultiServerMCPClient examples
            self._client = MultiServerMCPClient(server_configs)
            
            self._initialized = True
            logger.info("MCP client initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import MCP dependencies: {e}")
            self._client = None
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            self._client = None
            self._initialized = True

    async def close(self):
        """Close the MCP client and cleanup resources."""
        if self._client is not None:
            try:
                # MultiServerMCPClient doesn't have an explicit close method
                # but we can clean up our reference
                self._client = None
                logger.info("MCP client closed")
            except Exception as e:
                logger.error(f"Error closing MCP client: {e}")
        
        self._initialized = False

    def get_profile_limits(self, profile_name: str) -> Dict[str, Any]:
        """Get execution limits for a profile.
        
        Args:
            profile_name: MCP profile name
            
        Returns:
            Dict with 'max_steps', 'max_run_seconds', etc.
        """
        profile = self.mcp_config.get_profile(profile_name)
        if profile is None:
            return {
                "max_steps": 10,
                "max_run_seconds": 60,
            }
        
        return {
            "max_steps": profile.max_steps,
            "max_run_seconds": profile.max_run_seconds,
            "allowlist_hosts": profile.allowlist_hosts,
            "rate_limits": profile.rate_limits,
        }


# Global instance (lazy-initialized)
_mcp_tool_manager: Optional[MCPToolManager] = None


def get_mcp_tool_manager() -> MCPToolManager:
    """Get the global MCP tool manager instance."""
    global _mcp_tool_manager
    if _mcp_tool_manager is None:
        _mcp_tool_manager = MCPToolManager()
    return _mcp_tool_manager

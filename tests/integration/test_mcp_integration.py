"""
Integration tests for MCP tool functionality

Tests the full flow of MCP tool integration using a simple math server.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock

from inference_core.llm.config import MCPConfig, MCPServerConfig, MCPProfileConfig
from inference_core.llm.mcp_tools import MCPToolManager


@pytest.mark.integration
class TestMCPIntegration:
    """Integration tests for MCP with real MCP server (math example)"""

    @pytest.fixture
    def math_server_path(self):
        """Get path to example math server"""
        repo_root = Path(__file__).parent.parent.parent
        server_path = repo_root / "examples" / "mcp_math_server.py"
        assert server_path.exists(), f"Math server not found at {server_path}"
        return str(server_path)

    @pytest.fixture
    def mcp_config_with_math(self, math_server_path):
        """Create MCP config with math server"""
        return MCPConfig(
            enabled=True,
            require_superuser=False,  # Allow non-superuser for testing
            default_profile="math",
            profiles={
                "math": MCPProfileConfig(
                    description="Math operations",
                    servers=["math"],
                    max_steps=5,
                    max_run_seconds=30
                )
            },
            servers={
                "math": MCPServerConfig(
                    transport="stdio",
                    command="python",
                    args=[math_server_path],
                    env=None,
                    cwd=None
                )
            }
        )

    @pytest.mark.asyncio
    async def test_mcp_tool_manager_loads_tools(self, mcp_config_with_math):
        """Test that MCPToolManager can load tools from math server"""
        manager = MCPToolManager(mcp_config=mcp_config_with_math)
        
        try:
            # Get tools for math profile
            tools = await manager.get_tools(profile_name="math", user=None)
            
            # Should have loaded tools (add and multiply)
            assert len(tools) >= 2, f"Expected at least 2 tools, got {len(tools)}"
            
            tool_names = [t.name for t in tools]
            assert "add" in tool_names, f"'add' tool not found. Tools: {tool_names}"
            assert "multiply" in tool_names, f"'multiply' tool not found. Tools: {tool_names}"
            
        finally:
            # Cleanup
            await manager.close()

    @pytest.mark.asyncio
    async def test_mcp_tool_execution(self, mcp_config_with_math):
        """Test executing tools loaded from MCP server"""
        manager = MCPToolManager(mcp_config=mcp_config_with_math)
        
        try:
            # Get tools
            tools = await manager.get_tools(profile_name="math", user=None)
            
            # Find the add tool
            add_tool = next((t for t in tools if t.name == "add"), None)
            assert add_tool is not None, "Add tool not found"
            
            # Execute the tool
            # Source: LangChain tools – ainvoke method for async execution
            result = await add_tool.ainvoke({"a": 5, "b": 3})
            
            # Should return the sum
            # Note: result format may vary (could be str or structured)
            if isinstance(result, str):
                assert "8" in result, f"Expected '8' in result, got: {result}"
            else:
                assert result == 8, f"Expected 8, got: {result}"
            
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_mcp_permissions_enforcement(self, mcp_config_with_math):
        """Test that MCP respects permission settings"""
        # Create config that requires superuser
        strict_config = MCPConfig(
            enabled=True,
            require_superuser=True,
            default_profile="math",
            profiles=mcp_config_with_math.profiles,
            servers=mcp_config_with_math.servers
        )
        
        manager = MCPToolManager(mcp_config=strict_config)
        
        # Non-superuser should be denied
        non_superuser = {"is_superuser": False}
        with pytest.raises(PermissionError, match="does not have permission"):
            await manager.get_tools(profile_name="math", user=non_superuser)
        
        # Superuser should be allowed
        superuser = {"is_superuser": True}
        tools = await manager.get_tools(profile_name="math", user=superuser)
        assert len(tools) > 0
        
        await manager.close()

    @pytest.mark.asyncio
    async def test_mcp_disabled_returns_empty(self):
        """Test that disabled MCP returns empty tools list"""
        disabled_config = MCPConfig(
            enabled=False,  # Disabled
            require_superuser=False,
            default_profile=None,
            profiles={},
            servers={}
        )
        
        manager = MCPToolManager(mcp_config=disabled_config)
        
        # Should not be able to get tools when disabled
        with pytest.raises(PermissionError):
            await manager.get_tools(profile_name="test", user=None)


@pytest.mark.integration
class TestMCPErrorHandling:
    """Test error handling in MCP integration"""

    @pytest.mark.asyncio
    async def test_mcp_server_not_found(self):
        """Test handling of non-existent MCP server"""
        bad_config = MCPConfig(
            enabled=True,
            require_superuser=False,
            default_profile="bad",
            profiles={
                "bad": MCPProfileConfig(
                    description="Bad server",
                    servers=["nonexistent"],
                    max_steps=5,
                    max_run_seconds=30
                )
            },
            servers={
                "nonexistent": MCPServerConfig(
                    transport="stdio",
                    command="python",
                    args=["/nonexistent/path/server.py"]
                )
            }
        )
        
        manager = MCPToolManager(mcp_config=bad_config)
        
        # Should raise an error when trying to connect to nonexistent server
        with pytest.raises(Exception):  # Could be various exception types
            tools = await manager.get_tools(profile_name="bad", user=None)
        
        await manager.close()

    @pytest.mark.asyncio
    async def test_mcp_invalid_transport(self):
        """Test handling of MCP server with invalid config"""
        # Missing URL for HTTP transport - validation happens in MCPServerConfig
        # We're just testing that the config allows None url (validation can be optional)
        # The actual error will happen when trying to initialize the client
        config_with_missing_url = MCPConfig(
            enabled=True,
            require_superuser=False,
            default_profile="bad",
            profiles={
                "bad": MCPProfileConfig(
                    description="Bad HTTP server",
                    servers=["bad_http"],
                    max_steps=5,
                    max_run_seconds=30
                )
            },
            servers={
                "bad_http": MCPServerConfig(
                    transport="streamable_http",
                    # Missing url - should be caught during client init, not config creation
                    url=None
                )
            }
        )
        
        manager = MCPToolManager(mcp_config=config_with_missing_url)
        
        # Initialization should skip servers with missing required fields
        # and either return empty tools or raise an error
        try:
            tools = await manager.get_tools(profile_name="bad", user=None)
            # If no error, should return empty (no valid servers configured)
            assert len(tools) == 0, "Expected empty tools list for misconfigured server"
        except Exception:
            # Also acceptable to raise an error
            pass
        finally:
            await manager.close()

"""
Unit tests for MCP Tool Manager
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from inference_core.llm.mcp_tools import MCPToolManager, get_mcp_tool_manager
from inference_core.llm.config import (
    MCPConfig,
    MCPServerConfig,
    MCPProfileConfig,
)


class TestMCPToolManager:
    """Test MCPToolManager functionality"""

    def test_init_with_default_config(self):
        """Test initialization with default config"""
        manager = MCPToolManager()
        assert manager.mcp_config is not None
        assert isinstance(manager.mcp_config, MCPConfig)
        assert manager._client is None
        assert manager._initialized is False

    def test_init_with_custom_config(self):
        """Test initialization with custom config"""
        custom_config = MCPConfig(
            enabled=True,
            require_superuser=False
        )
        manager = MCPToolManager(mcp_config=custom_config)
        assert manager.mcp_config == custom_config
        assert manager.mcp_config.enabled is True

    def test_is_enabled_false_by_default(self):
        """Test MCP is disabled by default"""
        manager = MCPToolManager()
        assert manager.is_enabled() is False

    def test_is_enabled_true_when_configured(self):
        """Test MCP enabled when configured"""
        config = MCPConfig(enabled=True)
        manager = MCPToolManager(mcp_config=config)
        assert manager.is_enabled() is True

    def test_check_permissions_mcp_disabled(self):
        """Test permission check when MCP is disabled"""
        config = MCPConfig(enabled=False)
        manager = MCPToolManager(mcp_config=config)
        
        # Should fail even with superuser
        user = {"is_superuser": True}
        assert manager.check_permissions(user) is False

    def test_check_permissions_no_user_superuser_required(self):
        """Test permission check with no user when superuser required"""
        config = MCPConfig(enabled=True, require_superuser=True)
        manager = MCPToolManager(mcp_config=config)
        
        # Should fail without user
        assert manager.check_permissions(None) is False
        assert manager.check_permissions({}) is False

    def test_check_permissions_non_superuser_when_required(self):
        """Test permission check with non-superuser when superuser required"""
        config = MCPConfig(enabled=True, require_superuser=True)
        manager = MCPToolManager(mcp_config=config)
        
        # Should fail for non-superuser
        user = {"is_superuser": False}
        assert manager.check_permissions(user) is False

    def test_check_permissions_superuser_succeeds(self):
        """Test permission check with superuser succeeds"""
        config = MCPConfig(enabled=True, require_superuser=True)
        manager = MCPToolManager(mcp_config=config)
        
        # Should succeed with superuser
        user = {"is_superuser": True}
        assert manager.check_permissions(user) is True

    def test_check_permissions_no_superuser_requirement(self):
        """Test permission check when superuser not required"""
        config = MCPConfig(enabled=True, require_superuser=False)
        manager = MCPToolManager(mcp_config=config)
        
        # Should succeed even without superuser
        user = {"is_superuser": False}
        assert manager.check_permissions(user) is True
        
        # Should also work with no user
        assert manager.check_permissions(None) is True

    @pytest.mark.asyncio
    async def test_get_tools_permission_denied(self):
        """Test get_tools raises PermissionError when denied"""
        config = MCPConfig(enabled=True, require_superuser=True)
        manager = MCPToolManager(mcp_config=config)
        
        # Should raise PermissionError for non-superuser
        user = {"is_superuser": False}
        with pytest.raises(PermissionError, match="does not have permission"):
            await manager.get_tools(profile_name="test", user=user)

    @pytest.mark.asyncio
    async def test_get_tools_no_profile_no_default(self):
        """Test get_tools with no profile and no default configured"""
        config = MCPConfig(
            enabled=True,
            require_superuser=False,
            default_profile=None
        )
        manager = MCPToolManager(mcp_config=config)
        
        # Should return empty list
        tools = await manager.get_tools(profile_name=None, user=None)
        assert tools == []

    @pytest.mark.asyncio
    async def test_get_tools_profile_not_found(self):
        """Test get_tools with non-existent profile"""
        config = MCPConfig(
            enabled=True,
            require_superuser=False,
            profiles={}
        )
        manager = MCPToolManager(mcp_config=config)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="profile .* not found"):
            await manager.get_tools(profile_name="nonexistent", user=None)

    @pytest.mark.asyncio
    async def test_get_tools_with_default_profile(self):
        """Test get_tools uses default profile when not specified"""
        profile = MCPProfileConfig(
            description="Default profile",
            servers=["test"],
            max_steps=10,
            max_run_seconds=30
        )
        config = MCPConfig(
            enabled=True,
            require_superuser=False,
            default_profile="default",
            profiles={"default": profile},
            servers={
                "test": MCPServerConfig(
                    transport="stdio",
                    command="python",
                    args=["/path/to/server.py"]
                )
            }
        )
        manager = MCPToolManager(mcp_config=config)
        
        # Mock the client and tools - patch where it's imported
        with patch('langchain_mcp_adapters.client.MultiServerMCPClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_tools.return_value = []
            mock_client_class.return_value = mock_client
            
            # Should use default profile
            tools = await manager.get_tools(profile_name=None, user=None)
            
            # Client should have been initialized
            assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_client_stdio_server(self):
        """Test client initialization with stdio server"""
        config = MCPConfig(
            enabled=True,
            servers={
                "math": MCPServerConfig(
                    transport="stdio",
                    command="python",
                    args=["/path/to/math.py"],
                    env={"LOG_LEVEL": "info"},
                    cwd="/tmp"
                )
            }
        )
        manager = MCPToolManager(mcp_config=config)
        
        with patch('langchain_mcp_adapters.client.MultiServerMCPClient') as mock_client_class:
            await manager._initialize_client()
            
            # Should have been called with correct config
            assert mock_client_class.called
            call_args = mock_client_class.call_args[0][0]
            
            assert "math" in call_args
            math_config = call_args["math"]
            assert math_config["transport"] == "stdio"
            assert math_config["command"] == "python"
            assert math_config["args"] == ["/path/to/math.py"]
            assert math_config["env"] == {"LOG_LEVEL": "info"}
            assert math_config["cwd"] == "/tmp"

    @pytest.mark.asyncio
    async def test_initialize_client_http_server(self):
        """Test client initialization with streamable_http server"""
        config = MCPConfig(
            enabled=True,
            servers={
                "playwright": MCPServerConfig(
                    transport="streamable_http",
                    url="http://localhost:3000/mcp",
                    headers={"Authorization": "Bearer token"},
                    timeouts=None
                )
            }
        )
        manager = MCPToolManager(mcp_config=config)
        
        with patch('langchain_mcp_adapters.client.MultiServerMCPClient') as mock_client_class:
            await manager._initialize_client()
            
            # Should have been called with correct config
            assert mock_client_class.called
            call_args = mock_client_class.call_args[0][0]
            
            assert "playwright" in call_args
            pw_config = call_args["playwright"]
            assert pw_config["transport"] == "streamable_http"
            assert pw_config["url"] == "http://localhost:3000/mcp"
            assert pw_config["headers"] == {"Authorization": "Bearer token"}

    @pytest.mark.asyncio
    async def test_initialize_client_missing_command_for_stdio(self):
        """Test client initialization with stdio server missing command"""
        config = MCPConfig(
            enabled=True,
            servers={
                "bad_server": MCPServerConfig(
                    transport="stdio",
                    # Missing command
                    args=["/path/to/server.py"]
                )
            }
        )
        manager = MCPToolManager(mcp_config=config)
        
        with patch('langchain_mcp_adapters.client.MultiServerMCPClient') as mock_client_class:
            await manager._initialize_client()
            
            # Should not have been called (no valid servers)
            assert not mock_client_class.called

    @pytest.mark.asyncio
    async def test_initialize_client_missing_url_for_http(self):
        """Test client initialization with http server missing url"""
        config = MCPConfig(
            enabled=True,
            servers={
                "bad_server": MCPServerConfig(
                    transport="streamable_http",
                    # Missing url
                    headers={"Auth": "token"}
                )
            }
        )
        manager = MCPToolManager(mcp_config=config)
        
        with patch('langchain_mcp_adapters.client.MultiServerMCPClient') as mock_client_class:
            await manager._initialize_client()
            
            # Should not have been called (no valid servers)
            assert not mock_client_class.called

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing the MCP client"""
        config = MCPConfig(enabled=True)
        manager = MCPToolManager(mcp_config=config)
        
        # Set up a mock client
        manager._client = MagicMock()
        manager._initialized = True
        
        # Close should clean up
        await manager.close()
        
        assert manager._client is None
        assert manager._initialized is False

    def test_get_profile_limits(self):
        """Test get_profile_limits method"""
        profile = MCPProfileConfig(
            description="Test",
            servers=["test"],
            max_steps=5,
            max_run_seconds=60,
            allowlist_hosts=["example.com"],
            rate_limits=None
        )
        config = MCPConfig(
            profiles={"test": profile}
        )
        manager = MCPToolManager(mcp_config=config)
        
        limits = manager.get_profile_limits("test")
        
        assert limits["max_steps"] == 5
        assert limits["max_run_seconds"] == 60
        assert limits["allowlist_hosts"] == ["example.com"]

    def test_get_profile_limits_nonexistent(self):
        """Test get_profile_limits for non-existent profile"""
        manager = MCPToolManager()
        
        # Should return defaults
        limits = manager.get_profile_limits("nonexistent")
        
        assert limits["max_steps"] == 10
        assert limits["max_run_seconds"] == 60

    def test_get_mcp_tool_manager_singleton(self):
        """Test get_mcp_tool_manager returns singleton"""
        manager1 = get_mcp_tool_manager()
        manager2 = get_mcp_tool_manager()
        
        # Should be the same instance
        assert manager1 is manager2

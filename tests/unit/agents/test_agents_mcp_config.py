"""
Unit tests for Agent MCP configuration and tools.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_core.agents.agent_mcp_tools import AgentMCPToolManager
from inference_core.llm.config import MCPProfileConfig, MCPServerConfig
from inference_core.services.agents_service import AgentService


class TestAgentMCPToolManager:
    """Test AgentMCPToolManager functionality"""

    @pytest.fixture(autouse=True)
    def reset_manager(self):
        """Reset global MCP manager before and after each test"""
        import inference_core.agents.agent_mcp_tools

        inference_core.agents.agent_mcp_tools._agent_mcp_manager = None
        yield
        inference_core.agents.agent_mcp_tools._agent_mcp_manager = None

    @pytest.fixture
    def mock_config(self):
        """Mock global LLM config"""
        with patch("inference_core.agents.agent_mcp_tools.get_llm_config") as mock_get:
            config = MagicMock()
            # Mock MCPConfig instead of using real one
            mcp_config = MagicMock()
            mcp_config.enabled = False
            mcp_config.servers = {}
            config.mcp_config = mcp_config
            mock_get.return_value = config
            yield config

    @pytest.mark.asyncio
    async def test_get_tools_mcp_disabled(self, mock_config):
        """Test get_tools returns empty when MCP is disabled"""
        mock_config.mcp_config.enabled = False
        manager = AgentMCPToolManager()
        tools = await manager.get_tools_for_profile("test")
        assert tools == []

    @pytest.mark.asyncio
    async def test_get_tools_profile_not_found(self, mock_config):
        """Test get_tools returns empty when profile not found"""
        mock_config.mcp_config.enabled = True
        mock_config.mcp_config.get_profile.return_value = None

        manager = AgentMCPToolManager()
        tools = await manager.get_tools_for_profile("nonexistent")
        assert tools == []

    @pytest.mark.asyncio
    async def test_get_tools_client_creation_success(self, mock_config):
        """Test successful tool retrieval"""
        mock_config.mcp_config.enabled = True

        # Setup profile
        profile = MCPProfileConfig(
            description="test", servers=["test_server"], include_tools=None
        )
        mock_config.mcp_config.get_profile.return_value = profile

        # Setup server config
        server_config = MCPServerConfig(transport="stdio", command="python")
        mock_config.mcp_config.servers = {"test_server": server_config}

        manager = AgentMCPToolManager()

        # Mock MultiServerMCPClient
        with patch("langchain_mcp_adapters.client.MultiServerMCPClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_tool = MagicMock()
            mock_tool.name = "tool1"
            mock_client_instance.get_tools.return_value = [mock_tool]
            MockClient.return_value = mock_client_instance

            tools = await manager.get_tools_for_profile("test")

            assert len(tools) == 1
            assert tools[0].name == "tool1"

            # Verify client was cached
            assert "test" in manager._clients

    @pytest.mark.asyncio
    async def test_get_tools_with_filtering(self, mock_config):
        """Test tool filtering with include_tools"""
        mock_config.mcp_config.enabled = True

        # Setup profile with filtering
        profile = MCPProfileConfig(
            description="test",
            servers=["test_server"],
            include_tools=["tool1"],  # Only allow tool1
        )
        mock_config.mcp_config.get_profile.return_value = profile

        # Setup server config
        server_config = MCPServerConfig(transport="stdio", command="python")
        mock_config.mcp_config.servers = {"test_server": server_config}

        manager = AgentMCPToolManager()

        # Mock MultiServerMCPClient
        with patch("langchain_mcp_adapters.client.MultiServerMCPClient") as MockClient:
            mock_client_instance = AsyncMock()

            # Return two tools
            tool1 = MagicMock()
            tool1.name = "tool1"
            tool2 = MagicMock()
            tool2.name = "tool2"

            mock_client_instance.get_tools.return_value = [tool1, tool2]
            MockClient.return_value = mock_client_instance

            tools = await manager.get_tools_for_profile("test")

            # Should only have tool1
            assert len(tools) == 1
            assert tools[0].name == "tool1"

    @pytest.mark.asyncio
    async def test_get_tools_cached_client_error(self, mock_config):
        """Test recovery when cached client fails"""
        mock_config.mcp_config.enabled = True
        profile = MCPProfileConfig(description="test", servers=["test_server"])
        mock_config.mcp_config.get_profile.return_value = profile
        mock_config.mcp_config.servers = {
            "test_server": MCPServerConfig(transport="stdio", command="python")
        }

        manager = AgentMCPToolManager()

        # Pre-populate cache with a broken client
        broken_client = AsyncMock()
        broken_client.get_tools.side_effect = Exception("Connection lost")
        manager._clients["test"] = broken_client

        # Mock MultiServerMCPClient for the NEW client
        with patch("langchain_mcp_adapters.client.MultiServerMCPClient") as MockClient:
            new_client = AsyncMock()
            new_client.get_tools.return_value = []
            MockClient.return_value = new_client

            # Should catch exception, remove from cache, and create new one
            await manager.get_tools_for_profile("test")

            # Verify new client was created
            assert MockClient.called


class TestAgentServiceMCP:
    """Test AgentService MCP integration"""

    @pytest.fixture
    def mock_model_factory(self):
        """Mock model factory and config"""
        with patch(
            "inference_core.services.agents_service.get_model_factory"
        ) as mock_get:
            factory = MagicMock()
            factory.get_agent_model_name.return_value = "gpt-4"
            factory.get_model_for_agent.return_value = MagicMock()

            # Mock config
            agent_config = MagicMock()
            agent_config.mcp_profile = None
            agent_config.local_tool_providers = []
            factory.config.get_specific_agent_config.return_value = agent_config

            mock_get.return_value = factory
            yield factory

    @pytest.mark.asyncio
    async def test_load_providers_tools_passes_user_context(self, mock_model_factory):
        """Test _load_providers_tools forwards user-scoped context to providers."""
        mock_model_factory.config.get_specific_agent_config.return_value.local_tool_providers = [
            "test_provider"
        ]
        mock_model_factory.config.get_specific_agent_config.return_value.allowed_tools = [
            "tool1"
        ]

        service = AgentService(
            agent_name="test_agent",
            user_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            session_id="session-123",
            request_id="request-123",
        )

        with patch(
            "inference_core.services.agents_service.get_registered_providers",
            return_value={"test_provider": MagicMock()},
        ), patch(
            "inference_core.services.agents_service.load_tools_for_agent",
            new=AsyncMock(return_value=[]),
        ) as mock_loader:
            await service._load_providers_tools(
                user_context={"tenant_id": "tenant-1", "user_id": "custom-user"}
            )

            mock_loader.assert_awaited_once()
            _, loader_kwargs = mock_loader.await_args
            assert loader_kwargs["user_context"]["tenant_id"] == "tenant-1"
            assert loader_kwargs["user_context"]["user_id"] == "custom-user"
            assert loader_kwargs["session_id"] == "session-123"
            assert loader_kwargs["request_id"] == "request-123"

    @pytest.mark.asyncio
    async def test_load_providers_tools_adds_user_id_when_missing(
        self, mock_model_factory
    ):
        """Test _load_providers_tools fills missing user_id from AgentService identity."""
        mock_model_factory.config.get_specific_agent_config.return_value.local_tool_providers = [
            "test_provider"
        ]
        mock_model_factory.config.get_specific_agent_config.return_value.allowed_tools = (
            None
        )

        service = AgentService(
            agent_name="test_agent",
            user_id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
            session_id="session-abc",
            request_id="request-abc",
        )

        with patch(
            "inference_core.services.agents_service.get_registered_providers",
            return_value={"test_provider": MagicMock()},
        ), patch(
            "inference_core.services.agents_service.load_tools_for_agent",
            new=AsyncMock(return_value=[]),
        ) as mock_loader:
            await service._load_providers_tools(user_context={})

            mock_loader.assert_awaited_once()
            _, loader_kwargs = mock_loader.await_args
            assert loader_kwargs["user_id"] == "22222222-2222-2222-2222-222222222222"
            assert (
                loader_kwargs["user_context"]["user_id"]
                == "22222222-2222-2222-2222-222222222222"
            )
            assert loader_kwargs["session_id"] == "session-abc"
            assert loader_kwargs["request_id"] == "request-abc"

    @pytest.mark.asyncio
    async def test_load_mcp_tools_no_profile(self, mock_model_factory):
        """Test _load_mcp_tools does nothing when no profile configured"""
        service = AgentService(agent_name="test_agent")
        await service._load_mcp_tools()
        assert len(service.tools) == 0

    @pytest.mark.asyncio
    async def test_load_mcp_tools_with_profile(self, mock_model_factory):
        """Test _load_mcp_tools loads tools when profile configured"""
        # Setup agent config with profile
        mock_model_factory.config.get_specific_agent_config.return_value.mcp_profile = (
            "web"
        )

        service = AgentService(agent_name="test_agent")

        # Mock get_agent_mcp_manager
        with patch(
            "inference_core.agents.agent_mcp_tools.get_agent_mcp_manager"
        ) as mock_get_manager:
            manager = AsyncMock()
            tool = MagicMock()
            manager.get_tools_for_profile.return_value = [tool]
            mock_get_manager.return_value = manager

            await service._load_mcp_tools()

            assert len(service.tools) == 1
            assert service.tools[0] == tool

            # Verify manager was called with correct profile
            manager.get_tools_for_profile.assert_called_with("web")

    @pytest.mark.asyncio
    async def test_load_mcp_tools_error_handling(self, mock_model_factory):
        """Test _load_mcp_tools handles errors gracefully"""
        mock_model_factory.config.get_specific_agent_config.return_value.mcp_profile = (
            "web"
        )
        service = AgentService(agent_name="test_agent")

        with patch(
            "inference_core.agents.agent_mcp_tools.get_agent_mcp_manager"
        ) as mock_get_manager:
            manager = AsyncMock()
            manager.get_tools_for_profile.side_effect = Exception("MCP Error")
            mock_get_manager.return_value = manager

            # Should not raise exception
            await service._load_mcp_tools()

            assert len(service.tools) == 0

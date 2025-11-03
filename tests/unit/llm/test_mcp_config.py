"""
Unit tests for MCP configuration parsing
"""

import os
import pytest
from inference_core.llm.config import (
    MCPConfig,
    MCPServerConfig,
    MCPProfileConfig,
    MCPRateLimits,
    MCPServerTimeouts,
    get_llm_config,
)


class TestMCPConfigModels:
    """Test MCP configuration data models"""

    def test_mcp_server_timeouts(self):
        """Test MCPServerTimeouts model"""
        timeouts = MCPServerTimeouts(
            connect_seconds=10,
            read_seconds=300
        )
        assert timeouts.connect_seconds == 10
        assert timeouts.read_seconds == 300

    def test_mcp_server_config_stdio(self):
        """Test MCPServerConfig for stdio transport"""
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            args=["/path/to/server.py"],
            env={"LOG_LEVEL": "info"}
        )
        assert config.transport == "stdio"
        assert config.command == "python"
        assert config.args == ["/path/to/server.py"]
        assert config.env == {"LOG_LEVEL": "info"}

    def test_mcp_server_config_http(self):
        """Test MCPServerConfig for streamable_http transport"""
        config = MCPServerConfig(
            transport="streamable_http",
            url="http://localhost:3000/mcp",
            headers={"Authorization": "Bearer token"},
            timeouts=MCPServerTimeouts(connect_seconds=10, read_seconds=300)
        )
        assert config.transport == "streamable_http"
        assert config.url == "http://localhost:3000/mcp"
        assert config.headers == {"Authorization": "Bearer token"}
        assert config.timeouts.connect_seconds == 10

    def test_mcp_server_config_invalid_transport(self):
        """Test MCPServerConfig validation with invalid transport"""
        with pytest.raises(ValueError, match="transport must be one of"):
            MCPServerConfig(
                transport="invalid",
                url="http://example.com"
            )

    def test_mcp_rate_limits(self):
        """Test MCPRateLimits model"""
        limits = MCPRateLimits(
            requests_per_minute=30,
            tokens_per_minute=60000
        )
        assert limits.requests_per_minute == 30
        assert limits.tokens_per_minute == 60000

    def test_mcp_profile_config(self):
        """Test MCPProfileConfig model"""
        profile = MCPProfileConfig(
            description="Web browsing profile",
            servers=["playwright"],
            max_steps=5,
            max_run_seconds=60,
            allowlist_hosts=["example.com", "*.docs.example.com"],
            rate_limits=MCPRateLimits(
                requests_per_minute=30,
                tokens_per_minute=60000
            )
        )
        assert profile.description == "Web browsing profile"
        assert profile.servers == ["playwright"]
        assert profile.max_steps == 5
        assert profile.max_run_seconds == 60
        assert profile.allowlist_hosts == ["example.com", "*.docs.example.com"]
        assert profile.rate_limits.requests_per_minute == 30

    def test_mcp_config_defaults(self):
        """Test MCPConfig with default values"""
        config = MCPConfig()
        assert config.enabled is False
        assert config.require_superuser is True
        assert config.default_profile is None
        assert config.profiles == {}
        assert config.servers == {}

    def test_mcp_config_full(self):
        """Test MCPConfig with all options"""
        config = MCPConfig(
            enabled=True,
            require_superuser=False,
            default_profile="web-browsing",
            profiles={
                "web-browsing": MCPProfileConfig(
                    description="Web browsing",
                    servers=["playwright"],
                    max_steps=5,
                    max_run_seconds=60
                )
            },
            servers={
                "playwright": MCPServerConfig(
                    transport="streamable_http",
                    url="http://localhost:3000/mcp"
                )
            }
        )
        assert config.enabled is True
        assert config.require_superuser is False
        assert config.default_profile == "web-browsing"
        assert "web-browsing" in config.profiles
        assert "playwright" in config.servers

    def test_mcp_config_get_profile(self):
        """Test MCPConfig.get_profile method"""
        profile = MCPProfileConfig(
            description="Test",
            servers=["test"],
            max_steps=10,
            max_run_seconds=30
        )
        config = MCPConfig(
            profiles={"test": profile}
        )
        
        assert config.get_profile("test") == profile
        assert config.get_profile("nonexistent") is None

    def test_mcp_config_is_profile_enabled(self):
        """Test MCPConfig.is_profile_enabled method"""
        profile = MCPProfileConfig(
            description="Test",
            servers=["test"],
            max_steps=10,
            max_run_seconds=30
        )
        config = MCPConfig(
            profiles={"test": profile}
        )
        
        assert config.is_profile_enabled("test") is True
        assert config.is_profile_enabled("nonexistent") is False


class TestLLMConfigMCPLoading:
    """Test MCP configuration loading in LLMConfig"""

    def test_mcp_config_loaded_from_yaml(self):
        """Test that MCP config is loaded from YAML"""
        config = get_llm_config()
        
        # MCP config should exist
        assert config.mcp_config is not None
        assert isinstance(config.mcp_config, MCPConfig)
        
        # Default should be disabled
        assert config.mcp_config.enabled is False
        assert config.mcp_config.require_superuser is True

    def test_mcp_config_servers_loaded(self):
        """Test that MCP servers are loaded from YAML"""
        config = get_llm_config()
        
        # Check if example servers are in config
        servers = config.mcp_config.servers
        assert isinstance(servers, dict)
        
        # Playwright and math servers should be in example config
        if "playwright" in servers:
            playwright = servers["playwright"]
            assert playwright.transport == "streamable_http"
            assert playwright.url == "http://localhost:3000/mcp"
        
        if "math" in servers:
            math_server = servers["math"]
            assert math_server.transport == "stdio"
            assert math_server.command == "python"

    def test_mcp_config_profiles_loaded(self):
        """Test that MCP profiles are loaded from YAML"""
        config = get_llm_config()
        
        # Check if example profiles are in config
        profiles = config.mcp_config.profiles
        assert isinstance(profiles, dict)
        
        # Web-browsing and local-tools profiles should be in example config
        if "web-browsing" in profiles:
            web_profile = profiles["web-browsing"]
            assert "playwright" in web_profile.servers
            assert web_profile.max_steps == 5
            assert web_profile.max_run_seconds == 60
        
        if "local-tools" in profiles:
            local_profile = profiles["local-tools"]
            assert "math" in local_profile.servers
            assert local_profile.max_steps == 10

    def test_task_config_with_mcp_profile(self):
        """Test that task configs can reference MCP profiles"""
        config = get_llm_config()
        
        # Agent task should exist
        if "agent" in config.task_configs:
            agent_task = config.task_configs["agent"]
            assert agent_task.primary is not None
            # mcp_profile is optional, so it might be None
            assert hasattr(agent_task, "mcp_profile")

    def test_mcp_env_var_override(self):
        """Test that MCP_ENABLED env var overrides config"""
        # Save original
        original = os.getenv("MCP_ENABLED")
        
        try:
            # Test with env var set to true
            os.environ["MCP_ENABLED"] = "true"
            
            # Re-import to get fresh config
            from inference_core.llm.config import LLMConfig
            config = LLMConfig()
            
            assert config.mcp_config.enabled is True
            
        finally:
            # Restore original
            if original is None:
                os.environ.pop("MCP_ENABLED", None)
            else:
                os.environ["MCP_ENABLED"] = original

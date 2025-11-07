"""
Unit tests for pluggable tool provider system.

Tests the tool provider registry, tool loading, deduplication, and filtering logic.
"""

import pytest
from unittest.mock import MagicMock

from inference_core.llm.tools import (
    ToolProvider,
    register_tool_provider,
    get_registered_providers,
    unregister_tool_provider,
    clear_tool_providers,
    load_tools_for_task,
)


class MockTool:
    """Mock LangChain tool for testing"""

    def __init__(self, name: str, description: str = "A mock tool"):
        self.name = name
        self.description = description


class SimpleToolProvider:
    """Simple synchronous tool provider for testing"""

    def __init__(self, name: str, tools: list):
        self.name = name
        self._tools = tools

    async def get_tools(self, task_type: str, user_context=None):
        return self._tools


class DynamicToolProvider:
    """Tool provider that returns different tools based on context"""

    def __init__(self, name: str):
        self.name = name

    async def get_tools(self, task_type: str, user_context=None):
        if task_type == "chat":
            return [MockTool("chat_tool")]
        elif task_type == "completion":
            return [MockTool("completion_tool")]
        return []


class TestToolProviderRegistry:
    """Test the tool provider registration system"""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Clear registry before and after each test"""
        clear_tool_providers()
        yield
        clear_tool_providers()

    def test_register_provider(self):
        """Test registering a tool provider"""
        provider = SimpleToolProvider("test_provider", [])
        register_tool_provider(provider)

        providers = get_registered_providers()
        assert "test_provider" in providers
        assert providers["test_provider"] is provider

    def test_register_duplicate_provider_replaces(self):
        """Test that registering duplicate name replaces existing provider"""
        provider1 = SimpleToolProvider("test_provider", [MockTool("tool1")])
        provider2 = SimpleToolProvider("test_provider", [MockTool("tool2")])

        register_tool_provider(provider1)
        register_tool_provider(provider2)

        providers = get_registered_providers()
        assert len(providers) == 1
        assert providers["test_provider"] is provider2

    def test_register_provider_without_name_raises(self):
        """Test that provider without 'name' raises ValueError"""

        class BadProvider:
            async def get_tools(self, task_type, user_context=None):
                return []

        with pytest.raises(ValueError, match="must have a 'name' attribute"):
            register_tool_provider(BadProvider())

    def test_register_provider_without_get_tools_raises(self):
        """Test that provider without 'get_tools' raises ValueError"""

        class BadProvider:
            name = "bad_provider"

        with pytest.raises(ValueError, match="must have a 'get_tools' method"):
            register_tool_provider(BadProvider())

    def test_unregister_provider(self):
        """Test unregistering a provider"""
        provider = SimpleToolProvider("test_provider", [])
        register_tool_provider(provider)

        assert "test_provider" in get_registered_providers()

        unregister_tool_provider("test_provider")
        assert "test_provider" not in get_registered_providers()

    def test_unregister_nonexistent_provider_silent(self):
        """Test that unregistering non-existent provider is silent"""
        # Should not raise
        unregister_tool_provider("nonexistent")

    def test_clear_providers(self):
        """Test clearing all providers"""
        provider1 = SimpleToolProvider("provider1", [])
        provider2 = SimpleToolProvider("provider2", [])

        register_tool_provider(provider1)
        register_tool_provider(provider2)

        assert len(get_registered_providers()) == 2

        clear_tool_providers()
        assert len(get_registered_providers()) == 0


class TestLoadToolsForTask:
    """Test loading tools from providers for a task"""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Clear registry before and after each test"""
        clear_tool_providers()
        yield
        clear_tool_providers()

    @pytest.mark.asyncio
    async def test_load_tools_from_single_provider(self):
        """Test loading tools from a single provider"""
        tools = [MockTool("tool1"), MockTool("tool2")]
        provider = SimpleToolProvider("test_provider", tools)
        register_tool_provider(provider)

        loaded = await load_tools_for_task(
            task_type="chat",
            provider_names=["test_provider"],
        )

        assert len(loaded) == 2
        assert loaded[0].name == "tool1"
        assert loaded[1].name == "tool2"

    @pytest.mark.asyncio
    async def test_load_tools_from_multiple_providers(self):
        """Test loading and merging tools from multiple providers"""
        provider1 = SimpleToolProvider("provider1", [MockTool("tool1")])
        provider2 = SimpleToolProvider("provider2", [MockTool("tool2")])

        register_tool_provider(provider1)
        register_tool_provider(provider2)

        loaded = await load_tools_for_task(
            task_type="chat",
            provider_names=["provider1", "provider2"],
        )

        assert len(loaded) == 2
        tool_names = {t.name for t in loaded}
        assert tool_names == {"tool1", "tool2"}

    @pytest.mark.asyncio
    async def test_load_tools_deduplicates_by_name(self):
        """Test that tools with same name are deduplicated (first wins)"""
        provider1 = SimpleToolProvider("provider1", [MockTool("duplicate_tool")])
        provider2 = SimpleToolProvider("provider2", [MockTool("duplicate_tool")])

        register_tool_provider(provider1)
        register_tool_provider(provider2)

        loaded = await load_tools_for_task(
            task_type="chat",
            provider_names=["provider1", "provider2"],
        )

        assert len(loaded) == 1
        assert loaded[0].name == "duplicate_tool"

    @pytest.mark.asyncio
    async def test_load_tools_skips_unregistered_provider(self):
        """Test that unregistered provider names are skipped with warning"""
        provider = SimpleToolProvider("provider1", [MockTool("tool1")])
        register_tool_provider(provider)

        loaded = await load_tools_for_task(
            task_type="chat",
            provider_names=["provider1", "nonexistent_provider"],
        )

        # Should only get tools from provider1
        assert len(loaded) == 1
        assert loaded[0].name == "tool1"

    @pytest.mark.asyncio
    async def test_load_tools_with_allowlist(self):
        """Test that allowlist filters tools correctly"""
        tools = [MockTool("tool1"), MockTool("tool2"), MockTool("tool3")]
        provider = SimpleToolProvider("test_provider", tools)
        register_tool_provider(provider)

        loaded = await load_tools_for_task(
            task_type="chat",
            provider_names=["test_provider"],
            allowed_tools=["tool1", "tool3"],
        )

        assert len(loaded) == 2
        tool_names = {t.name for t in loaded}
        assert tool_names == {"tool1", "tool3"}

    @pytest.mark.asyncio
    async def test_load_tools_with_allowlist_empty_result(self):
        """Test that allowlist can filter out all tools"""
        tools = [MockTool("tool1"), MockTool("tool2")]
        provider = SimpleToolProvider("test_provider", tools)
        register_tool_provider(provider)

        loaded = await load_tools_for_task(
            task_type="chat",
            provider_names=["test_provider"],
            allowed_tools=["tool3"],  # None of the provider's tools
        )

        assert len(loaded) == 0

    @pytest.mark.asyncio
    async def test_load_tools_with_user_context(self):
        """Test that user context is passed to providers"""

        class ContextAwareProvider:
            name = "context_provider"

            async def get_tools(self, task_type, user_context=None):
                if user_context and user_context.get("is_superuser"):
                    return [MockTool("admin_tool")]
                return [MockTool("regular_tool")]

        provider = ContextAwareProvider()
        register_tool_provider(provider)

        # Without superuser
        loaded = await load_tools_for_task(
            task_type="chat",
            provider_names=["context_provider"],
            user_context={"is_superuser": False},
        )
        assert len(loaded) == 1
        assert loaded[0].name == "regular_tool"

        # With superuser
        loaded = await load_tools_for_task(
            task_type="chat",
            provider_names=["context_provider"],
            user_context={"is_superuser": True},
        )
        assert len(loaded) == 1
        assert loaded[0].name == "admin_tool"

    @pytest.mark.asyncio
    async def test_load_tools_dynamic_provider(self):
        """Test provider that returns different tools based on task type"""
        provider = DynamicToolProvider("dynamic")
        register_tool_provider(provider)

        # Chat task
        loaded = await load_tools_for_task(
            task_type="chat",
            provider_names=["dynamic"],
        )
        assert len(loaded) == 1
        assert loaded[0].name == "chat_tool"

        # Completion task
        loaded = await load_tools_for_task(
            task_type="completion",
            provider_names=["dynamic"],
        )
        assert len(loaded) == 1
        assert loaded[0].name == "completion_tool"

    @pytest.mark.asyncio
    async def test_load_tools_skips_tools_without_name(self):
        """Test that tools without 'name' attribute are skipped"""

        class BadTool:
            description = "Tool without name"

        tools = [MockTool("good_tool"), BadTool()]
        provider = SimpleToolProvider("test_provider", tools)
        register_tool_provider(provider)

        loaded = await load_tools_for_task(
            task_type="chat",
            provider_names=["test_provider"],
        )

        assert len(loaded) == 1
        assert loaded[0].name == "good_tool"

    @pytest.mark.asyncio
    async def test_load_tools_handles_provider_exception(self):
        """Test that exceptions from a provider don't stop loading from other providers"""

        class FailingProvider:
            name = "failing_provider"

            async def get_tools(self, task_type, user_context=None):
                raise RuntimeError("Provider failed!")

        provider1 = FailingProvider()
        provider2 = SimpleToolProvider("good_provider", [MockTool("tool1")])

        register_tool_provider(provider1)
        register_tool_provider(provider2)

        loaded = await load_tools_for_task(
            task_type="chat",
            provider_names=["failing_provider", "good_provider"],
        )

        # Should still get tools from good_provider
        assert len(loaded) == 1
        assert loaded[0].name == "tool1"

    @pytest.mark.asyncio
    async def test_load_tools_empty_provider_list(self):
        """Test loading with empty provider list returns empty list"""
        loaded = await load_tools_for_task(
            task_type="chat",
            provider_names=[],
        )
        assert len(loaded) == 0

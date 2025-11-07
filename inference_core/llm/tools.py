"""
Pluggable Tool Provider System for LLM Service

Provides a registry for custom LangChain tool providers that can be attached
to chat/completion tasks without requiring MCP servers.

Usage:
    # In application startup code:
    from inference_core.llm.tools import register_tool_provider

    class MyToolProvider:
        name = "my_tools"
        async def get_tools(self, task_type: str, user_context=None):
            return [my_custom_tool_instance]

    register_tool_provider(MyToolProvider())

    # In llm_config.yaml:
    tasks:
      assistant_converse:
        primary: gpt-5-mini
        local_tool_providers: ['my_tools']
        tool_limits:
          max_steps: 4
          max_run_seconds: 30
        allowed_tools: ['my_custom_tool']  # optional allowlist
"""

import logging
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class ToolProvider(Protocol):
    """Protocol for tool providers.

    Tool providers supply LangChain-compatible tools based on context
    (task type, user session, etc.).

    Attributes:
        name: Unique identifier for this provider
    """

    name: str

    async def get_tools(
        self,
        task_type: str,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Return list of LangChain tools for the given context.

        Args:
            task_type: Task type (e.g., "chat", "completion", "assistant_converse")
            user_context: Optional dict with user metadata (user_id, is_superuser, etc.)

        Returns:
            List of LangChain BaseTool instances
        """
        ...


# Global registry of tool providers
_tool_providers: Dict[str, ToolProvider] = {}


def register_tool_provider(provider: ToolProvider) -> None:
    """Register a tool provider globally.

    Args:
        provider: ToolProvider instance with a unique name

    Raises:
        ValueError: If provider name is already registered
    """
    if not hasattr(provider, "name"):
        raise ValueError("ToolProvider must have a 'name' attribute")

    if not hasattr(provider, "get_tools"):
        raise ValueError("ToolProvider must have a 'get_tools' method")

    name = provider.name
    if name in _tool_providers:
        logger.warning(
            f"Tool provider '{name}' is already registered; replacing with new instance"
        )

    _tool_providers[name] = provider
    logger.info(f"Registered tool provider: {name}")


def get_registered_providers() -> Dict[str, ToolProvider]:
    """Get all registered tool providers.

    Returns:
        Dict mapping provider names to provider instances
    """
    return dict(_tool_providers)


def unregister_tool_provider(name: str) -> None:
    """Unregister a tool provider by name.

    Args:
        name: Provider name to unregister

    Returns:
        None (silently ignores if provider doesn't exist)
    """
    if name in _tool_providers:
        del _tool_providers[name]
        logger.info(f"Unregistered tool provider: {name}")


def clear_tool_providers() -> None:
    """Clear all registered tool providers.

    Primarily useful for testing to reset state between tests.
    """
    _tool_providers.clear()
    logger.debug("Cleared all tool providers")


async def load_tools_for_task(
    task_type: str,
    provider_names: List[str],
    user_context: Optional[Dict[str, Any]] = None,
    allowed_tools: Optional[List[str]] = None,
) -> List[Any]:
    """Load tools from specified providers for a task.

    Args:
        task_type: Task type (e.g., "chat", "completion")
        provider_names: List of provider names to load tools from
        user_context: Optional user context dict
        allowed_tools: Optional allowlist of tool names

    Returns:
        List of LangChain tools, deduplicated by name

    Raises:
        ValueError: If a provider name is not registered
    """
    all_tools = []
    seen_names = set()

    for provider_name in provider_names:
        provider = _tool_providers.get(provider_name)
        if provider is None:
            logger.warning(
                f"Tool provider '{provider_name}' not found (referenced by task '{task_type}'). "
                f"Skipping. Available providers: {list(_tool_providers.keys())}"
            )
            continue

        try:
            tools = await provider.get_tools(task_type, user_context)
            logger.info(
                f"Provider '{provider_name}' returned {len(tools)} tools for task '{task_type}'"
            )

            # Deduplicate tools by name
            for tool in tools:
                tool_name = getattr(tool, "name", None)
                if tool_name is None:
                    logger.warning(
                        f"Tool from provider '{provider_name}' has no 'name' attribute; skipping"
                    )
                    continue

                # Apply allowlist if configured
                if allowed_tools is not None and tool_name not in allowed_tools:
                    logger.debug(
                        f"Tool '{tool_name}' from provider '{provider_name}' "
                        f"not in allowlist; skipping"
                    )
                    continue

                if tool_name in seen_names:
                    logger.debug(
                        f"Tool '{tool_name}' already loaded from another provider; skipping duplicate"
                    )
                    continue

                all_tools.append(tool)
                seen_names.add(tool_name)

        except Exception as exc:
            logger.error(
                f"Error loading tools from provider '{provider_name}': {exc}",
                exc_info=True,
            )
            # Continue with other providers

    logger.info(
        f"Loaded {len(all_tools)} tools for task '{task_type}' "
        f"from providers: {provider_names}"
    )
    return all_tools

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

import inspect
import logging
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


def _filter_kwargs_for_callable(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return a subset of *kwargs* accepted by *fn*'s signature.

    If *fn* declares ``**kwargs`` (VAR_KEYWORD), all kwargs are passed
    through. Otherwise only keys whose name matches a formal parameter are
    forwarded. This keeps backward compatibility with providers whose
    ``get_tools`` signature predates the multimodal/capability kwargs.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


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
        **kwargs,
    ) -> List[Any]:
        """Return list of LangChain tools for the given context.

        Args:
            task_type: Task type (e.g., "chat", "completion", "assistant_converse")
            **kwargs: Additional context-specific parameters

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


def tool_requires_multimodal(tool: Any) -> bool:
    """Return True if *tool* declares a multimodal input requirement.

    Tools opt in by setting the class-level attribute
    ``requires_multimodal = True`` on their :class:`BaseTool` subclass
    (e.g. a screenshot tool that returns image content blocks).
    """
    return bool(getattr(tool, "requires_multimodal", False))


def _apply_capability_filter(
    tool: Any,
    tool_name: str,
    provider_name: str,
    model_multimodal: bool,
    on_missing_capability: str,
) -> bool:
    """Decide whether *tool* should be kept given the active model capability.

    Returns True if the tool should be included in the returned list,
    False if it should be skipped. A ``delegate`` strategy always keeps the
    tool so a higher layer (e.g. :class:`ToolBasedModelSwitchMiddleware`)
    can route its invocation to a support model.
    """
    if not tool_requires_multimodal(tool):
        return True
    if model_multimodal:
        return True
    if on_missing_capability == "delegate":
        logger.info(
            "Tool '%s' from provider '%s' requires multimodal but active "
            "model is not multimodal; keeping for delegation.",
            tool_name,
            provider_name,
        )
        return True
    # skip
    logger.warning(
        "Tool '%s' from provider '%s' requires multimodal but active model "
        "is not multimodal; skipping (on_missing_capability='skip').",
        tool_name,
        provider_name,
    )
    return False


async def load_tools_for_task(
    task_type: str,
    provider_names: List[str],
    allowed_tools: Optional[List[str]] = None,
    *,
    model_multimodal: bool = True,
    on_missing_capability: str = "skip",
    multimodal_support_model: Optional[str] = None,
    **kwargs,
) -> List[Any]:
    """Load tools from specified providers for a task.

    Args:
        task_type: Task type (e.g., "chat", "completion")
        provider_names: List of provider names to load tools from
        allowed_tools: Optional allowlist of tool names
        model_multimodal: Whether the active model supports multimodal inputs.
            When False, tools with ``requires_multimodal = True`` are filtered
            or kept for delegation depending on *on_missing_capability*.
        on_missing_capability: ``"skip"`` to drop unsupported tools (default)
            or ``"delegate"`` to keep them for downstream routing to a
            support model.
        multimodal_support_model: Optional name of the support model used by
            providers that implement an in-tool vision fallback (e.g. the
            Electron browser tools). Forwarded to provider ``get_tools`` as a
            kwarg along with ``model_multimodal`` and ``on_missing_capability``.
        **kwargs: Additional context-specific parameters

    Returns:
        List of LangChain tools, deduplicated by name

    Raises:
        ValueError: If a provider name is not registered
    """
    all_tools = []
    seen_names = set()

    provider_kwargs = dict(kwargs)
    provider_kwargs.setdefault("model_multimodal", model_multimodal)
    provider_kwargs.setdefault("on_missing_capability", on_missing_capability)
    provider_kwargs.setdefault("multimodal_support_model", multimodal_support_model)

    for provider_name in provider_names:
        provider = _tool_providers.get(provider_name)
        if provider is None:
            logger.warning(
                f"Tool provider '{provider_name}' not found (referenced by task '{task_type}'). "
                f"Skipping. Available providers: {list(_tool_providers.keys())}"
            )
            continue

        try:
            call_kwargs = _filter_kwargs_for_callable(
                provider.get_tools, provider_kwargs
            )
            tools = await provider.get_tools(task_type, **call_kwargs)
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

                if not _apply_capability_filter(
                    tool,
                    tool_name,
                    provider_name,
                    model_multimodal,
                    on_missing_capability,
                ):
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


async def load_tools_for_agent(
    agent_name: str,
    provider_names: List[str],
    allowed_tools: Optional[List[str]] = None,
    *,
    model_multimodal: bool = True,
    on_missing_capability: str = "skip",
    multimodal_support_model: Optional[str] = None,
    **kwargs,
) -> List[Any]:
    """Load tools from specified providers for an agent.

    See :func:`load_tools_for_task` for parameter semantics. ``agent_name``
    replaces ``task_type`` and is forwarded to each provider's ``get_tools``.
    """
    all_tools = []
    seen_names = set()

    provider_kwargs = dict(kwargs)
    provider_kwargs.setdefault("model_multimodal", model_multimodal)
    provider_kwargs.setdefault("on_missing_capability", on_missing_capability)
    provider_kwargs.setdefault("multimodal_support_model", multimodal_support_model)

    for provider_name in provider_names:
        provider = _tool_providers.get(provider_name)
        if provider is None:
            logger.warning(
                f"Tool provider '{provider_name}' not found (referenced by task '{agent_name}'). "
                f"Skipping. Available providers: {list(_tool_providers.keys())}"
            )
            continue

        try:
            call_kwargs = _filter_kwargs_for_callable(
                provider.get_tools, provider_kwargs
            )
            tools = await provider.get_tools(agent_name, **call_kwargs)
            logger.info(
                f"Provider '{provider_name}' returned {len(tools)} tools for task '{agent_name}'"
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

                if not _apply_capability_filter(
                    tool,
                    tool_name,
                    provider_name,
                    model_multimodal,
                    on_missing_capability,
                ):
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
        f"Loaded {len(all_tools)} tools for task '{agent_name}' "
        f"from providers: {provider_names}"
    )
    return all_tools

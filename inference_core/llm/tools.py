"""
Pluggable Tool Provider System for AgentService

Provides a registry for custom LangChain tool providers that can be attached
to configured agents without requiring MCP servers.

Usage:
    # In application startup code:
    from inference_core.llm.tools import register_tool_provider

    class MyToolProvider:
        name = "my_tools"
        async def get_tools(self, task_type: str, user_context=None):
            return [my_custom_tool_instance]

    register_tool_provider(MyToolProvider())

    # In llm_config.yaml:
    agents:
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
from typing import Any, Dict, List, Literal, Optional, Protocol

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
    except TypeError, ValueError:
        return kwargs
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


class ToolProvider(Protocol):
    """Protocol for tool providers.

    Tool providers supply LangChain-compatible tools based on context
    (agent name, user session, request id, etc.).

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
            task_type: Context label kept as the first argument for provider
                compatibility. In the agent-only runtime this is the active
                agent name.
            **kwargs: Additional context-specific parameters

        Returns:
            List of LangChain BaseTool instances
        """
        ...


ToolEnvironment = Literal["production", "emulated", "strict_test"]
ToolDoubleStrategy = Literal["replace", "disable", "passthrough"]


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


def tool_has_test_double(tool: Any) -> bool:
    """Return whether a production tool declares an explicit test double.

    WHY: Strict no-cost environments need a cheap predicate that can reject
    tools before the model sees them. Providers may expose doubles either by
    setting ``tool.test_double`` or by implementing ``get_test_tools``.
    """
    return getattr(tool, "test_double", None) is not None


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


async def _load_provider_tools_with_doubles(
    provider: ToolProvider,
    provider_name: str,
    context_name: str,
    provider_kwargs: Dict[str, Any],
    *,
    tool_environment: ToolEnvironment,
) -> tuple[List[Any], Dict[str, Any]]:
    """Load production tools and optional test doubles from a provider.

    WHY: Keeping both calls together gives the resolver enough context to
    replace production tools with explicit doubles while preserving backward
    compatibility for providers that only implement ``get_tools``.
    """
    call_kwargs = _filter_kwargs_for_callable(provider.get_tools, provider_kwargs)
    tools = await provider.get_tools(context_name, **call_kwargs)

    if tool_environment == "production":
        return tools, {}

    get_test_tools = getattr(provider, "get_test_tools", None)
    if not callable(get_test_tools):
        return tools, {}

    test_call_kwargs = _filter_kwargs_for_callable(get_test_tools, provider_kwargs)
    test_tools = await get_test_tools(context_name, **test_call_kwargs)
    return tools, _index_test_tools(test_tools)


def _index_test_tools(test_tools: List[Any]) -> Dict[str, Any]:
    """Build a lookup for provider-level test doubles.

    WHY: Some doubles should keep the production tool name, while others may
    expose ``original_tool_name`` to make their test-only purpose explicit.
    """
    indexed: Dict[str, Any] = {}
    for test_tool in test_tools:
        name = getattr(test_tool, "original_tool_name", None) or getattr(
            test_tool, "name", None
        )
        if name:
            indexed[name] = test_tool
    return indexed


def _resolve_tool_for_environment(
    tool: Any,
    provider_test_tools: Dict[str, Any],
    provider_name: str,
    *,
    tool_environment: ToolEnvironment,
    require_test_doubles: bool,
    tool_double_strategy: ToolDoubleStrategy,
) -> Any | None:
    """Resolve a production tool into the tool exposed to the agent.

    WHY: E2E and security profiles need deterministic tool replacement. This
    function centralizes the policy so local agents and Agent Server tool
    loading can share the same behaviour.
    """
    if tool_environment == "production":
        return tool

    strict_required = tool_environment == "strict_test" or require_test_doubles
    if tool_double_strategy == "passthrough" and not strict_required:
        return tool

    tool_name = getattr(tool, "name", None)
    explicit_double = getattr(tool, "test_double", None)
    if callable(explicit_double) and not hasattr(explicit_double, "name"):
        explicit_double = explicit_double()

    test_double = explicit_double or provider_test_tools.get(tool_name)
    if test_double is not None and tool_double_strategy != "disable":
        return test_double

    if strict_required or tool_double_strategy == "disable":
        logger.warning(
            "Tool '%s' from provider '%s' has no test double; skipping in %s mode.",
            tool_name,
            provider_name,
            tool_environment,
        )
        return None

    return tool


async def _load_tools_from_providers(
    context_name: str,
    provider_names: List[str],
    allowed_tools: Optional[List[str]],
    *,
    model_multimodal: bool,
    on_missing_capability: str,
    multimodal_support_model: Optional[str],
    tool_environment: ToolEnvironment,
    require_test_doubles: bool,
    tool_double_strategy: ToolDoubleStrategy,
    log_context: str,
    **kwargs,
) -> List[Any]:
    """Shared implementation for loading tools for tasks and agents.

    WHY: Agent and legacy task loading must apply allowlists, deduplication,
    capability filtering, and test-double replacement in the same order.
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
                "Tool provider '%s' not found (referenced by %s '%s'). "
                "Skipping. Available providers: %s",
                provider_name,
                log_context,
                context_name,
                list(_tool_providers.keys()),
            )
            continue

        try:
            tools, provider_test_tools = await _load_provider_tools_with_doubles(
                provider,
                provider_name,
                context_name,
                provider_kwargs,
                tool_environment=tool_environment,
            )
            logger.info(
                "Provider '%s' returned %d tools for %s '%s'",
                provider_name,
                len(tools),
                log_context,
                context_name,
            )

            for tool in tools:
                tool_name = getattr(tool, "name", None)
                if tool_name is None:
                    logger.warning(
                        "Tool from provider '%s' has no 'name' attribute; skipping",
                        provider_name,
                    )
                    continue

                resolved_tool = _resolve_tool_for_environment(
                    tool,
                    provider_test_tools,
                    provider_name,
                    tool_environment=tool_environment,
                    require_test_doubles=require_test_doubles,
                    tool_double_strategy=tool_double_strategy,
                )
                if resolved_tool is None:
                    continue

                resolved_name = getattr(resolved_tool, "name", tool_name)

                if allowed_tools is not None and resolved_name not in allowed_tools:
                    logger.debug(
                        "Tool '%s' from provider '%s' not in allowlist; skipping",
                        resolved_name,
                        provider_name,
                    )
                    continue

                if resolved_name in seen_names:
                    logger.debug(
                        "Tool '%s' already loaded from another provider; skipping duplicate",
                        resolved_name,
                    )
                    continue

                if not _apply_capability_filter(
                    resolved_tool,
                    resolved_name,
                    provider_name,
                    model_multimodal,
                    on_missing_capability,
                ):
                    continue

                all_tools.append(resolved_tool)
                seen_names.add(resolved_name)

        except Exception as exc:
            logger.error(
                "Error loading tools from provider '%s': %s",
                provider_name,
                exc,
                exc_info=True,
            )

    logger.info(
        "Loaded %d tools for %s '%s' from providers: %s",
        len(all_tools),
        log_context,
        context_name,
        provider_names,
    )
    return all_tools


async def load_tools_for_task(
    task_type: str,
    provider_names: List[str],
    allowed_tools: Optional[List[str]] = None,
    *,
    model_multimodal: bool = True,
    on_missing_capability: str = "skip",
    multimodal_support_model: Optional[str] = None,
    tool_environment: ToolEnvironment = "production",
    require_test_doubles: bool = False,
    tool_double_strategy: ToolDoubleStrategy = "replace",
    **kwargs,
) -> List[Any]:
    """Legacy compatibility helper for context-based tool loading.

    WHY: Some providers still expose ``get_tools(task_type, ...)``. The
    agent runtime should prefer :func:`load_tools_for_agent`, which forwards
    the active agent name through the same first positional argument.

    Args:
        task_type: Legacy context label forwarded to provider ``get_tools``
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
    return await _load_tools_from_providers(
        task_type,
        provider_names,
        allowed_tools,
        model_multimodal=model_multimodal,
        on_missing_capability=on_missing_capability,
        multimodal_support_model=multimodal_support_model,
        tool_environment=tool_environment,
        require_test_doubles=require_test_doubles,
        tool_double_strategy=tool_double_strategy,
        log_context="task",
        **kwargs,
    )


async def load_tools_for_agent(
    agent_name: str,
    provider_names: List[str],
    allowed_tools: Optional[List[str]] = None,
    *,
    model_multimodal: bool = True,
    on_missing_capability: str = "skip",
    multimodal_support_model: Optional[str] = None,
    tool_environment: ToolEnvironment = "production",
    require_test_doubles: bool = False,
    tool_double_strategy: ToolDoubleStrategy = "replace",
    **kwargs,
) -> List[Any]:
    """Load tools from specified providers for an agent.

    ``agent_name`` is forwarded as the first positional argument to each
    provider's ``get_tools`` method for backward compatibility.
    """
    return await _load_tools_from_providers(
        agent_name,
        provider_names,
        allowed_tools,
        model_multimodal=model_multimodal,
        on_missing_capability=on_missing_capability,
        multimodal_support_model=multimodal_support_model,
        tool_environment=tool_environment,
        require_test_doubles=require_test_doubles,
        tool_double_strategy=tool_double_strategy,
        log_context="agent",
        **kwargs,
    )

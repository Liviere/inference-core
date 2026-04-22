"""
Declarative registry for Agent Server graphs and tool providers.

WHY: Moves Agent Server bootstrap logic out of ``agent_graphs.py`` so that
provider registration and graph selection are driven entirely by
``llm_config.yaml``. The entry point becomes a thin loader that calls the
two helpers in this module.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List

from inference_core.agents.graph_builder import build_agent_graph
from inference_core.llm.config import AgentConfig, LLMConfig, ToolProviderEntry
from inference_core.llm.tools import ToolProvider, register_tool_provider

logger = logging.getLogger(__name__)


def _import_class(class_path: str) -> type:
    """Resolve ``'module.path:ClassName'`` to an actual class object.

    WHY: ``ToolProviderEntry.class_path`` uses the colon-separated form to
    unambiguously split module vs. attribute. Format validation already
    happened in the Pydantic validator; here we only do the import + getattr.
    """
    module_path, attr_name = class_path.split(":", 1)
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(
            f"Module '{module_path}' has no attribute '{attr_name}' "
            f"(resolving tool provider class_path={class_path!r})"
        ) from exc


def register_providers_from_config(config: LLMConfig) -> List[str]:
    """Instantiate and register every enabled entry in ``tool_providers``.

    WHY: The Agent Server process needs providers registered before
    ``build_agent_graph()`` runs, otherwise ``local_tool_providers`` entries
    in each agent's YAML resolve to an empty set. This helper performs that
    step in a single pass, enforcing that the logical key in YAML matches
    the ``name`` attribute of the provider instance so typos are caught
    early.

    Args:
        config: Loaded :class:`LLMConfig` (usually from ``get_llm_config()``).

    Returns:
        List of logical provider names that were successfully registered.
        Entries with ``enabled=False``, import errors, or name mismatches
        are skipped with an error log.
    """
    registered: List[str] = []
    provider_configs: Dict[str, ToolProviderEntry] = config.tool_provider_configs or {}

    if not provider_configs:
        logger.info("No tool_providers section found in llm_config.yaml — skipping provider registration")
        return registered

    for logical_name, entry in provider_configs.items():
        if not entry.enabled:
            logger.info("Tool provider '%s' disabled via config — skipping", logical_name)
            continue

        try:
            cls = _import_class(entry.class_path)
        except Exception:
            logger.exception(
                "Failed to import tool provider '%s' (class_path=%s)",
                logical_name,
                entry.class_path,
            )
            continue

        try:
            instance = cls(**(entry.kwargs or {}))
        except Exception:
            logger.exception(
                "Failed to instantiate tool provider '%s' (class=%s, kwargs=%s)",
                logical_name,
                entry.class_path,
                entry.kwargs,
            )
            continue

        # Structural check — ToolProvider is a Protocol without
        # @runtime_checkable, so duck-type verify the two required members.
        if not hasattr(instance, "name") or not callable(
            getattr(instance, "get_tools", None)
        ):
            logger.error(
                "Tool provider '%s' (%s) is missing required members "
                "(name, get_tools); skipping",
                logical_name,
                entry.class_path,
            )
            continue

        # Enforce YAML key == provider.name so the agent's local_tool_providers
        # list (which references the key) unambiguously maps to this instance.
        instance_name = getattr(instance, "name", None)
        if instance_name != logical_name:
            logger.error(
                "Tool provider key mismatch for '%s': instance.name=%r "
                "(class %s). YAML key must equal ToolProvider.name.",
                logical_name,
                instance_name,
                entry.class_path,
            )
            continue

        register_tool_provider(instance)
        registered.append(logical_name)

    logger.info("Registered %d tool providers from YAML: %s", len(registered), registered)
    return registered


def _resolve_use_memory(agent_config: AgentConfig) -> bool:
    """Decide whether this agent's graph should be compiled with memory.

    WHY: The previous hand-written ``agent_graphs.py`` hardcoded ``use_memory=True``
    for ``default_agent`` and ``deep_planner``. That decision now lives in YAML.

    Resolution order:
      1. Explicit ``agent_config.use_memory`` (True or False) wins.
      2. Auto-detect: enable when the agent has any memory-related YAML hints
         (``memory_tools`` configured, ``memory_session_context_enabled``,
         ``memory_tool_instructions_enabled``) AND the global
         ``AGENT_MEMORY_ENABLED`` setting is True.
    """
    if agent_config.use_memory is not None:
        return agent_config.use_memory

    from inference_core.core.config import get_settings

    settings = get_settings()
    if not settings.agent_memory_enabled:
        return False

    has_memory_hints = (
        agent_config.memory_tools is not None
        or agent_config.memory_session_context_enabled is not None
        or agent_config.memory_tool_instructions_enabled is not None
    )
    return has_memory_hints


def _should_build_server_graph(agent_config: AgentConfig) -> bool:
    """Decide whether the Agent Server should expose a compiled graph.

    WHY: Keeps the per-agent choice in YAML. Default rule (``server_graph=None``)
    is to build iff the agent is marked ``execution_mode: remote`` — the
    natural case for the LangGraph Agent Server entry point. Explicit
    ``server_graph: true/false`` overrides the auto rule for edge cases
    (e.g., exposing a local-mode agent for Studio debugging).
    """
    if agent_config.server_graph is not None:
        return agent_config.server_graph
    return agent_config.execution_mode == "remote"


def build_server_graphs(config: LLMConfig) -> Dict[str, Any]:
    """Build all Agent Server graphs selected by YAML policy.

    WHY: Replaces the hand-written list of ``build_agent_graph(...)`` calls
    in ``agent_graphs.py``. Iterates every agent in ``config.agent_configs``
    and compiles a graph for those matched by :func:`_should_build_server_graph`.

    Args:
        config: Loaded :class:`LLMConfig`.

    Returns:
        Mapping ``{agent_name: compiled_graph}`` ready to be exposed as
        module-level attributes for the LangGraph CLI.
    """
    graphs: Dict[str, Any] = {}

    for agent_name, agent_config in (config.agent_configs or {}).items():
        if not _should_build_server_graph(agent_config):
            logger.debug(
                "Skipping agent '%s' — server_graph=%s, execution_mode=%s",
                agent_name,
                agent_config.server_graph,
                agent_config.execution_mode,
            )
            continue

        use_memory = _resolve_use_memory(agent_config)
        try:
            graphs[agent_name] = build_agent_graph(agent_name, use_memory=use_memory)
            logger.info(
                "Built Agent Server graph '%s' (use_memory=%s)", agent_name, use_memory
            )
        except Exception:
            logger.exception("Failed to build Agent Server graph for '%s'", agent_name)

    logger.info("Loaded %d Agent Server graphs: %s", len(graphs), sorted(graphs.keys()))
    return graphs


__all__ = [
    "register_providers_from_config",
    "build_server_graphs",
]

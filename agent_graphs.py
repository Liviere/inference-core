"""
Agent Server graph definitions — entry point for ``langgraph dev`` / ``langgraph up``.

WHY: The LangGraph CLI reads ``langgraph.json`` which points to this module.
Each top-level variable is a compiled StateGraph served by the Agent Server.

This file is now a thin, fully YAML-driven loader:

  1. ``tool_providers:`` in ``llm_config.yaml`` declares which provider
     classes to import and register (``class_path: 'module:ClassName'``).
  2. Each agent entry selects whether a server graph is built via
     ``server_graph`` (explicit) or via the auto rule based on
     ``execution_mode: remote`` (default).
  3. Memory compilation is controlled by ``use_memory`` on the agent
     (explicit) or auto-detected from its memory_* hints.

The compiled graphs are injected into the module globals so LangGraph CLI
can resolve ``./agent_graphs.py:<agent_name>`` — make sure the graph names
in ``langgraph.json`` match the agent keys in YAML (``scripts/sync_langgraph_json.py``
regenerates the JSON from YAML when configuration changes).

IMPORTANT: This module is loaded directly by the Agent Server process —
it is NOT imported by the FastAPI application or Celery workers.
"""

import logging

from inference_core.agents.graph_registry import (
    build_server_graphs,
    register_providers_from_config,
)
from inference_core.llm.config import get_llm_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Load YAML-driven configuration.
# 2. Register every enabled ``tool_providers`` entry BEFORE building graphs —
#    ``_load_provider_tools`` in graph_builder only resolves providers that
#    are already registered.
# 3. Build all agent graphs selected by ``server_graph`` / ``execution_mode``
#    and expose them as module-level attributes for the LangGraph CLI.
# ---------------------------------------------------------------------------

_config = get_llm_config()

register_providers_from_config(_config)

_graphs = build_server_graphs(_config)

# Expose each compiled graph as a module attribute so ``langgraph.json``
# entries like ``./agent_graphs.py:default_agent`` can resolve them.
globals().update(_graphs)

logger.info(
    "Agent Server graphs loaded from YAML: %s", sorted(_graphs.keys())
)

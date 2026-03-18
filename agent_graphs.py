"""
Agent Server graph definitions — entry point for ``langgraph dev`` / ``langgraph up``.

WHY: The LangGraph CLI reads ``langgraph.json`` which points to this module.
Each top-level variable is a compiled StateGraph served by the Agent Server.

CUSTOMISATION:
  Register your own ToolProviders BEFORE the graphs are built.  E.g.:

      from my_app.tools import WeatherToolProvider
      from inference_core.llm.tools import register_tool_provider
      register_tool_provider(WeatherToolProvider())

  Then rebuild: ``langgraph dev``

IMPORTANT: This module is loaded directly by the Agent Server process —
it is NOT imported by the FastAPI application or Celery workers.
"""

import logging

from inference_core.agents.graph_builder import build_agent_graph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Register application-specific tool providers here.
#
# Example:
#   from my_app.tools import MyToolProvider
#   from inference_core.llm.tools import register_tool_provider
#   register_tool_provider(MyToolProvider())
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Build graphs for each agent defined in llm_config.yaml.
# The variable names MUST match the keys in langgraph.json → "graphs".
# ---------------------------------------------------------------------------

default_agent = build_agent_graph("default_agent", use_memory=True)
weather_agent = build_agent_graph("weather_agent")
deep_planner = build_agent_graph("deep_planner", use_memory=True)

logger.info("Agent Server graphs loaded: default_agent, weather_agent, deep_planner")

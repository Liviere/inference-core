"""
Graph builder for the LangGraph Agent Server.

WHY: The Agent Server (langgraph dev / langgraph up) requires pre-compiled
StateGraph objects at module level.  This module bridges inference-core's
AgentService config system with that requirement — it reads llm_config.yaml,
builds the correct model + tool set, and returns a compiled graph ready for
the server to serve.

The Agent Server handles its own checkpointing (PostgreSQL / in-memory),
so we intentionally omit checkpointer setup here.
"""

import asyncio
import logging
from typing import Any, Optional

from langchain.agents import create_agent

from inference_core.llm.config import get_llm_config
from inference_core.llm.models import get_model_factory
from inference_core.llm.tools import get_registered_providers, load_tools_for_agent

logger = logging.getLogger(__name__)


def build_agent_graph(
    agent_name: str,
    *,
    extra_tools: Optional[list[Any]] = None,
    system_prompt: Optional[str] = None,
) -> Any:
    """Build a compiled LangGraph agent from YAML config.

    WHY: Provides a single-call interface for the Agent Server entry point
    to create graphs identically to how AgentService does it locally, but
    without per-request runtime state (user_id, session, cost middleware).

    Args:
        agent_name: Agent key in llm_config.yaml ``agents:`` section.
        extra_tools: Additional tools beyond what providers supply.
        system_prompt: Override the default system prompt.

    Returns:
        A compiled LangGraph StateGraph (``CompiledStateGraph``).
    """
    factory = get_model_factory()
    agent_config = factory.config.get_specific_agent_config(agent_name)
    model = factory.get_model_for_agent(agent_name)

    # Load tools from registered providers (async → run in a fresh loop)
    tools = list(extra_tools or [])
    configured_providers = agent_config.local_tool_providers or []

    if configured_providers:
        registered = get_registered_providers()
        matched = [p for p in configured_providers if p in registered]
        if matched:
            try:
                loop = asyncio.new_event_loop()
                provider_tools = loop.run_until_complete(
                    load_tools_for_agent(
                        agent_name,
                        matched,
                        allowed_tools=agent_config.allowed_tools,
                    )
                )
                tools.extend(provider_tools)
                loop.close()
            except Exception:
                logger.exception(
                    "Failed to load provider tools for agent '%s'", agent_name
                )

    effective_prompt = system_prompt or agent_config.description

    graph = create_agent(
        model,
        tools=tools,
        system_prompt=effective_prompt,
    )

    logger.info(
        "Built Agent Server graph for '%s' (model=%s, tools=%d)",
        agent_name,
        factory.get_agent_model_name(agent_name),
        len(tools),
    )
    return graph

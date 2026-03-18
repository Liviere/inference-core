"""
Graph builder for the LangGraph Agent Server.

WHY: The Agent Server (langgraph dev / langgraph up) requires pre-compiled
StateGraph objects at module level.  This module bridges inference-core's
AgentService config system with that requirement — it reads llm_config.yaml,
builds the correct model + tool set, and returns a compiled graph ready for
the server to serve.

The Agent Server handles its own checkpointing (PostgreSQL / in-memory),
so we intentionally omit checkpointer setup here.

Middleware (CostTracking, ToolModelSwitch) is compiled into the graph so
that Agent Server executions have the same observability and model-switching
behaviour as local runs.  Per-request context (user_id, session_id) is
resolved at runtime via ``runtime.configurable`` and ``contextvars``.
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
    without per-request runtime state (user_id, session, etc.).  Per-request
    context is resolved later via ``runtime.configurable`` + contextvars.

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

    # Build middleware for Agent Server context
    middleware = _build_server_middleware(agent_name, agent_config, factory)

    graph = create_agent(
        model,
        tools=tools,
        middleware=middleware or None,
        system_prompt=effective_prompt,
    )

    logger.info(
        "Built Agent Server graph for '%s' (model=%s, tools=%d, middleware=%d)",
        agent_name,
        factory.get_agent_model_name(agent_name),
        len(tools),
        len(middleware),
    )
    return graph


def _build_server_middleware(
    agent_name: str,
    agent_config: Any,
    factory: Any,
) -> list[Any]:
    """Build middleware list for an Agent Server graph.

    WHY: Mirrors the middleware stack that AgentService._build_middleware()
    produces for local execution, but without per-request state.  User-level
    context (user_id, session_id) is resolved at runtime via contextvars
    populated from ``runtime.configurable``.

    Includes:
        - CostTrackingMiddleware (always): token/cost tracking per model step
        - ToolBasedModelSwitchMiddleware (if configured): model switching per tool

    MemoryMiddleware is NOT included here — it requires a Store instance
    with embedding setup and per-request user_id.  Will be added in a
    follow-up phase when the Agent Server supports store injection.
    """
    from inference_core.agents.middleware.cost_tracking import CostTrackingMiddleware
    from inference_core.agents.middleware.tool_model_switch import (
        create_tool_model_switch_middleware,
    )

    middleware: list[Any] = []
    model_name = factory.get_agent_model_name(agent_name)

    # --- CostTrackingMiddleware (user_id=None → resolved from runtime) ---
    pricing_config = None
    provider = None
    try:
        llm_config = get_llm_config()
        model_cfg = llm_config.models.get(model_name)
        if model_cfg:
            pricing_config = model_cfg.pricing
            provider = model_cfg.provider
    except Exception as e:
        logger.debug("Could not load pricing config for '%s': %s", model_name, e)

    cost_middleware = CostTrackingMiddleware(
        pricing_config=pricing_config,
        user_id=None,
        session_id=None,
        request_id=None,
        task_type="agent",
        request_mode="sync",
        provider=provider,
        model_name=model_name,
    )
    middleware.append(cost_middleware)

    # --- ToolBasedModelSwitchMiddleware (if overrides configured) ---
    overrides = agent_config.tool_model_overrides
    if overrides:
        try:
            override_dicts = [
                {
                    "tool_name": o.tool_name,
                    "model": o.model,
                    "trigger": o.trigger,
                    "description": o.description,
                }
                for o in overrides
            ]
            tool_middleware = create_tool_model_switch_middleware(
                overrides=override_dicts,
                default_model=model_name,
                model_factory=factory,
                cache_models=True,
            )
            middleware.append(tool_middleware)
        except Exception as e:
            logger.error(
                "Failed to build ToolBasedModelSwitchMiddleware for '%s': %s",
                agent_name,
                e,
                exc_info=True,
            )

    return middleware

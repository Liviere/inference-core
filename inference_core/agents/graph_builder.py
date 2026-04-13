"""
Graph builder for the LangGraph Agent Server.

WHY: The Agent Server (langgraph dev / langgraph up) requires pre-compiled
StateGraph objects at module level.  This module bridges inference-core's
AgentService config system with that requirement — it reads llm_config.yaml,
builds the correct model + tool set, and returns a compiled graph ready for
the server to serve.

The Agent Server handles its own checkpointing (PostgreSQL / in-memory),
so we intentionally omit checkpointer setup here.

Middleware (CostTracking, ToolModelSwitch, MemoryMiddleware, SubAgentMiddleware,
SkillsMiddleware) is compiled into the graph so that Agent Server executions have
the same observability, model-switching, memory support, sub-agent orchestration,
and skills behaviour as local runs.  Per-request context (user_id, session_id) is
resolved at runtime via ``runtime.configurable`` and ``contextvars``.
"""

import asyncio
import logging
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from langchain.agents import create_agent

from inference_core.llm.config import get_llm_config
from inference_core.llm.models import LLMModelFactory, get_model_factory
from inference_core.llm.tools import get_registered_providers, load_tools_for_agent

logger = logging.getLogger(__name__)

# Base directory for resolving relative skill file paths in YAML config.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Module-level singleton for the memory store (created once at first
# build_agent_graph(use_memory=True) call).  Shared across all graphs.
_memory_store: Any = None


def _init_memory_store() -> Any:
    """Create and cache a module-level memory store singleton.

    WHY: The store requires a DB connection + embedding function.  Both are
    expensive to set up so we reuse a single instance across all agent
    graphs compiled in this process.  Mirrors the pattern in
    ``AgentService._get_memory_store()``.
    """
    global _memory_store
    if _memory_store is not None:
        return _memory_store

    from inference_core.core.config import get_settings
    from inference_core.services.embedding_service import get_embedding_service

    settings = get_settings()
    url = settings.database_url
    # Map async drivers to sync counterparts (store backends use sync conn)
    for async_drv, sync_drv in [
        ("+aiosqlite", ""),
        ("+asyncpg", ""),
        ("+aiomysql", ""),
    ]:
        if async_drv in url:
            url = url.replace(async_drv, sync_drv)
            break

    svc = get_embedding_service()
    embed_fn = svc.get_embed_fn()
    dims = svc.get_dimension()
    index_cfg = {"embed": embed_fn, "dims": dims}

    if "sqlite" in url:
        from langgraph.store.sqlite import SqliteStore

        _memory_store = SqliteStore.from_conn_string(url, index=index_cfg)
    elif "postgresql" in url:
        from langgraph.store.postgres import PostgresStore

        _memory_store = PostgresStore.from_conn_string(url, index=index_cfg)
    else:
        from langgraph.store.memory import InMemoryStore

        logger.warning(
            "Unknown DB dialect for memory store (%s), using InMemoryStore", url
        )
        _memory_store = InMemoryStore(index=index_cfg)

    logger.info(
        "Initialized Agent Server memory store: %s", type(_memory_store).__name__
    )
    return _memory_store


def _init_memory_service(store: Any, agent_name: str) -> Any:
    """Create an AgentMemoryStoreService for *agent_name*.

    Each agent gets its own service instance so that episodic/procedural
    memories are scoped per-agent via ``MemoryNamespaceBuilder``.
    """
    from inference_core.core.config import get_settings
    from inference_core.services.agent_memory_service import AgentMemoryStoreService

    settings = get_settings()
    return AgentMemoryStoreService(
        store=store,
        base_namespace=(settings.agent_memory_collection,),
        max_results=settings.agent_memory_max_results,
        agent_name=agent_name,
    )


def build_agent_graph(
    agent_name: str,
    *,
    extra_tools: Optional[list[Any]] = None,
    system_prompt: Optional[str] = None,
    use_memory: bool = False,
) -> Any:
    """Build a compiled LangGraph agent from YAML config.

    WHY: Provides a single-call interface for the Agent Server entry point
    to create graphs identically to how AgentService does it locally, but
    without per-request runtime state (user_id, session, etc.).  Per-request
    context is resolved later via ``runtime.configurable`` + contextvars.

    Automatically detects deep agents (those with ``subagents`` or ``skills``
    in YAML config) and compiles SubAgentMiddleware / SkillsMiddleware into
    the graph, mirroring the local DeepAgentService behaviour.

    Args:
        agent_name: Agent key in llm_config.yaml ``agents:`` section.
        extra_tools: Additional tools beyond what providers supply.
        system_prompt: Override the default system prompt.
        use_memory: When ``True``, initialise the memory store singleton,
            create an ``AgentMemoryStoreService`` for this agent, add
            ``MemoryMiddleware`` (with ``user_id=None`` — resolved at
            runtime) and memory tools to the graph.

    Returns:
        A compiled LangGraph StateGraph (``CompiledStateGraph``).
    """
    factory = get_model_factory()
    agent_config = factory.config.get_specific_agent_config(agent_name)
    model = factory.get_model_for_agent(agent_name)

    # Resolve whether this agent requests reasoning output — used by
    # InstanceConfigMiddleware when swapping the model at runtime so that
    # the replacement model also receives reasoning_config kwargs.
    agent_reasoning_output = getattr(agent_config, "reasoning_output", False)

    # Load tools from registered providers (async → run in a fresh loop)
    tools = list(extra_tools or [])
    _load_provider_tools(agent_name, agent_config, tools)

    # --- Memory support (store + middleware + tools) ---
    memory_service = None
    if use_memory:
        try:
            store = _init_memory_store()
            memory_service = _init_memory_service(store, agent_name)
            _add_memory_tools(
                memory_service,
                tools,
                include_tools=agent_config.memory_tools,
            )
        except Exception as exc:
            logger.error(
                "Failed to initialise memory for agent '%s': %s",
                agent_name,
                exc,
                exc_info=True,
            )
            memory_service = None

    effective_prompt = system_prompt or agent_config.description

    # On Agent Server, memory tool instructions are injected per-request by
    # MemoryMiddleware.wrap_model_call via the contextvar override, not baked
    # into the compiled prompt.  This allows per-instance/runtime toggling.
    # The compile-time AgentConfig.memory_tool_instructions_enabled value is
    # stored on the middleware as a fallback default.

    # Tool-call limit instructions: baked into compile-time prompt because
    # the policy is static per agent config (no per-request overrides needed).
    from inference_core.llm.config import ToolCallLimitsConfig

    _tcl = getattr(agent_config, "tool_call_limits", None)
    if isinstance(_tcl, ToolCallLimitsConfig):
        from inference_core.agents.middleware.tool_call_limits import (
            generate_tool_call_limits_instructions,
        )

        _tcl_instructions = generate_tool_call_limits_instructions(_tcl)
        if _tcl_instructions:
            if effective_prompt:
                effective_prompt = f"{effective_prompt}\n\n{_tcl_instructions}"
            else:
                effective_prompt = _tcl_instructions

    # Build middleware for Agent Server context
    middleware = _build_server_middleware(
        agent_name,
        agent_config,
        factory,
        memory_service=memory_service,
        reasoning_output=agent_reasoning_output,
        memory_tools_config=agent_config.memory_tools,
        memory_tool_instructions_default=(
            agent_config.memory_tool_instructions_enabled
        ),
    )

    # Deep-agent support: compile SubAgentMiddleware and/or SkillsMiddleware
    # into the graph when the YAML config declares subagents or skills.
    _build_deep_agent_middleware(agent_config, factory, middleware)

    graph = create_agent(
        model,
        tools=tools,
        middleware=middleware or None,
        system_prompt=effective_prompt,
    )

    logger.info(
        "Built Agent Server graph for '%s' (model=%s, tools=%d, middleware=%d: [%s])",
        agent_name,
        factory.get_agent_model_name(agent_name),
        len(tools),
        len(middleware),
        ", ".join(type(m).__name__ for m in middleware),
    )
    return graph


def _build_server_middleware(
    agent_name: str,
    agent_config: Any,
    factory: Any,
    *,
    include_instance_config: bool = True,
    memory_service: Any = None,
    reasoning_output: bool = False,
    memory_tools_config: list[str] | None = None,
    memory_tool_instructions_default: bool | None = None,
) -> list[Any]:
    """Build middleware list for an Agent Server graph.

    WHY: Mirrors the middleware stack that AgentService._build_middleware()
    produces for local execution, but without per-request state.  User-level
    context (user_id, session_id) is resolved at runtime via contextvars
    populated from ``runtime.configurable``.

    Includes:
        - InstanceConfigMiddleware (top-level graphs only): per-user model / prompt
          overrides.  Excluded from subagent graphs because LangGraph propagates
          the parent's ``configurable`` into subgraphs — if the subagent also ran
          InstanceConfigMiddleware, it would read the *parent's* overrides
          (e.g. primary_model) and erroneously swap its own model.
        - MemoryMiddleware (when *memory_service* is provided): CoALA memory
          context injection.  ``user_id=None`` — resolved at runtime from
          configurable via context vars.  Inserted before CostTracking so
          memory enrichment runs first.
        - CostTrackingMiddleware (always): token/cost tracking per model step
        - ToolBasedModelSwitchMiddleware (if configured): model switching per tool

    Args:
        include_instance_config: Whether to include InstanceConfigMiddleware.
            Must be ``False`` for subagent graphs to prevent configurable
            leak from the parent graph.
        memory_service: Optional ``AgentMemoryStoreService`` instance.
            When provided, ``MemoryMiddleware`` is added to the stack.
        reasoning_output: Whether the agent has reasoning output enabled.
            Forwarded to ``InstanceConfigMiddleware`` via
            ``configurable["reasoning_output"]`` and passed directly to
            ``ToolBasedModelSwitchMiddleware`` so that switched models are
            also created with reasoning/thinking output enabled.
    """
    from inference_core.agents.middleware.cost_tracking import CostTrackingMiddleware
    from inference_core.agents.middleware.instance_config import (
        InstanceConfigMiddleware,
    )
    from inference_core.agents.middleware.tool_model_switch import (
        create_tool_model_switch_middleware,
    )

    middleware: list[Any] = []
    model_name = factory.get_agent_model_name(agent_name)

    # --- InstanceConfigMiddleware (model/prompt override from DB instance) ---
    # Excluded from subagent graphs: LangGraph propagates the parent's
    # RunnableConfig.configurable into subgraph invocations, so the subagent
    # would read the parent's primary_model / system_prompt_* and override
    # its own config — making it look like "the agent calls itself".
    if include_instance_config:
        instance_middleware = InstanceConfigMiddleware(model_factory=factory)
        middleware.append(instance_middleware)
    else:
        logger.debug(
            "Skipping InstanceConfigMiddleware for subagent '%s' "
            "(would inherit parent's configurable)",
            agent_name,
        )

    # --- MemoryMiddleware (user_id=None → resolved from configurable) ---
    if memory_service is not None:
        from inference_core.agents.middleware.memory import MemoryMiddleware
        from inference_core.core.config import get_settings as _get_settings

        _mem_settings = _get_settings()

        # Per-agent session context override: AgentConfig.memory_session_context_enabled
        # takes precedence over global auto_recall when explicitly set.
        _auto_recall = _mem_settings.agent_memory_auto_recall
        if agent_config.memory_session_context_enabled is not None:
            _auto_recall = agent_config.memory_session_context_enabled

        mem_middleware = MemoryMiddleware(
            memory_service=memory_service,
            user_id=None,
            auto_recall=_auto_recall,
            max_recall_results=_mem_settings.agent_memory_max_results,
            postrun_analysis=_mem_settings.agent_memory_postrun_analysis_enabled,
            postrun_analysis_model=_mem_settings.agent_memory_postrun_analysis_model,
            tool_instructions_enabled=memory_tool_instructions_default,
            active_tool_names=memory_tools_config,
        )
        middleware.append(mem_middleware)

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
                reasoning_output=reasoning_output,
            )
            middleware.append(tool_middleware)
        except Exception as e:
            logger.error(
                "Failed to build ToolBasedModelSwitchMiddleware for '%s': %s",
                agent_name,
                e,
                exc_info=True,
            )

    # --- ToolCallLimitMiddleware (if configured) ---
    from inference_core.llm.config import ToolCallLimitsConfig

    tool_call_limits = getattr(agent_config, "tool_call_limits", None)
    if isinstance(tool_call_limits, ToolCallLimitsConfig):
        from inference_core.agents.middleware.tool_call_limits import (
            build_tool_call_limit_middleware,
        )

        limiters = build_tool_call_limit_middleware(tool_call_limits)
        # Insert before CostTracking (which must stay last in the list).
        ct_index = next(
            (
                i
                for i, m in enumerate(middleware)
                if isinstance(m, CostTrackingMiddleware)
            ),
            len(middleware),
        )
        for j, limiter in enumerate(limiters):
            middleware.insert(ct_index + j, limiter)

    return middleware


# ------------------------------------------------------------------
# Tool loading
# ------------------------------------------------------------------


def _load_provider_tools(
    agent_name: str,
    agent_config: Any,
    tools: list[Any],
) -> None:
    """Load tools from registered providers into *tools* (mutated in place).

    WHY: Extracted from build_agent_graph so subagent compilation can
    reuse the same async-loop-based loading logic.
    """
    configured_providers = agent_config.local_tool_providers or []
    if not configured_providers:
        return

    registered = get_registered_providers()
    matched = [p for p in configured_providers if p in registered]
    if not matched:
        return

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
        logger.exception("Failed to load provider tools for agent '%s'", agent_name)


def _add_memory_tools(
    memory_service: Any,
    tools: list[Any],
    include_tools: list[str] | None = None,
) -> None:
    """Add memory CRUD tools to *tools* (mutated in place).

    WHY: ``get_memory_tools`` creates Save / Recall / Update / Delete tools
    wired to the given service.  ``user_id=None`` means each tool will
    resolve user_id from the ``_runtime_context`` context var at call time.

    When ``include_tools`` is provided, only the named tools are added.
    An empty list means no model-facing memory tools — the middleware can
    still run for after_agent postrun analysis.
    """
    from inference_core.agents.tools.memory_tools import get_memory_tools
    from inference_core.core.config import get_settings

    settings = get_settings()
    mem_tools = get_memory_tools(
        memory_service=memory_service,
        user_id=None,
        session_id=None,
        max_recall_results=settings.agent_memory_max_results,
        include_tools=include_tools,
    )
    tools.extend(mem_tools)
    logger.debug(
        "Added %d memory tools to tool list (filter=%s)", len(mem_tools), include_tools
    )


# ------------------------------------------------------------------
# Deep-agent support (SubAgentMiddleware + SkillsMiddleware)
# ------------------------------------------------------------------


def _build_deep_agent_middleware(
    agent_config: Any,
    factory: LLMModelFactory,
    middleware: list[Any],
    visited: Optional[set[str]] = None,
) -> None:
    """Add SubAgentMiddleware and/or SkillsMiddleware to *middleware*.

    WHY: Deep agents (agents with ``subagents`` or ``skills`` in YAML config)
    require additional middleware to support sub-agent orchestration and
    skill-based prompt injection.  This mirrors the local DeepAgentService
    logic but runs at graph-build time (module load).

    SkillsMiddleware uses a FilesystemBackend that reads skills directly
    from disk.  This avoids passing a custom InMemoryStore to create_agent(),
    which LangGraph Platform rejects — the platform manages its own store.
    """
    from deepagents.backends import StateBackend
    from deepagents.middleware import SkillsMiddleware, SubAgentMiddleware

    has_subagents = bool(agent_config.subagents)
    has_skills = bool(agent_config.skills)

    if not has_subagents and not has_skills:
        return

    # --- SubAgentMiddleware ---
    if has_subagents:
        subagent_specs = _build_server_subagents(
            agent_config.subagents, factory, visited=visited
        )
        if subagent_specs:
            sub_middleware = SubAgentMiddleware(
                backend=StateBackend,
                subagents=subagent_specs,
            )
            middleware.append(sub_middleware)

    # --- SkillsMiddleware ---
    if has_skills:
        backend, skill_sources = _build_skills_backend(agent_config.skills)
        skills_middleware = SkillsMiddleware(
            backend=backend,
            sources=skill_sources,
        )
        middleware.append(skills_middleware)


def _build_server_subagents(
    subagent_names: list[str],
    factory: LLMModelFactory,
    visited: Optional[set[str]] = None,
) -> list[Any]:
    """Compile YAML-defined subagents into CompiledSubAgent specs.

    WHY: SubAgentMiddleware requires pre-compiled runnables.  For each
    subagent name declared in the parent agent's ``subagents:`` list, we
    build its model + tools + middleware and compile a full graph, then
    wrap it as a CompiledSubAgent.  If the subagent itself has nested
    subagents, the process recurses.

    The compiled subagent graphs include CostTrackingMiddleware for
    independent usage tracking, and SubagentConfigMiddleware for applying
    per-user subagent overrides at runtime.

    InstanceConfigMiddleware is intentionally EXCLUDED — LangGraph propagates
    the parent's ``RunnableConfig.configurable`` into subgraph invocations,
    so it would read the parent's ``primary_model`` / prompt overrides and
    erroneously swap the subagent's own model.  SubagentConfigMiddleware
    reads the nested ``subagent_configs[agent_name]`` dict instead, applying
    only the overrides meant for this specific subagent.
    """
    from deepagents import CompiledSubAgent

    from inference_core.agents.middleware.subagent_config import (
        SubagentConfigMiddleware,
    )

    visited = visited or set()
    specs: list[Any] = []

    for name in subagent_names:
        if name in visited:
            logger.debug(
                "Skipping subagent '%s' — already visited (recursion guard)", name
            )
            continue
        visited.add(name)

        sub_config = factory.config.get_specific_agent_config(name)
        if not sub_config:
            logger.warning("No config found for subagent '%s' — skipping", name)
            continue

        sub_model = factory.get_model_for_agent(name)
        if sub_model is None:
            logger.warning("Could not create model for subagent '%s' — skipping", name)
            continue

        # Load subagent's own tools
        sub_tools: list[Any] = []
        _load_provider_tools(name, sub_config, sub_tools)

        # Build subagent's own server middleware (cost tracking, etc.)
        # Exclude InstanceConfigMiddleware — subgraphs inherit the parent's
        # configurable, so it would apply the parent's overrides to the child.
        sub_middleware = _build_server_middleware(
            name,
            sub_config,
            factory,
            include_instance_config=False,
        )

        # Insert SubagentConfigMiddleware at position 0 so it runs first.
        # It reads subagent-specific overrides from configurable["subagent_configs"]
        # keyed by this subagent's base agent name — unlike InstanceConfigMiddleware,
        # which would incorrectly read the parent's top-level primary_model.
        subagent_config_mw = SubagentConfigMiddleware(
            agent_name=name,
            model_factory=factory,
        )
        sub_middleware.insert(0, subagent_config_mw)

        # Recurse into nested deep-agent subagents + skills
        _build_deep_agent_middleware(
            sub_config, factory, sub_middleware, visited=set(visited)
        )

        sub_graph = create_agent(
            sub_model,
            tools=sub_tools,
            middleware=sub_middleware or None,
            system_prompt=(
                sub_config.system_prompt or sub_config.description or f"You are {name}."
            ),
            name=name,
        )

        specs.append(
            CompiledSubAgent(
                name=name,
                description=sub_config.description or f"Subagent {name}",
                runnable=sub_graph,
            )
        )
        logger.info(
            "Compiled server subagent '%s' (model=%s, tools=%d, middleware=[%s])",
            name,
            factory.get_agent_model_name(name),
            len(sub_tools),
            ", ".join(type(m).__name__ for m in sub_middleware),
        )

    return specs


def _build_skills_backend(
    skill_paths: list[str],
) -> tuple[Any, list[str]]:
    """Create a FilesystemBackend and derive source directories for SkillsMiddleware.

    WHY: SkillsMiddleware needs a backend to discover and read skill files.
    On the Agent Server we use FilesystemBackend rooted at the project
    directory so skills declared in YAML (e.g. 'skills/check_weather/SKILL.md')
    are read directly from disk at runtime.  This avoids creating a custom
    InMemoryStore, which LangGraph Platform rejects when passed to
    ``create_agent(store=...)``.

    Args:
        skill_paths: Filesystem paths from YAML config (relative to project root).

    Returns:
        ``(backend, sources)`` — a FilesystemBackend instance and the list of
        source directories to pass to ``SkillsMiddleware(sources=...)``.
    """
    from deepagents.backends.filesystem import FilesystemBackend

    backend = FilesystemBackend(root_dir=_PROJECT_ROOT, virtual_mode=True)

    # Derive unique source directories from skill file paths.
    # e.g.  'skills/check_weather/SKILL.md'  →  source = 'skills/'
    sources: list[str] = []
    seen: set[str] = set()
    for raw_path in skill_paths:
        source_dir = str(PurePosixPath(raw_path).parent.parent)
        if source_dir == ".":
            source_dir = ""
        normalized = source_dir.rstrip("/") + "/" if source_dir else "/"
        if normalized not in seen:
            seen.add(normalized)
            sources.append(normalized)

    return backend, sources

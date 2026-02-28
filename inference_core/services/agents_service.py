import logging
import uuid
from contextlib import ExitStack
from datetime import UTC, datetime
from typing import Any, Callable, Optional

from deepagents import CompiledSubAgent, SubAgent
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware import SkillsMiddleware, SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import InterruptOnConfig
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore
from langgraph.store.sqlite import SqliteStore
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from inference_core.agents.middleware import (
    CostTrackingMiddleware,
    MemoryMiddleware,
    ToolBasedModelSwitchMiddleware,
    create_tool_model_switch_middleware,
)
from inference_core.agents.tools.memory_tools import (
    generate_memory_tools_system_instructions,
    get_memory_tools,
)
from inference_core.core.config import get_settings
from inference_core.llm.config import LLMConfig, get_llm_config
from inference_core.llm.models import LLMModelFactory, get_model_factory
from inference_core.llm.tools import get_registered_providers, load_tools_for_agent
from inference_core.services.agent_memory_service import AgentMemoryStoreService


class AgentMetadata(BaseModel):
    model_name: str
    tools_used: list[str]
    start_time: datetime
    end_time: Optional[datetime] = None


class AgentCostMetrics(BaseModel):
    """Cost and usage metrics from an agent run."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    extra_tokens: dict[str, int] = {}
    model_call_count: int = 0
    tool_call_count: int = 0


class AgentResponse(BaseModel):
    result: dict[str, Any]
    steps: list[dict[str, Any]]
    metadata: AgentMetadata
    cost_metrics: Optional[AgentCostMetrics] = None


class AgentService:
    def __init__(
        self,
        agent_name: Optional[str] = "default_agent",
        tools: Optional[list[Callable]] = None,
        use_checkpoints: bool = False,
        use_memory: bool = False,
        checkpoint_config: Optional[dict[str, Any]] = None,
        context_schema: Optional[Any] = None,
        middleware: Optional[list[Any]] = None,
        enable_cost_tracking: bool = True,
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        config: Optional[LLMConfig] = None,
    ):
        """Initialize the AgentService.

        Args:
            agent_name: Name of the agent configuration to use.
            tools: List of tools to provide to the agent.
            use_checkpoints: Whether to enable checkpointing for conversation history.
            use_memory: Whether to automatically add memory tools and MemoryMiddleware.
                        Requires agent_memory_enabled=True in settings and user_id.
            checkpoint_config: Configuration for the checkpointer (required if use_checkpoints=True).
            context_schema: Optional schema for agent context.
            middleware: List of middleware to apply to the agent. If None and
                       enable_cost_tracking=True, CostTrackingMiddleware is added automatically.
            enable_cost_tracking: Whether to automatically add CostTrackingMiddleware.
                                  Set to False to disable cost tracking.
            user_id: Optional user ID for cost tracking and memory attribution.
            session_id: Optional session ID for grouping related requests.
            request_id: Optional correlation ID (e.g., Celery task ID).
            config: Optional LLMConfig instance with overrides (e.g. user preferences).
        """
        # Model and tools setup
        self.agent_name = agent_name
        self.config = config

        if self.config:
            self.model_factory = LLMModelFactory(self.config)
        else:
            self.model_factory = get_model_factory()

        self.agent_config = self.model_factory.config.get_specific_agent_config(
            agent_name
        )
        self.tools = tools or []
        self.context_schema = context_schema

        self.model_name = self.model_factory.get_agent_model_name(agent_name)
        assert self.model_name is not None, "Model must have a valid name."
        self.model = self.model_factory.get_model_for_agent(agent_name)
        assert (
            self.model is not None
        ), f"Model for agent {agent_name} could not be found."

        # Exit stack to handle context managers for checkpointers
        self._exit_stack = ExitStack()

        # Checkpointing setup
        self.use_checkpoints = use_checkpoints
        self.checkpoint_config = checkpoint_config
        if use_checkpoints:
            assert checkpoint_config is not None, "Checkpoint config must be provided."
            self.checkpointer = self._initialize_checkpointer()
        else:
            self.checkpointer = None

        # Memory setup
        self.use_memory = use_memory
        self._enable_memory = False
        if use_memory:
            self.memory_store = self._initialize_memory_store()
            self._enable_memory = True
        else:
            self.memory_store = None

        # Middleware setup
        self._middleware = middleware or []
        self._enable_cost_tracking = enable_cost_tracking
        self._user_id = user_id
        self._session_id = session_id
        self._request_id = request_id

        # Memory service reference (store-backed)
        self._memory_service = None
        self._memory_store_service = None
        self._enhanced_system_prompt = None

    async def create_agent(
        self, system_prompt: Optional[str] = None, **kwargs
    ) -> Callable:
        """Create and configure the agent with tools and middleware.

        This method:
        1. Loads tools from registered providers
        2. Loads MCP tools if configured
        3. Builds middleware list (including CostTrackingMiddleware if enabled)
        4. Creates the agent with all configurations
        5. Appends memory tools instructions to system_prompt if memory is enabled

        Args:
            system_prompt: Optional system prompt/instructions for the agent.
                          If memory is enabled, memory tools usage instructions
                          will be automatically appended.
            **kwargs: Additional arguments passed to create_agent.

        Returns:
            The configured agent callable.
        """
        # Load tools from registered providers if any are configured
        provider_context = self._build_provider_user_context(kwargs.get("user_context"))
        await self._load_providers_tools(user_context=provider_context)

        # Load tools from MCP if configured
        await self._load_mcp_tools()

        # Load tools from memory if configured
        await self._load_memory_tools()

        # Build enhanced system prompt with memory instructions if enabled
        self._enhanced_system_prompt = self._build_system_prompt(system_prompt)

        # Build middleware list
        middleware = self._build_middleware()

        # Create the agent
        self.agent = create_agent(
            self.model,
            tools=self.tools,
            checkpointer=self.checkpointer,
            context_schema=self.context_schema,
            middleware=middleware,
            system_prompt=self._enhanced_system_prompt,
            **kwargs,
        )
        self.model_params = self.model_factory.config.get_model_params(self.model_name)
        return self.agent

    def _build_middleware(self) -> list[Any]:
        """Build the middleware list for the agent.

        Adds CostTrackingMiddleware automatically if enabled and not already present.
        Adds MemoryMiddleware if memory is enabled and service is available.

        Returns:
            List of middleware instances.
        """
        middleware = list(self._middleware)

        # Add CostTrackingMiddleware if enabled and not already present
        if self._enable_cost_tracking:
            # Check if CostTrackingMiddleware is already in the list
            has_cost_tracking = any(
                isinstance(m, CostTrackingMiddleware) for m in middleware
            )

            if not has_cost_tracking:
                # Get pricing config for the model
                pricing_config = None
                provider = None
                try:
                    llm_config = get_llm_config()
                    model_cfg = llm_config.models.get(self.model_name)
                    if model_cfg:
                        pricing_config = model_cfg.pricing
                        provider = model_cfg.provider
                except Exception as e:
                    logging.debug(f"Could not load pricing config: {e}")

                cost_middleware = CostTrackingMiddleware(
                    pricing_config=pricing_config,
                    user_id=self._user_id,
                    session_id=self._session_id,
                    request_id=self._request_id,
                    task_type="agent",
                    request_mode="sync",
                    provider=provider,
                )
                # Insert at the beginning so it wraps everything
                middleware.insert(0, cost_middleware)
                logging.debug(
                    f"Added CostTrackingMiddleware for agent '{self.agent_name}'"
                )

        # Add MemoryMiddleware if enabled and service available
        if self.use_memory and self._memory_service and self._user_id:
            has_memory = any(isinstance(m, MemoryMiddleware) for m in middleware)

            if not has_memory:
                try:
                    settings = get_settings()
                    memory_middleware = MemoryMiddleware(
                        memory_service=self._memory_service,
                        user_id=str(self._user_id),
                        auto_recall=settings.agent_memory_auto_recall,
                        max_recall_results=settings.agent_memory_max_results,
                        # Uses canonical MemoryType values from service by default
                    )
                    # Insert after cost tracking (if present) so memory context is logged
                    insert_pos = 1 if self._enable_cost_tracking else 0
                    middleware.insert(insert_pos, memory_middleware)
                    logging.debug(
                        f"Added MemoryMiddleware for agent '{self.agent_name}'"
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to add MemoryMiddleware for agent '{self.agent_name}': {e}",
                        exc_info=True,
                    )

        # Add ToolBasedModelSwitchMiddleware if configured in agent config
        self._add_tool_model_switch_middleware(middleware)

        return middleware

    def _build_system_prompt(self, base_prompt: Optional[str] = None) -> Optional[str]:
        """Build enhanced system prompt with memory instructions if enabled.

        Args:
            base_prompt: Base system prompt provided by user.

        Returns:
            Enhanced system prompt with memory instructions appended,
            or None if no prompt and no memory.
        """
        # If memory is enabled and we have memory tools, append instructions
        if self._memory_service and self.use_memory:
            memory_instructions = generate_memory_tools_system_instructions()

            if base_prompt:
                # Append memory instructions to existing prompt
                return f"{base_prompt}\n\n{memory_instructions}"
            else:
                # Use memory instructions as the system prompt
                return memory_instructions

        # No memory or no base prompt - return as is
        return base_prompt

    def _add_tool_model_switch_middleware(self, middleware: list[Any]) -> None:
        """Add ToolBasedModelSwitchMiddleware if tool_model_overrides is configured.

        Checks the agent configuration for tool_model_overrides and creates
        the middleware if any overrides are defined.

        Args:
            middleware: List of middleware to append to (modified in place).
        """
        if not self.agent_config:
            return

        overrides = self.agent_config.tool_model_overrides
        if not overrides:
            return

        # Check if already present
        has_tool_model_switch = any(
            isinstance(m, ToolBasedModelSwitchMiddleware) for m in middleware
        )
        if has_tool_model_switch:
            return

        try:
            # Convert Pydantic models to dicts for the factory function
            override_dicts = [
                {
                    "tool_name": o.tool_name,
                    "model": o.model,
                    "trigger": o.trigger,
                    "description": o.description,
                }
                for o in overrides
            ]

            tool_model_middleware = create_tool_model_switch_middleware(
                overrides=override_dicts,
                default_model=self.model_name,
                model_factory=self.model_factory,
                cache_models=True,
            )

            # Insert at the end (after cost tracking and memory)
            # This ensures model switching happens during the actual model call
            middleware.append(tool_model_middleware)

            logging.info(
                f"Added ToolBasedModelSwitchMiddleware for agent '{self.agent_name}' "
                f"with {len(overrides)} override(s)"
            )
        except Exception as e:
            logging.error(
                f"Failed to add ToolBasedModelSwitchMiddleware for agent "
                f"'{self.agent_name}': {e}",
                exc_info=True,
            )

    def close(self) -> None:
        self._exit_stack.close()

    def __enter__(self) -> "AgentService":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _get_checkpointer(self):
        """Initialize and return a checkpointer for the agent."""
        url = self._sync_connection_string()
        if "sqlite" in url:
            return SqliteSaver.from_conn_string(url)
        elif "postgresql" in url:
            return PostgresSaver.from_conn_string(url)
        elif "mysql" in url:
            logging.warning(
                "MySQL checkpointer not implemented, using in-memory saver."
            )
            return InMemorySaver()
        else:
            logging.warning(
                "Unknown database type for checkpointer, using in-memory saver."
            )
            return InMemorySaver()

    def _build_provider_user_context(
        self, user_context: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Build a stable context payload for provider-level tool scoping.

        This exists to keep per-request and per-user identity fields in one place
        so every provider gets the same context shape, regardless of how the agent
        was instantiated.
        """
        merged_context = dict(user_context or {})

        if self._user_id and "user_id" not in merged_context:
            merged_context["user_id"] = str(self._user_id)

        if self._session_id and "session_id" not in merged_context:
            merged_context["session_id"] = self._session_id

        if self._request_id and "request_id" not in merged_context:
            merged_context["request_id"] = self._request_id

        return merged_context

    async def _load_providers_tools(
        self,
        user_context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Load tools from registered providers for the agent."""
        # Collect tools from local providers if configured
        configured_providers = self.agent_config.local_tool_providers or []
        if not configured_providers:
            return

        registered_providers = get_registered_providers()
        confirmed_providers = []
        for registered_provider_name in registered_providers.keys():
            if registered_provider_name in configured_providers:
                logging.info(
                    f"Loading tools from provider '{registered_provider_name}' "
                    f"for agent '{self.agent_name}'"
                )
                confirmed_providers.append(registered_provider_name)

        if not confirmed_providers:
            logging.warning(
                "No registered tool providers found; skipping tool loading."
            )
            return

        try:
            provider_context = self._build_provider_user_context(user_context)
            provider_tools = await load_tools_for_agent(
                self.agent_name,
                confirmed_providers,
                allowed_tools=self.agent_config.allowed_tools,
                user_context=provider_context,
                user_id=provider_context.get("user_id"),
                session_id=provider_context.get("session_id"),
                request_id=provider_context.get("request_id"),
            )
            self.tools.extend(provider_tools)
        except Exception as e:
            logging.error(
                f"Error loading tools from provider '{registered_provider_name}': {e}",
                exc_info=True,
            )

    async def _load_mcp_tools(self) -> None:
        """Load tools from MCP if configured for this agent."""
        if not self.agent_config.mcp_profile:
            return

        try:
            from inference_core.agents.agent_mcp_tools import get_agent_mcp_manager

            manager = get_agent_mcp_manager()
            mcp_tools = await manager.get_tools_for_profile(
                self.agent_config.mcp_profile
            )

            if mcp_tools:
                logging.info(
                    f"Adding {len(mcp_tools)} MCP tools to agent '{self.agent_name}'"
                )
                self.tools.extend(mcp_tools)

        except Exception as e:
            logging.error(
                f"Error loading MCP tools for agent '{self.agent_name}': {e}",
                exc_info=True,
            )

    async def _load_memory_tools(self) -> None:
        """Load memory tools if memory is enabled for this agent.

        Memory tools (save_memory, recall_memories) are added when:
        - use_memory=True was passed to constructor
        - agent_memory_enabled=True in settings
        - user_id is available for namespace isolation
        - Store backend is configured
        """
        if not self._enable_memory:
            return

        if not self._user_id:
            logging.warning(
                f"Memory enabled for agent '{self.agent_name}' but no user_id provided. "
                "Memory tools require user_id for namespace isolation."
            )
            return

        settings = get_settings()

        # Store-based memory only
        if not (self.use_memory and self.memory_store):
            logging.warning(
                "Memory enabled but store not initialized; set use_memory=True and ensure store setup."
            )
            return

        if not self._memory_store_service:
            try:
                base_namespace = (settings.agent_memory_collection,)
                self._memory_store_service = AgentMemoryStoreService(
                    store=self.memory_store,
                    base_namespace=base_namespace,
                    max_results=settings.agent_memory_max_results,
                    agent_name=self.agent_name,
                )
            except Exception as exc:
                logging.error(
                    "Failed to initialize AgentMemoryStoreService: %s",
                    exc,
                    exc_info=True,
                )
                return

        # Expose store service as _memory_service so MemoryMiddleware can use it
        self._memory_service = self._memory_store_service

        try:
            memory_tools = get_memory_tools(
                memory_service=self._memory_store_service,
                user_id=str(self._user_id),
                session_id=self._session_id,
                max_recall_results=settings.agent_memory_max_results,
            )
            self.tools.extend(memory_tools)
            logging.info(
                "Added %d store memory tools to agent '%s'",
                len(memory_tools),
                self.agent_name,
            )
        except Exception as exc:
            logging.error(
                "Error loading store memory tools for agent '%s': %s",
                self.agent_name,
                exc,
                exc_info=True,
            )

    def _initialize_checkpointer(self):
        """Enter checkpointer context managers and run setup if available."""
        checkpointer_obj = self._get_checkpointer()
        if hasattr(checkpointer_obj, "__enter__") and hasattr(
            checkpointer_obj, "__exit__"
        ):
            checkpointer = self._exit_stack.enter_context(checkpointer_obj)
        else:
            checkpointer = checkpointer_obj

        if hasattr(checkpointer, "setup") and callable(checkpointer.setup):
            checkpointer.setup()

        return checkpointer

    def get_memory_store(self):
        """Return the initialized memory store."""
        if self.memory_store is None:
            raise ValueError("Memory store is not initialized.")
        return self.memory_store

    def _get_memory_store(self):
        """Initialize and return a memory store for the agent."""
        url = self._sync_connection_string()
        embed_fn, dims, embeddings_obj = self._init_embeddings()
        if "sqlite" in url:
            return SqliteStore.from_conn_string(
                url, index={"embed": embed_fn, "dims": dims}
            )
        elif "postgresql" in url:
            return PostgresStore.from_conn_string(
                url, index={"embed": embed_fn, "dims": dims}
            )
        elif "mysql" in url:
            logging.warning(
                "MySQL memory store not implemented, using in-memory store."
            )
            return InMemoryStore(index={"embed": embed_fn, "dims": dims})
        else:
            logging.warning(
                "Unknown database type for memory store, using in-memory store."
            )
            return InMemoryStore(index={"embed": embed_fn, "dims": dims})

    def _initialize_memory_store(self):
        """Enter memory store context managers and run setup if available."""
        store_obj = self._get_memory_store()
        if hasattr(store_obj, "__enter__") and hasattr(store_obj, "__exit__"):
            store = self._exit_stack.enter_context(store_obj)
        else:
            store = store_obj

        if hasattr(store, "setup") and callable(store.setup):
            store.setup()

        return store

    def _init_embeddings(self):
        """Prepare embedding function and dimension for store semantic search."""

        settings = get_settings()
        model_name = settings.vector_embedding_model

        try:
            model = SentenceTransformer(model_name)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logging.error("Failed to load embedding model %s: %s", model_name, exc)
            # fallback tiny embedder
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        sample = model.encode(["test"], convert_to_tensor=False)
        dims = settings.vector_dim
        try:
            # sample may be ndarray or list of lists
            if hasattr(sample, "shape") and len(getattr(sample, "shape")) >= 2:
                dims = int(sample.shape[1])
            elif isinstance(sample, (list, tuple)) and sample and sample[0] is not None:
                dims = len(sample[0])
        except Exception as exc:  # pragma: no cover - fallback
            logging.warning(
                "Falling back to configured vector_dim; dim detection failed: %s", exc
            )

        def embed_fn(texts: list[str]) -> list[list[float]]:
            return model.encode(texts, convert_to_tensor=False).tolist()

        return embed_fn, dims, model

    @staticmethod
    def _sync_connection_string() -> str:
        """Return a sync SQLAlchemy connection string for SQLChatMessageHistory.

        Map async drivers to their sync counterparts for sync-mode history.
        """
        settings = get_settings()
        url = settings.database_url
        if "+aiosqlite" in url:
            return url.replace("+aiosqlite", "")
        if "+asyncpg" in url:
            return url.replace("+asyncpg", "")
        if "+aiomysql" in url:
            return url.replace("+aiomysql", "")
        return url

    def _merge_params(
        self, task: str, runtime_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge default model params with runtime params (runtime wins)."""
        base = dict(self._default_model_params.get(task, {}))
        base.update(runtime_params)
        return base

    def get_agent(self) -> Callable:
        """Return a callable LangChain agent.

        The agent is created in the constructor and can be used to execute tools/chains.
        """
        return self.agent

    def run_agent_steps(self, user_input: str, context=None) -> AgentResponse:
        """Execute the agent with the given input and return the response.

        This method streams agent execution, collecting steps and results.
        If CostTrackingMiddleware is enabled, usage metrics are captured
        automatically and included in the response.

        Args:
            user_input: The user's input message.
            context: Optional context to pass to the agent.

        Returns:
            AgentResponse with result, steps, metadata, and optional cost metrics.
        """
        steps = []
        start_time = datetime.now(UTC)

        messages = [HumanMessage(content=user_input)]
        configurable = {}

        if self.checkpoint_config:
            configurable.update(self.checkpoint_config)

        # Track state for cost metrics and final result
        accumulated_state: dict[str, Any] = {}
        last_agent_result: dict[str, Any] = {}

        for chunk in self.agent.stream(
            {"messages": messages},
            {"configurable": configurable},
            stream_mode="updates",
            context=context,
        ):
            for step, data in chunk.items():
                steps.append({"name": step, "data": data})

                # Accumulate state updates
                if isinstance(data, dict):
                    # Track cost-related fields from middleware
                    for key in [
                        "accumulated_input_tokens",
                        "accumulated_output_tokens",
                        "accumulated_total_tokens",
                        "accumulated_extra_tokens",
                        "model_call_count",
                        "tool_call_count",
                        "usage_session_id",
                    ]:
                        if key in data:
                            accumulated_state[key] = data[key]

                    # Track agent result (messages, etc.) - skip middleware-only updates
                    if "messages" in data:
                        last_agent_result = data
                elif step == "__interrupt__":
                    last_agent_result = {"__interrupt__": data[0]}

        # Use the last result that contained messages, or fallback to last step
        if last_agent_result:
            result = last_agent_result
        elif steps:
            # Fallback: find the last step with actual content
            for step in reversed(steps):
                if isinstance(step.get("data"), dict) and step["data"].get("messages"):
                    result = step["data"]
                    break
            else:
                result = steps[-1].get("data", {}) if steps else {}
        else:
            result = {}

        tools_used = [step["name"] for step in steps]
        metadata = AgentMetadata(
            model_name=self.model_name,
            tools_used=tools_used,
            start_time=start_time,
            end_time=datetime.now(UTC),
        )

        # Extract cost metrics from accumulated state if available
        cost_metrics = None
        if accumulated_state and self._enable_cost_tracking:
            cost_metrics = AgentCostMetrics(
                input_tokens=accumulated_state.get("accumulated_input_tokens", 0),
                output_tokens=accumulated_state.get("accumulated_output_tokens", 0),
                total_tokens=accumulated_state.get("accumulated_total_tokens", 0),
                extra_tokens=accumulated_state.get("accumulated_extra_tokens", {}),
                model_call_count=accumulated_state.get("model_call_count", 0),
                tool_call_count=accumulated_state.get("tool_call_count", 0),
            )

        return AgentResponse(
            result=result,
            steps=steps,
            metadata=metadata,
            cost_metrics=cost_metrics,
        )


class DeepAgentService(AgentService):
    """Service for creating agents with sub-agent orchestration via SubAgentMiddleware.

    Uses the standard langchain `create_agent` with deepagents middleware
    (SubAgentMiddleware, SkillsMiddleware) instead of the opinionated
    `create_deep_agent` factory.  This gives full control over the parent
    agent's middleware stack while still supporting:

    * Declarative subagent specs (SubAgent TypedDict) built from YAML config
    * Pre-compiled subagent runnables (CompiledSubAgent)
    * Per-subagent interrupt_on / skills / middleware
    * SkillsMiddleware for the parent agent itself
    """

    # Source: deepagents v0.4.4 – SubAgentMiddleware, SkillsMiddleware docs

    def __init__(
        self,
        agent_name: str,
        tools: Optional[list[Callable]] = None,
        subagents: Optional[list["AgentService | SubAgent | CompiledSubAgent"]] = None,
        interrupt_on: Optional[dict[str, bool | InterruptOnConfig]] = None,
        use_checkpoints: bool = False,
        use_memory: bool = False,
        checkpoint_config: Optional[dict[str, Any]] = None,
        context_schema: Optional[Any] = None,
        middleware: Optional[list[Any]] = None,
        enable_cost_tracking: bool = True,
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        backend: Optional[BackendProtocol | BackendFactory] = None,
        _visited_agents: Optional[set[str]] = None,
    ):
        """Initialize the DeepAgentService.

        Args:
            agent_name: Name of the agent configuration to use.
            tools: List of tools to provide to the parent agent.
            subagents: Subagent specifications.  Accepts a mix of:
                - AgentService instances (compiled into CompiledSubAgent)
                - SubAgent dicts  (passed directly to SubAgentMiddleware)
                - CompiledSubAgent dicts (passed directly to SubAgentMiddleware)
            interrupt_on: Default interrupt_on config applied to config-based
                subagents that do not declare their own.
            use_checkpoints: Whether to enable checkpointing.
            use_memory: Whether to add memory tools and MemoryMiddleware.
            checkpoint_config: Checkpointer settings.
            context_schema: Optional schema for agent context.
            middleware: Extra middleware for the parent agent.
            enable_cost_tracking: Auto-add CostTrackingMiddleware.
            user_id: User ID for cost tracking / memory isolation.
            session_id: Session ID for grouping related requests.
            request_id: Correlation ID (e.g. Celery task ID).
            config: LLMConfig instance with overrides.
            backend: Backend for SubAgentMiddleware / SkillsMiddleware.
                Defaults to StateBackend (lazy factory).
            _visited_agents: Internal recursion guard.
        """
        super().__init__(
            agent_name,
            tools,
            use_checkpoints=use_checkpoints,
            use_memory=use_memory,
            checkpoint_config=checkpoint_config,
            context_schema=context_schema,
            middleware=middleware,
            enable_cost_tracking=enable_cost_tracking,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            config=config,
        )
        self._explicit_subagents = subagents or []

        # Merge interrupt_on: YAML config base + dynamic overrides on top
        self.interrupt_on: dict[str, Any] = {}
        if self.agent_config and self.agent_config.interrupt_on:
            self.interrupt_on.update(self.agent_config.interrupt_on)
        if interrupt_on:
            self.interrupt_on.update(interrupt_on)

        # Backend for SubAgentMiddleware / SkillsMiddleware; StateBackend is
        # a lightweight factory that requires no filesystem root.
        self._backend: BackendProtocol | BackendFactory = backend or StateBackend

        self._visited_agents = _visited_agents or set()
        self._visited_agents.add(agent_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_agent(
        self, system_prompt: Optional[str] = None, **kwargs
    ) -> Callable:
        """Create a standard agent with SubAgentMiddleware & SkillsMiddleware.

        Steps:
        1. Load parent's own tools (providers, MCP, memory)
        2. Build subagent specs (explicit + YAML-config-based)
        3. Inject SubAgentMiddleware into middleware stack
        4. Optionally inject SkillsMiddleware
        5. Call langchain create_agent (NOT create_deep_agent)

        Args:
            system_prompt: System prompt for the parent agent.
            **kwargs: Forwarded to ``create_agent``.

        Returns:
            The compiled agent graph.
        """
        # 1. Load parent's own tools
        provider_context = self._build_provider_user_context(kwargs.get("user_context"))
        await self._load_providers_tools(user_context=provider_context)
        await self._load_mcp_tools()
        await self._load_memory_tools()

        # 2. Build enhanced system prompt
        self._enhanced_system_prompt = self._build_system_prompt(system_prompt)

        # 3. Build base middleware (cost tracking, memory, tool-model switch)
        middleware = self._build_middleware()

        # 4. Build subagent specs and add SubAgentMiddleware
        subagent_specs = await self._build_subagent_specs()
        if subagent_specs:
            sub_middleware = SubAgentMiddleware(
                backend=self._backend,
                subagents=subagent_specs,
            )
            middleware.append(sub_middleware)

        # 5. Add SkillsMiddleware if parent has skills configured
        if self.agent_config and self.agent_config.skills:
            skills_middleware = SkillsMiddleware(
                backend=self._backend,
                sources=self.agent_config.skills,
            )
            middleware.append(skills_middleware)

        # 6. Create the agent via standard create_agent
        self.agent = create_agent(
            self.model,
            tools=self.tools,
            checkpointer=self.checkpointer,
            context_schema=self.context_schema,
            middleware=middleware,
            system_prompt=self._enhanced_system_prompt,
            **kwargs,
        )
        self.model_params = self.model_factory.config.get_model_params(self.model_name)
        return self.agent

    # ------------------------------------------------------------------
    # Subagent spec builders
    # ------------------------------------------------------------------

    async def _build_subagent_specs(
        self,
    ) -> list[SubAgent | CompiledSubAgent]:
        """Merge explicit subagents with YAML-config-based ones.

        Explicit subagents are processed first; config-based ones are added
        only if their name has not already been claimed by an explicit spec.
        A visited-agents set prevents infinite recursion.
        """
        specs: list[SubAgent | CompiledSubAgent] = []

        # --- Explicit subagents (passed via constructor) ---
        for subagent in self._explicit_subagents:
            spec = await self._resolve_explicit_subagent(subagent)
            specs.append(spec)

        # --- Config-defined subagents ---
        if self.agent_config and self.agent_config.subagents:
            added_names = self._collect_spec_names(specs)
            for subagent_name in self.agent_config.subagents:
                if subagent_name in added_names:
                    continue
                if subagent_name in self._visited_agents:
                    logging.debug(
                        "Skipping subagent '%s' – already visited (recursion guard)",
                        subagent_name,
                    )
                    continue

                spec = await self._build_subagent_from_config(subagent_name)
                if spec is not None:
                    specs.append(spec)

        return specs

    async def _resolve_explicit_subagent(
        self,
        subagent: "AgentService | SubAgent | CompiledSubAgent",
    ) -> SubAgent | CompiledSubAgent:
        """Turn an explicit subagent into a middleware-ready spec.

        AgentService instances are compiled and wrapped as CompiledSubAgent.
        Raw SubAgent / CompiledSubAgent dicts are passed through.
        """
        if isinstance(subagent, AgentService):
            if not hasattr(subagent, "agent") or subagent.agent is None:
                await subagent.create_agent()
            return CompiledSubAgent(
                name=subagent.agent_name,
                description=(
                    subagent.agent_config.description
                    or f"Subagent {subagent.agent_name}"
                ),
                runnable=subagent.agent,
            )

        # Already a SubAgent or CompiledSubAgent TypedDict
        return subagent

    async def _build_subagent_from_config(
        self, subagent_name: str
    ) -> Optional[SubAgent | CompiledSubAgent]:
        """Build a SubAgent spec from YAML agent configuration.

        Constructs a declarative SubAgent dict so that SubAgentMiddleware
        handles compilation internally.  Falls back to compiling via
        AgentService if the subagent itself has nested subagents.
        """
        sub_config = self.model_factory.config.get_specific_agent_config(subagent_name)
        if not sub_config:
            logging.warning(
                "No config found for subagent '%s' – skipping", subagent_name
            )
            return None

        # If the subagent itself has nested subagents, recursively create a
        # DeepAgentService and wrap it as CompiledSubAgent.
        if sub_config.subagents:
            return await self._compile_nested_deep_subagent(subagent_name, sub_config)

        # Otherwise build a declarative SubAgent spec
        model = self.model_factory.get_model_for_agent(subagent_name)
        tools = await self._load_tools_for_subagent(subagent_name, sub_config)

        spec: dict[str, Any] = {
            "name": subagent_name,
            "description": sub_config.description or f"Subagent {subagent_name}",
            "system_prompt": (
                sub_config.system_prompt
                or sub_config.description
                or f"You are {subagent_name}."
            ),
            "model": model,
        }

        if tools:
            spec["tools"] = tools

        # Per-subagent interrupt_on: own config wins, parent's as fallback
        sub_interrupt = sub_config.interrupt_on or self.interrupt_on
        if sub_interrupt:
            spec["interrupt_on"] = sub_interrupt

        if sub_config.skills:
            spec["skills"] = sub_config.skills

        return spec

    async def _compile_nested_deep_subagent(
        self, subagent_name: str, sub_config: Any
    ) -> CompiledSubAgent:
        """Recursively compile a subagent that itself has subagents.

        Creates a nested DeepAgentService, compiles its graph, and wraps
        the result as CompiledSubAgent for the parent's SubAgentMiddleware.
        """
        nested_service = DeepAgentService(
            agent_name=subagent_name,
            use_checkpoints=self.use_checkpoints,
            use_memory=self.use_memory,
            checkpoint_config=self.checkpoint_config,
            context_schema=self.context_schema,
            enable_cost_tracking=self._enable_cost_tracking,
            user_id=self._user_id,
            session_id=self._session_id,
            request_id=self._request_id,
            config=self.config,
            backend=self._backend,
            _visited_agents=set(self._visited_agents),
        )
        await nested_service.create_agent(
            system_prompt=(
                sub_config.system_prompt
                or sub_config.description
                or f"You are {subagent_name}."
            )
        )
        return CompiledSubAgent(
            name=subagent_name,
            description=sub_config.description or f"Subagent {subagent_name}",
            runnable=nested_service.agent,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load_tools_for_subagent(
        self, agent_name: str, agent_config: Any
    ) -> list[Any]:
        """Load tools from registered providers for a config-based subagent.

        Mirrors the tool-loading logic of AgentService._load_providers_tools
        but returns the tools instead of mutating self.tools.
        """
        configured_providers = agent_config.local_tool_providers or []
        if not configured_providers:
            return []

        registered_providers = get_registered_providers()
        confirmed = [
            name for name in configured_providers if name in registered_providers
        ]
        if not confirmed:
            return []

        try:
            return await load_tools_for_agent(
                agent_name,
                provider_names=confirmed,
                allowed_tools=agent_config.allowed_tools,
            )
        except Exception as e:
            logging.warning("Failed to load tools for subagent '%s': %s", agent_name, e)
            return []

    @staticmethod
    def _collect_spec_names(
        specs: list[SubAgent | CompiledSubAgent],
    ) -> set[str]:
        """Extract the 'name' field from a list of subagent specs."""
        names: set[str] = set()
        for s in specs:
            if isinstance(s, dict):
                name = s.get("name")
            else:
                name = getattr(s, "name", None)
            if name:
                names.add(name)
        return names

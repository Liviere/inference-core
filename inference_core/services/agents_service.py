import logging
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Optional

from deepagents import CompiledSubAgent, SubAgent
from deepagents.backends import StateBackend, StoreBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.backends.utils import create_file_data
from deepagents.middleware import SkillsMiddleware, SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import InterruptOnConfig
from langchain.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore
from langgraph.store.sqlite import SqliteStore
from pydantic import BaseModel

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
from inference_core.database.sql.models.user_agent_instance import UserAgentInstance
from inference_core.llm.config import LLMConfig, get_llm_config
from inference_core.llm.models import LLMModelFactory, get_model_factory
from inference_core.llm.tools import get_registered_providers, load_tools_for_agent
from inference_core.services._cancel import AgentCancelled
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


@dataclass(frozen=True)
class InstanceContext:
    """Identity of a UserAgentInstance backing this agent run.

    WHY: Bundles instance-level identity so that memory namespacing,
    cost tracking, and logging can distinguish between different user
    instances of the same base agent.

    subagent_configs is a dict mapping base_agent_name → override dict
    (primary_model, system_prompt_override, system_prompt_append, etc.)
    for subagents that have user instance overrides in the DB.
    Forwarded to Agent Server configurable so SubagentConfigMiddleware
    can apply per-subagent overrides at runtime.
    """

    instance_id: uuid.UUID
    instance_name: str
    base_agent_name: str
    subagent_configs: dict[str, dict[str, Any]] = field(default_factory=dict)


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
        instance_context: Optional[InstanceContext] = None,
        system_prompt_override: Optional[str] = None,
        system_prompt_append: Optional[str] = None,
        user_skills: Optional[list[dict[str, str]]] = None,
        memory_postrun_analysis: Optional[bool] = None,
        memory_postrun_model: Optional[str] = None,
    ):
        """Initialize the AgentService.

        Args:
            agent_name: Name of the agent configuration to use (YAML config key).
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
            instance_context: Optional identity of the UserAgentInstance backing this run.
                              When set, enables per-instance memory isolation, cost tracking
                              attribution, and enriched logging.
            system_prompt_override: When set, fully replaces the system_prompt passed to
                create_agent().  Used by from_user_instance() when the DB instance
                has a complete custom prompt.
            system_prompt_append: When set, appended to the system_prompt passed to
                create_agent().  Used when the DB instance extends the base prompt.
            memory_postrun_analysis: Override for post-run memory analysis flag.
                When None (default), uses settings.agent_memory_postrun_analysis_enabled.
            memory_postrun_model: Optional model name for the post-run extraction LLM
                call.  When None, uses settings.agent_memory_postrun_analysis_model
                (typically the agent's own model).
        """
        # Model and tools setup
        self.agent_name = agent_name
        self.instance_context = instance_context
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

        # DB-backed prompt customisation: stored here, applied in create_agent()
        self._system_prompt_override = system_prompt_override
        self._system_prompt_append = system_prompt_append

        # User-defined skills: list of {name, description, content} dicts
        self._user_skills = user_skills or []

        # Post-run memory analysis overrides (None = use settings defaults)
        self._memory_postrun_analysis = memory_postrun_analysis
        self._memory_postrun_model = memory_postrun_model

    @property
    def display_name(self) -> str:
        """Human-readable agent identity for logging and observability.

        Returns instance_name when running as a user agent instance,
        otherwise falls back to the YAML agent_name.
        """
        if self.instance_context:
            return self.instance_context.instance_name
        return self.agent_name

    @property
    def _is_remote(self) -> bool:
        """Whether this agent should delegate to the remote Agent Server.

        WHY: Central check used by run/arun/stream to decide execution
        path.  Returns True only when both the global feature flag AND
        the per-agent YAML setting are active.
        """
        settings = get_settings()
        return (
            settings.agent_server_enabled
            and self.agent_config.execution_mode == "remote"
        )

    def set_cancel_check(self, cancel_check: Callable[[], bool] | None) -> None:
        """Propagate a cancellation callback to middleware.

        When set, the middleware will check this callback before each model
        and tool call, raising ``AgentCancelled`` if it returns ``True``.
        """
        for mw in self._middleware:
            if isinstance(mw, CostTrackingMiddleware):
                mw.cancel_check = cancel_check
            elif hasattr(mw, "cancel_check"):
                mw.cancel_check = cancel_check

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

        # Apply DB-backed prompt overrides, then build enhanced system prompt
        # (memory tool instructions are appended by _build_system_prompt if enabled)
        effective_prompt = self._apply_prompt_overrides(system_prompt)
        self._enhanced_system_prompt = self._build_system_prompt(effective_prompt)

        # Build middleware list
        middleware = self._build_middleware()

        # Add SkillsMiddleware if user-defined skills are present
        if self._user_skills:
            store, skill_sources = self._build_skills_store(self._user_skills)
            skills_middleware = SkillsMiddleware(
                backend=(lambda rt: StoreBackend(rt)),
                sources=skill_sources,
            )
            middleware.append(skills_middleware)
            # Inject store into kwargs so create_agent passes it to the graph
            kwargs.setdefault("store", store)

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
                    model_name=self.model_name,
                    instance_id=(
                        self.instance_context.instance_id
                        if self.instance_context
                        else None
                    ),
                    instance_name=(
                        self.instance_context.instance_name
                        if self.instance_context
                        else None
                    ),
                )
                # Append at the end so CostTracking runs FIRST in the
                # reversed after_model phase — before any interrupt-causing
                # middleware (e.g. HumanInTheLoopMiddleware, SubAgentMiddleware).
                # Source: LangChain middleware docs – execution order, 2025-08
                middleware.append(cost_middleware)
                logging.debug(
                    f"Added CostTrackingMiddleware for agent '{self.display_name}'"
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
                        postrun_analysis=(
                            self._memory_postrun_analysis
                            if self._memory_postrun_analysis is not None
                            else settings.agent_memory_postrun_analysis_enabled
                        ),
                        postrun_analysis_model=(
                            self._memory_postrun_model
                            or settings.agent_memory_postrun_analysis_model
                        ),
                    )
                    # Insert before CostTracking (which sits at the end)
                    # so memory context enrichment runs in before_model first.
                    ct_index = next(
                        (
                            i
                            for i, m in enumerate(middleware)
                            if isinstance(m, CostTrackingMiddleware)
                        ),
                        len(middleware),
                    )
                    middleware.insert(ct_index, memory_middleware)
                    logging.debug(
                        f"Added MemoryMiddleware for agent '{self.display_name}'"
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to add MemoryMiddleware for agent '{self.display_name}': {e}",
                        exc_info=True,
                    )

        # Add ToolBasedModelSwitchMiddleware if configured in agent config
        self._add_tool_model_switch_middleware(middleware)

        return middleware

    @staticmethod
    def _build_skills_store(
        user_skills: list[dict[str, str]],
        base_path: str = "/skills/",
    ) -> tuple[InMemoryStore, list[str]]:
        """Build an InMemoryStore populated with user-defined skills.

        Each skill entry is a dict with 'name', 'description', 'content'.
        The content is stored as a SKILL.md file inside the store under
        ``{base_path}{name}/SKILL.md``.

        Returns:
            A tuple of (store, [base_path]) to pass to StoreBackend/SkillsMiddleware.
            The source path must be the *parent* directory of all skill subdirectories,
            because SkillsMiddleware.ls_info scans one level down for skill dirs.
        """
        store = InMemoryStore()

        for skill in user_skills:
            skill_name = skill["name"]
            store_key = f"{base_path}{skill_name}/SKILL.md"

            store.put(
                namespace=("filesystem",),
                key=store_key,
                value=create_file_data(skill["content"]),
            )

        # Return the parent directory as the single source so SkillsMiddleware
        # can discover each skill subdirectory via ls_info(base_path).
        return store, [base_path]

    def _apply_prompt_overrides(self, base_prompt: Optional[str]) -> Optional[str]:
        """Apply system_prompt_override or system_prompt_append from a UserAgentInstance.

        WHY: DB-backed instances let users fully replace or extend the caller-supplied
        system prompt without the caller needing to know about DB configuration.

        Precedence (highest to lowest):
          1. system_prompt_override  – replaces base_prompt entirely
          2. system_prompt_append    – concatenated onto base_prompt (or used alone)
          3. base_prompt             – unchanged if neither DB field is set
        """
        if self._system_prompt_override is not None:
            return self._system_prompt_override
        if self._system_prompt_append is not None:
            if base_prompt:
                return f"{base_prompt}\n\n{self._system_prompt_append}"
            return self._system_prompt_append
        return base_prompt

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
                f"Added ToolBasedModelSwitchMiddleware for agent '{self.display_name}' "
                f"with {len(overrides)} override(s)"
            )
        except Exception as e:
            logging.error(
                f"Failed to add ToolBasedModelSwitchMiddleware for agent "
                f"'{self.display_name}': {e}",
                exc_info=True,
            )

    @staticmethod
    def _ensure_cost_tracking_last(middleware: list[Any]) -> None:
        """Move CostTrackingMiddleware to the absolute end of the list.

        WHY: In the reversed ``after_model`` phase, the last middleware runs
        first.  Placing CostTracking last guarantees its state-accumulation
        hook executes before any interrupt-causing middleware
        (HumanInTheLoopMiddleware, SubAgentMiddleware) can halt the graph.

        The primary DB persistence already happens in ``wrap_model_call``
        (which is interrupt-safe), so this is a defensive measure for the
        best-effort state accumulation in ``after_model``.
        """
        ct_instances = [m for m in middleware if isinstance(m, CostTrackingMiddleware)]
        if not ct_instances:
            return
        for ct in ct_instances:
            middleware.remove(ct)
            middleware.append(ct)

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

        if self.instance_context:
            merged_context.setdefault(
                "instance_id", str(self.instance_context.instance_id)
            )
            merged_context.setdefault(
                "instance_name", self.instance_context.instance_name
            )

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
                    f"for agent '{self.display_name}'"
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
                    f"Adding {len(mcp_tools)} MCP tools to agent '{self.display_name}'"
                )
                self.tools.extend(mcp_tools)

        except Exception as e:
            logging.error(
                f"Error loading MCP tools for agent '{self.display_name}': {e}",
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
                f"Memory enabled for agent '{self.display_name}' but no user_id provided. "
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
                    agent_name=self.display_name,
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
                self.display_name,
            )
        except Exception as exc:
            logging.error(
                "Error loading store memory tools for agent '%s': %s",
                self.display_name,
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
        """Prepare embedding function and dimension for store semantic search.

        Delegates to the global EmbeddingService instead of loading
        SentenceTransformer directly. The actual compute happens either
        on a Celery prefork worker (local) or via an API provider (remote).
        """
        from inference_core.services.embedding_service import get_embedding_service

        service = get_embedding_service()
        embed_fn = service.get_embed_fn()
        dims = service.get_dimension()

        return embed_fn, dims, service

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

    @staticmethod
    def _process_message_chunk(
        token: Any,
        metadata: dict[str, Any],
        on_token: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """Extract text/reasoning from a ``messages`` stream chunk and forward to *on_token*.

        Called for every ``(token, metadata)`` pair emitted when ``stream_mode``
        includes ``"messages"``.  The *token* is typically an ``AIMessageChunk``.

        Content blocks are inspected to distinguish between regular text,
        reasoning/thinking tokens and partial tool-call arguments.  Each
        non-empty piece is forwarded to the caller-provided *on_token*
        callback with a ``meta`` dict containing:

        * ``type`` – ``"text"`` | ``"reasoning"`` | ``"tool_call"``
        * ``node`` – the LangGraph node name that produced the token
        * ``agent_name`` – agent name if available (for sub-agent disambiguation)
        """
        if not isinstance(token, AIMessageChunk):
            return

        node = metadata.get("langgraph_node", "")

        # Skip tokens emitted by middleware nodes (e.g.
        # PromptInjectionGuardMiddleware, CostTrackingMiddleware) — only
        # the main agent / model tokens should reach the UI.
        if "Middleware" in node:
            return
        agent_name = metadata.get("lc_agent_name")

        base_meta: dict[str, Any] = {"node": node}
        if agent_name:
            base_meta["agent_name"] = agent_name

        # Prefer content_blocks (normalised across providers) when available.
        content_blocks = getattr(token, "content_blocks", None)
        if content_blocks:
            for block in content_blocks:
                btype = block.get("type", "")
                if btype == "text" and block.get("text"):
                    on_token(block["text"], {**base_meta, "type": "text"})
                elif btype == "reasoning" and block.get("reasoning"):
                    on_token(block["reasoning"], {**base_meta, "type": "reasoning"})
                elif btype == "tool_call_chunk":
                    args = block.get("args")
                    if args:
                        on_token(args, {**base_meta, "type": "tool_call"})
            return

        # Fallback: plain string content (older providers / simple models).
        if token.content and isinstance(token.content, str):
            on_token(token.content, {**base_meta, "type": "text"})

    def run_agent_steps(
        self,
        user_input: str,
        context=None,
        on_step: Callable[[str, Any], None] | None = None,
        on_token: Callable[[str, dict[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        graceful_cancel: bool = True,
    ) -> AgentResponse:
        """Execute the agent with the given input and return the response.

        This method streams agent execution, collecting steps and results.
        If CostTrackingMiddleware is enabled, usage metrics are captured
        automatically and included in the response.

        Args:
            user_input: The user's input message.
            context: Optional context to pass to the agent.
            on_step: Optional callback invoked for every ``updates`` chunk.
                     Receives ``(step_name, data)`` — best-effort, never
                     breaks agent execution on callback errors.
            on_token: Optional callback invoked for every ``messages`` chunk.
                     Receives ``(text, meta)`` where *text* is the token
                     string and *meta* carries ``type`` (``"text"`` /
                     ``"reasoning"`` / ``"tool_call"``), ``node``, and
                     optionally ``agent_name``.  When provided the stream
                     switches to ``stream_mode=["updates", "messages"]``
                     so that LLM tokens are emitted alongside state
                     updates.  Best-effort — callback errors never break
                     agent execution.
            cancel_check: Optional callable that returns ``True`` when the
                          execution should be cancelled.  Checked after every
                          streaming chunk.
            graceful_cancel: When True and *cancel_check* fires, a
                ``GraphInterrupt`` is raised inside the LLM via a callback.
                This stops token generation *immediately* (HTTP connection
                closed, no further billing) and LangGraph treats the
                interrupt as a clean pause — LangSmith graph-level trace
                shows Success.  A brief blocking drain consumes the few
                remaining internal events.
                Set to False for instant cancellation via ``.close()`` at
                the cost of a failed LangSmith trace.

        Returns:
            AgentResponse with result, steps, metadata, and optional cost metrics.

        Raises:
            AgentCancelled: If *cancel_check* returns ``True``.
            RuntimeError: If agent is configured for remote execution (use arun_agent_steps instead).
        """
        if self._is_remote:
            raise RuntimeError(
                f"Agent '{self.display_name}' is configured for remote execution "
                f"(execution_mode='remote'). Use arun_agent_steps() instead — "
                f"the sync run_agent_steps() does not support remote delegation."
            )

        steps = []
        start_time = datetime.now(UTC)

        messages = [HumanMessage(content=user_input)]
        configurable = {}

        if self.checkpoint_config:
            configurable.update(self.checkpoint_config)

        # Track state for cost metrics and final result
        accumulated_state: dict[str, Any] = {}
        last_agent_result: dict[str, Any] = {}

        from inference_core.services.stream_utils import (
            StreamCancelCallback,
            SyncInterruptibleStream,
        )

        # When on_token is provided, stream both updates and messages so
        # that LLM tokens are emitted alongside the normal state updates.
        use_token_streaming = on_token is not None
        stream_mode: Any = ["updates", "messages"] if use_token_streaming else "updates"

        # Build config; inject StreamCancelCallback so we can raise
        # GraphInterrupt inside the LLM to stop token generation.
        cancel_cb: StreamCancelCallback | None = None
        stream_config: dict[str, Any] = {"configurable": configurable}
        if graceful_cancel:
            cancel_cb = StreamCancelCallback()
            stream_config["callbacks"] = [cancel_cb]

        raw_stream = self.agent.stream(
            {"messages": messages},
            stream_config,
            stream_mode=stream_mode,
            context=context,
        )

        # Wrap in SyncInterruptibleStream for clean shutdown.
        # When the cancel callback fires GraphInterrupt, the LLM stops;
        # remaining cheap internal LangGraph events are drained blocking.
        if graceful_cancel:
            stream: Any = SyncInterruptibleStream(raw_stream, cancel_callback=cancel_cb)
        else:
            stream = raw_stream

        try:
            for chunk in stream:
                # ------ Multi-mode (tuples) vs single-mode (dicts) ------
                if use_token_streaming:
                    # chunk is a tuple: (mode_name, payload)
                    mode, payload = chunk

                    if mode == "messages":
                        # payload = (AIMessageChunk, metadata_dict)
                        token, meta = payload
                        try:
                            self._process_message_chunk(token, meta, on_token)
                        except Exception:
                            pass  # best-effort

                        # Check cancellation after each token for fast interrupts
                        if cancel_check:
                            try:
                                if cancel_check():
                                    raise AgentCancelled(
                                        "Agent execution cancelled by user"
                                    )
                            except AgentCancelled:
                                raise
                            except Exception:
                                pass
                        continue

                    # mode == "updates" — fall through to existing logic
                    chunk = payload

                for step, data in chunk.items():
                    steps.append({"name": step, "data": data})

                    if on_step:
                        try:
                            on_step(step, data)
                        except Exception:
                            pass  # best-effort, never break agent execution

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
                        last_agent_result = {"__interrupt__": data}

                # Check cancellation flag after processing each chunk
                if cancel_check:
                    try:
                        if cancel_check():
                            raise AgentCancelled("Agent execution cancelled by user")
                    except AgentCancelled:
                        raise
                    except Exception:
                        pass  # best-effort, never break on check errors
        except AgentCancelled:
            # Trigger GraphInterrupt in the LLM (stops token generation)
            # and drain remaining internal events for a clean LangSmith trace.
            if graceful_cancel and isinstance(stream, SyncInterruptibleStream):
                stream.close()
            raise
        finally:
            # For non-cancel exits (normal completion or unexpected errors)
            # close the raw generator to free resources.  If already drained
            # by the except block, .close() on an exhausted generator is a no-op.
            if hasattr(raw_stream, "close"):
                try:
                    raw_stream.close()
                except Exception:
                    pass

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

    async def arun_agent_steps(
        self,
        user_input: str,
        context=None,
        on_step: Callable[[str, Any], None] | None = None,
        on_token: Callable[[str, dict[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        graceful_cancel: bool = True,
    ) -> AgentResponse:
        """Async version of run_agent_steps using astream.

        Use this in async contexts (e.g. FastAPI endpoints) to avoid
        asyncio event-loop conflicts.

        Args:
            graceful_cancel: When True and *cancel_check* fires, a
                ``GraphInterrupt`` is raised inside the LLM via a callback.
                This stops token generation *immediately* (HTTP connection
                closed, no further billing) and LangGraph treats the
                interrupt as a clean pause — LangSmith graph-level trace
                shows Success.  A brief background drain consumes the few
                remaining internal events.
                Set to False for instant cancellation via ``aclose()`` at
                the cost of a failed LangSmith trace.

        See :meth:`run_agent_steps` for a full description of *on_token*.
        """
        # --- Remote execution path ---
        if self._is_remote:
            return await self._arun_agent_steps_remote(
                user_input=user_input,
                on_step=on_step,
                on_token=on_token,
                cancel_check=cancel_check,
            )

        # --- Local execution path ---
        from inference_core.services.stream_utils import (
            InterruptibleStream,
            StreamCancelCallback,
        )

        steps = []
        start_time = datetime.now(UTC)
        messages = [HumanMessage(content=user_input)]
        configurable = {}
        if self.checkpoint_config:
            configurable.update(self.checkpoint_config)

        accumulated_state: dict[str, Any] = {}
        last_agent_result: dict[str, Any] = {}

        use_token_streaming = on_token is not None
        stream_mode: Any = ["updates", "messages"] if use_token_streaming else "updates"

        # Build config; inject StreamCancelCallback so we can raise
        # GraphInterrupt inside the LLM to stop token generation.
        cancel_cb: StreamCancelCallback | None = None
        stream_config: dict[str, Any] = {"configurable": configurable}
        if graceful_cancel:
            cancel_cb = StreamCancelCallback()
            stream_config["callbacks"] = [cancel_cb]

        raw_stream = self.agent.astream(
            {"messages": messages},
            stream_config,
            stream_mode=stream_mode,
            context=context,
        )

        # Wrap in InterruptibleStream for clean shutdown.  When the cancel
        # callback fires GraphInterrupt, the LLM stops; remaining cheap
        # internal LangGraph events are drained in the background.
        if graceful_cancel:
            stream = InterruptibleStream(raw_stream, cancel_callback=cancel_cb)
        else:
            stream = raw_stream

        try:
            async for chunk in stream:
                # ------ Multi-mode (tuples) vs single-mode (dicts) ------
                if use_token_streaming:
                    mode, payload = chunk

                    if mode == "messages":
                        token, meta = payload
                        try:
                            self._process_message_chunk(token, meta, on_token)
                        except Exception:
                            pass

                        if cancel_check:
                            try:
                                if cancel_check():
                                    raise AgentCancelled(
                                        "Agent execution cancelled by user"
                                    )
                            except AgentCancelled:
                                raise
                            except Exception:
                                pass
                        continue

                    chunk = payload

                for step, data in chunk.items():
                    steps.append({"name": step, "data": data})

                    if on_step:
                        try:
                            on_step(step, data)
                        except Exception:
                            pass  # best-effort, never break agent execution

                    if isinstance(data, dict):
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
                        if "messages" in data:
                            last_agent_result = data
                    elif step == "__interrupt__":
                        last_agent_result = {"__interrupt__": data}

                # Check cancellation flag after processing each chunk
                if cancel_check:
                    try:
                        if cancel_check():
                            raise AgentCancelled("Agent execution cancelled by user")
                    except AgentCancelled:
                        raise
                    except Exception:
                        pass  # best-effort, never break on check errors
        except AgentCancelled:
            # Trigger GraphInterrupt in the LLM (stops token generation)
            # and drain remaining internal events for a clean LangSmith trace.
            if graceful_cancel and isinstance(stream, InterruptibleStream):
                await stream.stop()
                await stream.close()
            raise

        if last_agent_result:
            result = last_agent_result
        elif steps:
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

    async def _arun_agent_steps_remote(
        self,
        user_input: str,
        on_step: Callable[[str, Any], None] | None = None,
        on_token: Callable[[str, dict[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> AgentResponse:
        """Execute agent via the remote LangGraph Agent Server.

        Phase 1 migration path — delegates the full agent run to
        Agent Server while preserving the AgentResponse interface.
        Callers (Celery tasks, FastAPI endpoints) don't need to know
        whether execution is local or remote.
        """
        from inference_core.services.agent_server_client import (
            run_remote,
            stream_remote,
        )

        start_time = datetime.now(UTC)
        steps: list[dict[str, Any]] = []

        remote_graph_id = self.agent_config.remote_graph_id

        metadata: dict[str, Any] = {
            "agent_name": self.agent_name,
            "model_name": self.model_name,
        }
        if self._user_id:
            metadata["user_id"] = str(self._user_id)
        if self._session_id:
            metadata["session_id"] = self._session_id
        if self._request_id:
            metadata["request_id"] = self._request_id
        if self.instance_context:
            metadata["instance_id"] = str(self.instance_context.instance_id)
            metadata["instance_name"] = self.instance_context.instance_name

        # Forward instance-level overrides so InstanceConfigMiddleware on
        # the Agent Server can swap model / prompt at runtime.
        if self._system_prompt_override is not None:
            metadata["system_prompt_override"] = self._system_prompt_override
        if self._system_prompt_append is not None:
            metadata["system_prompt_append"] = self._system_prompt_append
        # Forward subagent-specific overrides so SubagentConfigMiddleware
        # on each subagent graph can apply per-user overrides at runtime.
        if self.instance_context and self.instance_context.subagent_configs:
            metadata["subagent_configs"] = self.instance_context.subagent_configs
            logging.debug(
                "Remote metadata includes subagent_configs for: %s",
                list(self.instance_context.subagent_configs.keys()),
            )
        # primary_model: if the resolved config changed the model vs. the
        # base YAML config, forward the override name so the server graph
        # uses the correct model.
        if self.config:
            base_config = get_llm_config()
            base_model = base_config.agent_models.get(self.agent_name)
            if self.model_name and self.model_name != base_model:
                metadata["primary_model"] = self.model_name

        # Determine interrupt config from YAML
        interrupt_before = None
        interrupt_after = None
        if self.agent_config.interrupt_on:
            interrupt_before = self.agent_config.interrupt_on.get("before")
            interrupt_after = self.agent_config.interrupt_on.get("after")

        use_streaming = on_token is not None or on_step is not None

        logging.debug(
            "Remote agent '%s': metadata keys=%s, primary_model=%r, "
            "subagent_configs=%s",
            self.agent_name,
            list(metadata.keys()),
            metadata.get("primary_model"),
            (
                list(metadata["subagent_configs"].keys())
                if "subagent_configs" in metadata
                else "NONE"
            ),
        )

        def _capture_step(name: str, data: Any) -> None:
            steps.append({"name": name, "data": data})
            if on_step:
                try:
                    on_step(name, data)
                except Exception:
                    pass

        if use_streaming:
            result = await stream_remote(
                agent_name=self.agent_name,
                remote_graph_id=remote_graph_id,
                user_input=user_input,
                thread_id=(
                    self.checkpoint_config.get("thread_id")
                    if self.checkpoint_config
                    else None
                ),
                checkpoint_config=self.checkpoint_config,
                metadata=metadata,
                on_token=on_token,
                on_step=_capture_step,
                cancel_check=cancel_check,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
            )
        else:
            result = await run_remote(
                agent_name=self.agent_name,
                remote_graph_id=remote_graph_id,
                user_input=user_input,
                thread_id=(
                    self.checkpoint_config.get("thread_id")
                    if self.checkpoint_config
                    else None
                ),
                checkpoint_config=self.checkpoint_config,
                metadata=metadata,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
            )

        agent_metadata = AgentMetadata(
            model_name=self.model_name,
            tools_used=[s["name"] for s in steps],
            start_time=start_time,
            end_time=datetime.now(UTC),
        )

        # Remote runs don't provide local cost metrics — the Agent Server
        # tracks usage on its side.  Return zeroed metrics so consumers
        # don't need to handle None differently.
        return AgentResponse(
            result=result if isinstance(result, dict) else {"raw": result},
            steps=steps,
            metadata=agent_metadata,
            cost_metrics=None,
        )

    @staticmethod
    def build_config_for_instance(
        agent_instance_config: "dict | UserAgentInstance",
        base_config: Optional[LLMConfig] = None,
    ) -> LLMConfig:
        """Build an LLMConfig with all DB overrides from a UserAgentInstance applied.

        WHY: UserAgentInstance stores user customisations (model, tools, prompt params)
        as flat ORM columns and a JSON blob.  LLMConfig.with_overrides() needs those
        translated into typed dicts keyed by agent_name and model_name.  This method
        owns that translation so callers deal only with ORM objects or ready configs.

        Args:
            agent_instance_config: A dict (from UserAgentInstance.to_dict()) or an ORM
                UserAgentInstance object.  ORM objects are auto-converted via to_dict().
            base_config: Base LLMConfig to apply overrides onto.  When None, falls back
                to the global YAML config.  Pass the result of
                LLMConfigService.get_config_with_overrides() to include admin/user
                dynamic configuration in the resolution chain.

        Translation rules:
          primary_model                    → agent_overrides[base_agent_name]["primary"]
          config_overrides.fallback        → agent_overrides[...]["fallback"]
          config_overrides.allowed_tools   → agent_overrides[...]["allowed_tools"]
          config_overrides.mcp_profile     → agent_overrides[...]["mcp_profile"]
          config_overrides.temperature     → model_overrides[effective_model]["temperature"]
          config_overrides.max_tokens      → model_overrides[effective_model]["max_tokens"]
          instance.description             → agent_overrides[...]["description"]
        """
        # Normalise input: accept both ORM objects and plain dicts.
        if not isinstance(agent_instance_config, dict):
            agent_instance_config = agent_instance_config.to_dict()

        config = base_config if base_config is not None else get_llm_config()
        agent_name = agent_instance_config.get("base_agent_name")

        agent_overrides: dict[str, Any] = {}
        model_overrides: dict[str, dict[str, Any]] = {}

        primary_model = agent_instance_config.get("primary_model")
        if primary_model:
            agent_overrides["primary"] = primary_model

        description = agent_instance_config.get("description")
        if description:
            agent_overrides["description"] = description

        # Keys from config_overrides that map directly to AgentConfig fields
        _AGENT_LEVEL_KEYS = ("fallback", "allowed_tools", "mcp_profile")
        # Keys from config_overrides that map to ModelConfig / ModelParams fields
        _MODEL_LEVEL_KEYS = (
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "reasoning_effort",
        )

        config_overrides = agent_instance_config.get("config_overrides", {})
        if config_overrides:
            for key in _AGENT_LEVEL_KEYS:
                if key in config_overrides:
                    agent_overrides[key] = config_overrides[key]

            model_params = {
                k: config_overrides[k]
                for k in _MODEL_LEVEL_KEYS
                if k in config_overrides
            }
            if model_params:
                # Target the effective model name: DB override if set, else YAML primary.
                # Must be resolved BEFORE with_overrides() is called so the correct
                # model entry is patched regardless of whether primary_model changes it.
                target_model = primary_model or config.agent_models.get(agent_name)
                if target_model:
                    model_overrides[target_model] = model_params

        if not agent_overrides and not model_overrides:
            return config  # Nothing to override; return base config unchanged

        return config.with_overrides(
            agent_overrides={agent_name: agent_overrides} if agent_overrides else None,
            model_overrides=model_overrides if model_overrides else None,
        )

    @classmethod
    def from_user_instance(
        cls,
        instance: "UserAgentInstance",
        use_checkpoints: bool = False,
        use_memory: bool = False,
        checkpoint_config: Optional[dict[str, Any]] = None,
        context_schema: Optional[Any] = None,
        middleware: Optional[list[Any]] = None,
        enable_cost_tracking: bool = True,
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        base_config: Optional[LLMConfig] = None,
    ) -> "AgentService":
        """Create an AgentService fully configured from a UserAgentInstance ORM object.

        Steps:
          1. Apply all DB overrides (model, params, tools) to LLMConfig via
             build_config_for_instance() layered on top of base_config.
          2. Construct InstanceContext from the ORM instance fields.
          3. Pluck system_prompt_override / system_prompt_append from the DB row
             and store on the service so create_agent() applies them transparently.

        Args:
            instance: A UserAgentInstance ORM object.
            use_checkpoints / use_memory / ... : Runtime params forwarded to __init__.
            base_config: Base LLMConfig with admin/user dynamic overrides already
                         applied (e.g. from LLMConfigService.get_config_with_overrides()).
                         When None, falls back to raw YAML config.

        Returns:
            AgentService ready for ``await service.create_agent(system_prompt=...)``
            (or usable directly as a context manager).
        """
        resolved_config = cls.build_config_for_instance(
            instance, base_config=base_config
        )

        # Build subagent_configs dict for Agent Server remote execution.
        # Maps base_agent_name → override dict so SubagentConfigMiddleware
        # on each subagent graph can apply per-user overrides at runtime.
        subagent_configs: dict[str, dict[str, Any]] = {}
        for sub in getattr(instance, "subagents", None) or []:
            subagent_configs[sub.base_agent_name] = {
                "instance_id": str(sub.id),
                "instance_name": sub.instance_name,
                "primary_model": sub.primary_model,
                "system_prompt_override": sub.system_prompt_override,
                "system_prompt_append": sub.system_prompt_append,
            }
        if subagent_configs:
            logging.debug(
                "Built subagent_configs for '%s': %s",
                instance.instance_name,
                list(subagent_configs.keys()),
            )

        instance_ctx = InstanceContext(
            instance_id=instance.id,
            instance_name=instance.instance_name,
            base_agent_name=instance.base_agent_name,
            subagent_configs=subagent_configs,
        )

        return cls(
            agent_name=instance.base_agent_name,
            use_checkpoints=use_checkpoints,
            use_memory=use_memory,
            checkpoint_config=checkpoint_config,
            context_schema=context_schema,
            middleware=middleware,
            enable_cost_tracking=enable_cost_tracking,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            config=resolved_config,
            instance_context=instance_ctx,
            system_prompt_override=instance.system_prompt_override,
            system_prompt_append=instance.system_prompt_append,
            user_skills=instance.skills,
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
        system_prompt_override: Optional[str] = None,
        system_prompt_append: Optional[str] = None,
        instance_context: Optional[InstanceContext] = None,
        user_skills: Optional[list[dict[str, str]]] = None,
        _visited_agents: Optional[set[str]] = None,
    ):
        """Initialize the DeepAgentService.

        Args:
            agent_name: Name of the agent configuration to use (YAML config key).
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
            system_prompt_override: Fully replaces the system_prompt passed to
                create_agent().  Used by from_user_instance() when the DB instance
                has a complete custom prompt.
            system_prompt_append: Appended to the system_prompt passed to
                create_agent().  Used when the DB instance extends the base prompt.
            instance_context: Optional identity of the UserAgentInstance backing this run.
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
            instance_context=instance_context,
            system_prompt_override=system_prompt_override,
            system_prompt_append=system_prompt_append,
            user_skills=user_skills,
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

        # 2. Apply DB-backed prompt overrides, then build enhanced system prompt
        # (memory tool instructions are appended by _build_system_prompt if enabled)
        effective_prompt = self._apply_prompt_overrides(system_prompt)
        self._enhanced_system_prompt = self._build_system_prompt(effective_prompt)

        # 3. Build base middleware (cost tracking, memory, tool-model switch)
        middleware = self._build_middleware()

        # Extract CostTrackingMiddleware and MemoryMiddleware to propagate to subagents
        propagated_middleware = [
            m
            for m in middleware
            if isinstance(m, (CostTrackingMiddleware, MemoryMiddleware))
        ]

        # 4. Build subagent specs and add SubAgentMiddleware
        subagent_specs = await self._build_subagent_specs(
            inherited_middleware=propagated_middleware
        )
        if subagent_specs:
            sub_middleware = SubAgentMiddleware(
                backend=self._backend,
                subagents=subagent_specs,
            )
            middleware.append(sub_middleware)

        # 5. Add SkillsMiddleware if parent has skills configured (YAML or user-defined)
        yaml_skills = (self.agent_config.skills if self.agent_config else None) or []
        user_skills = self._user_skills or []

        if yaml_skills or user_skills:
            if user_skills:
                # User skills require a StoreBackend populated with skill files.
                # Merge YAML skill sources with user-defined skill sources.
                store, user_skill_sources = self._build_skills_store(user_skills)
                all_sources = list(yaml_skills) + user_skill_sources
                skills_backend = lambda rt: StoreBackend(rt)
                kwargs.setdefault("store", store)
            else:
                all_sources = list(yaml_skills)
                skills_backend = self._backend

            skills_middleware = SkillsMiddleware(
                backend=skills_backend,
                sources=all_sources,
            )
            middleware.append(skills_middleware)

        # 6. Ensure CostTrackingMiddleware is absolute last so its
        #    after_model (state accumulation) runs FIRST in reverse order,
        #    before any interrupt-causing middleware.
        self._ensure_cost_tracking_last(middleware)

        # 7. Create the agent via standard create_agent
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

    @classmethod
    async def from_user_instance(
        cls,
        instance: UserAgentInstance,
        use_checkpoints: bool = False,
        use_memory: bool = False,
        checkpoint_config: Optional[dict[str, Any]] = None,
        context_schema: Optional[Any] = None,
        middleware: Optional[list[Any]] = None,
        enable_cost_tracking: bool = True,
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        backend: Optional[BackendProtocol | BackendFactory] = None,
        base_config: Optional[LLMConfig] = None,
        _visited_agents: Optional[set[str]] = None,
    ) -> "DeepAgentService":
        """Create a DeepAgentService fully configured from a UserAgentInstance ORM object.

        WHY: UserAgentInstance stores user-specific agent customisations in the DB
        (custom model, prompt overrides, JSON config blob, M2M subagent relationships).
        This factory bridges the ORM layer to the runtime service by:
          1. Applying all DB overrides to LLMConfig so that model/config resolution
             downstream uses DB values instead of raw YAML values.
          2. Resolving the M2M subagents relationship recursively so nested deep agents
             are pre-compiled and simple agents become declarative SubAgent specs.
          3. Storing prompt override/append on the returned service so create_agent()
             applies them transparently without the caller knowing about DB config.

        Args:
            instance: A UserAgentInstance with .subagents loaded.  The default
                      lazy="selectin" on the relationship means subagents are loaded
                      automatically as long as the SQLAlchemy session is still open.
            use_checkpoints / use_memory / ... : Runtime params forwarded to __init__.
            base_config: Base LLMConfig with admin/user dynamic overrides already
                         applied.  Pass LLMConfigService.get_config_with_overrides()
                         result to include the full dynamic config resolution chain.
                         When None, falls back to raw YAML config.
            _visited_agents: Internal recursion guard.  Callers must not set this.
                             Holds both YAML agent names and instance UUID strings.

        Returns:
            DeepAgentService ready for ``await service.create_agent(system_prompt=...)``.

        Raises:
            ValueError: When a circular subagent reference is detected.
        """
        visited: set[str] = set(_visited_agents or set())
        instance_key = str(instance.id)

        if instance_key in visited:
            raise ValueError(
                f"Circular subagent reference detected for instance "
                f"'{instance.instance_name}' (id={instance.id})"
            )
        visited.add(instance_key)

        # 1. Build an LLMConfig that incorporates all DB overrides for this instance.
        #    When base_config is provided (e.g. from LLMConfigService), admin overrides
        #    and user preferences are already baked in; instance overrides go on top.
        resolved_config = cls.build_config_for_instance(
            instance, base_config=base_config
        )

        # 2. Resolve the DB subagents relationship into middleware-ready specs.
        db_subagents: list[SubAgent | CompiledSubAgent] = []

        for sub_instance in instance.subagents or []:
            sub_key = str(sub_instance.id)
            if sub_key in visited:
                logging.debug(
                    "Skipping DB subagent '%s' (id=%s) – already visited (recursion guard)",
                    sub_instance.instance_name,
                    sub_instance.id,
                )
                continue

            if sub_instance.is_deepagent or sub_instance.subagents:
                # Nested deep agent: recursively build and pre-compile its graph so the
                # parent's SubAgentMiddleware receives a CompiledSubAgent runnable.
                nested_service = await cls.from_user_instance(
                    sub_instance,
                    use_checkpoints=use_checkpoints,
                    use_memory=use_memory,
                    checkpoint_config=checkpoint_config,
                    context_schema=context_schema,
                    middleware=middleware,
                    enable_cost_tracking=enable_cost_tracking,
                    user_id=user_id,
                    session_id=session_id,
                    request_id=request_id,
                    backend=backend,
                    base_config=base_config,
                    _visited_agents=visited | {sub_key},
                )
                # system_prompt=None: the nested service applies its own
                # _system_prompt_override / _system_prompt_append internally.
                await nested_service.create_agent()
                db_subagents.append(nested_service)
            else:
                # Simple (non-deep) subagent: build a declarative SubAgent dict.
                # Model is resolved via LLMModelFactory so SubAgentMiddleware
                # receives a BaseChatModel instance and skips init_chat_model().
                sub_config = cls.build_config_for_instance(
                    sub_instance, base_config=base_config
                )
                sub_model = LLMModelFactory(sub_config).get_model_for_agent(
                    sub_instance.base_agent_name
                )
                if sub_model is None:
                    logging.warning(
                        "Could not create model for DB subagent '%s' (id=%s) – skipping",
                        sub_instance.instance_name,
                        sub_instance.id,
                    )
                    continue

                effective_system_prompt: Optional[str] = (
                    sub_instance.system_prompt_override
                    or sub_instance.description
                    or f"You are {sub_instance.instance_name}."
                )
                if (
                    sub_instance.system_prompt_append
                    and not sub_instance.system_prompt_override
                ):
                    effective_system_prompt = f"{effective_system_prompt}\n\n{sub_instance.system_prompt_append}"

                spec = cls._build_declarative_subagent_spec(
                    name=sub_instance.instance_name,
                    description=sub_instance.description
                    or f"Subagent {sub_instance.instance_name}",
                    system_prompt=effective_system_prompt,
                    model=sub_model,
                    tools=[],
                    base_agent_name=sub_instance.base_agent_name,
                )
                db_subagents.append(spec)

        # 3. Construct and return the service.  resolved_config propagates through
        #    LLMModelFactory so all model/config lookups use DB-overridden values.
        # Build subagent_configs for Agent Server remote execution (same as
        # base AgentService.from_user_instance — needed when DeepAgentService
        # is used with execution_mode='remote').
        subagent_configs: dict[str, dict[str, Any]] = {}
        for sub in instance.subagents or []:
            subagent_configs[sub.base_agent_name] = {
                "instance_id": str(sub.id),
                "instance_name": sub.instance_name,
                "primary_model": sub.primary_model,
                "system_prompt_override": sub.system_prompt_override,
                "system_prompt_append": sub.system_prompt_append,
            }

        instance_ctx = InstanceContext(
            instance_id=instance.id,
            instance_name=instance.instance_name,
            base_agent_name=instance.base_agent_name,
            subagent_configs=subagent_configs,
        )
        return cls(
            agent_name=instance.base_agent_name,
            subagents=db_subagents,
            use_checkpoints=use_checkpoints,
            use_memory=use_memory,
            checkpoint_config=checkpoint_config,
            context_schema=context_schema,
            middleware=middleware,
            enable_cost_tracking=enable_cost_tracking,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            config=resolved_config,
            backend=backend,
            system_prompt_override=instance.system_prompt_override,
            system_prompt_append=instance.system_prompt_append,
            instance_context=instance_ctx,
            user_skills=instance.skills,
            _visited_agents=visited,
        )

    async def _build_subagent_specs(
        self,
        inherited_middleware: Optional[list[Any]] = None,
    ) -> list[SubAgent | CompiledSubAgent]:
        """Merge explicit subagents with YAML-config-based ones.

        Explicit subagents are processed first; config-based ones are added
        only if their name has not already been claimed by an explicit spec.
        A visited-agents set prevents infinite recursion.
        """
        specs: list[SubAgent | CompiledSubAgent] = []
        excluded_base_names: set[str] = set()

        # --- Explicit subagents (passed via constructor) ---
        for subagent in self._explicit_subagents:
            if isinstance(subagent, AgentService):
                if subagent.instance_context:
                    excluded_base_names.add(subagent.instance_context.base_agent_name)
                elif subagent.agent_name:
                    excluded_base_names.add(subagent.agent_name)
            elif isinstance(subagent, dict) and "base_agent_name" in subagent:
                excluded_base_names.add(subagent["base_agent_name"])
                # Remove it so we don't leak non-standard keys to downstream middlewares
                subagent = {k: v for k, v in subagent.items() if k != "base_agent_name"}

            spec = await self._resolve_explicit_subagent(
                subagent, inherited_middleware=inherited_middleware
            )
            specs.append(spec)

        # --- Config-defined subagents ---
        if self.agent_config and self.agent_config.subagents:
            added_names = self._collect_spec_names(specs)
            for subagent_name in self.agent_config.subagents:
                if subagent_name in added_names or subagent_name in excluded_base_names:
                    continue
                if subagent_name in self._visited_agents:
                    logging.debug(
                        "Skipping subagent '%s' – already visited (recursion guard)",
                        subagent_name,
                    )
                    continue

                spec = await self._build_subagent_from_config(
                    subagent_name, inherited_middleware=inherited_middleware
                )
                if spec is not None:
                    specs.append(spec)

        return specs

    async def _resolve_explicit_subagent(
        self,
        subagent: "AgentService | SubAgent | CompiledSubAgent",
        inherited_middleware: Optional[list[Any]] = None,
    ) -> SubAgent | CompiledSubAgent:
        """Turn an explicit subagent into a middleware-ready spec.

        AgentService instances are compiled and wrapped as CompiledSubAgent.
        Raw SubAgent / CompiledSubAgent dicts are passed through.
        """
        if isinstance(subagent, AgentService):
            if not hasattr(subagent, "agent") or subagent.agent is None:
                await subagent.create_agent()
            return CompiledSubAgent(
                name=subagent.display_name,
                description=(
                    subagent.agent_config.description
                    or f"Subagent {subagent.display_name}"
                ),
                runnable=subagent.agent,
            )

        elif isinstance(subagent, dict) and "model" in subagent:
            subagent_name = subagent.get("name", "Unnamed Subagent")

            model = subagent.get("model")
            if model:
                if isinstance(model, str):
                    model = self.model_factory.create_model(model)
            else:
                model = self.model_factory.get_model_for_agent(subagent_name)

            if model is None:
                logging.warning(
                    "Could not create model for subagent '%s' – skipping",
                    subagent_name,
                )
                return None

            tools = subagent.get("tools", [])
            middleware = subagent.get("middleware", [])

            # Combine inherited middleware
            if inherited_middleware:
                for m in inherited_middleware:
                    # Avoid duplicates by type
                    if not any(
                        isinstance(existing, type(m)) for existing in middleware
                    ):
                        middleware.append(m)

            # CostTracking must be last so it runs first in after_model
            self._ensure_cost_tracking_last(middleware)

            # Per-subagent interrupt_on: own config wins, parent's as fallback
            sub_interrupt = subagent.get("interrupt_on") or self.interrupt_on
            description = subagent.get("description") or f"Subagent {subagent_name}"
            system_prompt = (
                subagent.get("system_prompt")
                or description
                or f"You are {subagent_name}."
            )
            skills = subagent.get("skills", [])

            spec = self._build_declarative_subagent_spec(
                name=subagent_name,
                description=description,
                system_prompt=system_prompt,
                model=model,
                tools=tools,
                interrupt_on=sub_interrupt,
                skills=skills,
                middleware=middleware,
            )

            return spec

        # Assume it's already a valid SubAgent or CompiledSubAgent
        return subagent

    async def _build_subagent_from_config(
        self,
        subagent_name: str,
        inherited_middleware: Optional[list[Any]] = None,
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
            return await self._compile_nested_deep_subagent(
                subagent_name, sub_config, inherited_middleware=inherited_middleware
            )

        # Otherwise build a declarative SubAgent spec
        model = self.model_factory.get_model_for_agent(subagent_name)
        if model is None:
            logging.warning(
                "Could not create model for subagent '%s' – skipping",
                subagent_name,
            )
            return None

        tools = await self._load_tools_for_subagent(subagent_name, sub_config)

        middleware = []
        if inherited_middleware:
            middleware.extend(inherited_middleware)

        # CostTracking must be last so it runs first in after_model
        self._ensure_cost_tracking_last(middleware)

        # Per-subagent interrupt_on: own config wins, parent's as fallback
        sub_interrupt = sub_config.interrupt_on or self.interrupt_on

        spec = self._build_declarative_subagent_spec(
            name=subagent_name,
            description=sub_config.description or f"Subagent {subagent_name}",
            system_prompt=(
                sub_config.system_prompt
                or sub_config.description
                or f"You are {subagent_name}."
            ),
            model=model,
            tools=tools,
            interrupt_on=sub_interrupt,
            skills=sub_config.skills,
            middleware=middleware,
        )

        return spec

    @staticmethod
    def _build_declarative_subagent_spec(
        name: str,
        description: str,
        system_prompt: str,
        model: Any,
        tools: list[Any],
        base_agent_name: Optional[str] = None,
        interrupt_on: Optional[dict[str, bool | InterruptOnConfig]] = None,
        skills: Optional[list[Any]] = None,
        middleware: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """Creates a standardized declarative SubAgent TypedDict.

        Using this ensures that SubAgentMiddleware receives a consistent dictionary
        with properly initialized instances, explicitly skipping string-based LangChain `init_chat_model` bugs.
        """
        spec: dict[str, Any] = {
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "model": model,
            "tools": tools,
        }

        if base_agent_name:
            spec["base_agent_name"] = base_agent_name
        if interrupt_on is not None:
            spec["interrupt_on"] = interrupt_on
        if skills is not None:
            spec["skills"] = skills
        if middleware is not None:
            spec["middleware"] = middleware

        return spec

    async def _compile_nested_deep_subagent(
        self,
        subagent_name: str,
        sub_config: Any,
        inherited_middleware: Optional[list[Any]] = None,
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
            middleware=inherited_middleware,
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
        but returns the tools instead of mutating self.tools.  The parent's
        user context (user_id, session_id, request_id) is forwarded so that
        user-scoped providers (e.g. email tools) can resolve per-user
        resources.
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

        # Build the same user context that _load_providers_tools passes,
        # reusing the parent agent's identity fields.
        provider_context = self._build_provider_user_context()

        try:
            return await load_tools_for_agent(
                agent_name,
                provider_names=confirmed,
                allowed_tools=agent_config.allowed_tools,
                user_context=provider_context,
                user_id=provider_context.get("user_id"),
                session_id=provider_context.get("session_id"),
                request_id=provider_context.get("request_id"),
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

import logging
import uuid
from contextlib import ExitStack
from datetime import UTC, datetime
from typing import Any, Callable, Optional

from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from inference_core.agents.middleware import (
    CostTrackingMiddleware,
    MemoryMiddleware,
    ToolBasedModelSwitchMiddleware,
    create_tool_model_switch_middleware,
)
from inference_core.agents.tools.memory_tools import get_memory_tools
from inference_core.core.config import get_settings
from inference_core.llm.config import get_llm_config
from inference_core.llm.models import get_model_factory
from inference_core.llm.tools import get_registered_providers, load_tools_for_agent
from inference_core.services.agent_memory_service import get_agent_memory_service


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
        checkpoint_config: Optional[dict[str, Any]] = None,
        context_schema: Optional[Any] = None,
        middleware: Optional[list[Any]] = None,
        enable_cost_tracking: bool = True,
        enable_memory: bool = False,
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        """Initialize the AgentService.

        Args:
            agent_name: Name of the agent configuration to use.
            tools: List of tools to provide to the agent.
            use_checkpoints: Whether to enable checkpointing for conversation history.
            checkpoint_config: Configuration for the checkpointer (required if use_checkpoints=True).
            context_schema: Optional schema for agent context.
            middleware: List of middleware to apply to the agent. If None and
                       enable_cost_tracking=True, CostTrackingMiddleware is added automatically.
            enable_cost_tracking: Whether to automatically add CostTrackingMiddleware.
                                  Set to False to disable cost tracking.
            enable_memory: Whether to automatically add memory tools and MemoryMiddleware.
                          Requires agent_memory_enabled=True in settings and user_id.
            user_id: Optional user ID for cost tracking and memory attribution.
            session_id: Optional session ID for grouping related requests.
            request_id: Optional correlation ID (e.g., Celery task ID).
        """
        # Model and tools setup
        self.agent_name = agent_name
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

        # Middleware setup
        self._middleware = middleware or []
        self._enable_cost_tracking = enable_cost_tracking
        self._enable_memory = enable_memory
        self._user_id = user_id
        self._session_id = session_id
        self._request_id = request_id

        # Memory service reference (lazy loaded)
        self._memory_service = None

    async def create_agent(self, **kwargs) -> Callable:
        """Create and configure the agent with tools and middleware.

        This method:
        1. Loads tools from registered providers
        2. Loads MCP tools if configured
        3. Builds middleware list (including CostTrackingMiddleware if enabled)
        4. Creates the agent with all configurations

        Returns:
            The configured agent callable.
        """
        # Load tools from registered providers if any are configured
        await self._load_providers_tools()

        # Load tools from MCP if configured
        await self._load_mcp_tools()

        # Build memory tools if memory is enabled
        await self._load_memory_tools()

        # Build middleware list
        middleware = self._build_middleware()

        # Create the agent
        self.agent = create_agent(
            self.model,
            tools=self.tools,
            checkpointer=self.checkpointer,
            context_schema=self.context_schema,
            middleware=middleware,
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
        if self._enable_memory and self._memory_service and self._user_id:
            has_memory = any(isinstance(m, MemoryMiddleware) for m in middleware)

            if not has_memory:
                try:
                    settings = get_settings()
                    memory_middleware = MemoryMiddleware(
                        memory_service=self._memory_service,
                        user_id=str(self._user_id),
                        auto_recall=settings.agent_memory_auto_recall,
                        max_recall_results=settings.agent_memory_max_results,
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

    async def _load_providers_tools(self) -> None:
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
            provider_tools = await load_tools_for_agent(
                self.agent_name,
                confirmed_providers,
                allowed_tools=self.agent_config.allowed_tools,
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
        - enable_memory=True was passed to constructor
        - agent_memory_enabled=True in settings
        - user_id is available for namespace isolation
        - Vector store backend is configured
        """
        if not self._enable_memory:
            return

        if not self._user_id:
            logging.warning(
                f"Memory enabled for agent '{self.agent_name}' but no user_id provided. "
                "Memory tools require user_id for namespace isolation."
            )
            return

        # Get memory service (lazy initialization)
        self._memory_service = get_agent_memory_service()
        if not self._memory_service:
            logging.warning(
                f"Memory enabled for agent '{self.agent_name}' but AgentMemoryService "
                "unavailable. Check agent_memory_enabled and vector_backend settings."
            )
            return

        try:
            settings = get_settings()
            memory_tools = get_memory_tools(
                memory_service=self._memory_service,
                user_id=str(self._user_id),
                session_id=self._session_id,
                upsert_mode=settings.agent_memory_upsert_by_similarity,
                max_recall_results=settings.agent_memory_max_results,
            )
            self.tools.extend(memory_tools)
            logging.info(
                f"Added {len(memory_tools)} memory tools to agent '{self.agent_name}'"
            )
        except Exception as e:
            logging.error(
                f"Error loading memory tools for agent '{self.agent_name}': {e}",
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
    """Service for creating deep agents with sub-agent support.

    Deep agents are designed for complex multi-step tasks that may require
    coordination with other specialized agents.
    """

    def __init__(
        self,
        agent_name: str,
        tools: Optional[list[Callable]] = None,
        subagents: Optional[list["AgentService"]] = None,
        use_checkpoints: bool = False,
        checkpoint_config: Optional[dict[str, Any]] = None,
        middleware: Optional[list[Any]] = None,
        enable_cost_tracking: bool = True,
        enable_memory: bool = False,
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        """Initialize the DeepAgentService.

        Args:
            agent_name: Name of the agent configuration to use.
            tools: List of tools to provide to the agent.
            subagents: List of sub-agents that can be invoked by this agent.
            use_checkpoints: Whether to enable checkpointing for conversation history.
            checkpoint_config: Configuration for the checkpointer.
            middleware: List of middleware to apply to the agent.
            enable_cost_tracking: Whether to automatically add CostTrackingMiddleware.
            enable_memory: Whether to automatically add memory tools and MemoryMiddleware.
            user_id: Optional user ID for cost tracking and memory attribution.
            session_id: Optional session ID for grouping related requests.
            request_id: Optional correlation ID (e.g., Celery task ID).
        """
        super().__init__(
            agent_name,
            tools,
            use_checkpoints=use_checkpoints,
            checkpoint_config=checkpoint_config,
            middleware=middleware,
            enable_cost_tracking=enable_cost_tracking,
            enable_memory=enable_memory,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
        )
        self.subagents = [s.agent for s in subagents] if subagents else []

    async def create_agent(self) -> Callable:
        """Create and configure the deep agent with tools, middleware, and sub-agents.

        Returns:
            The configured deep agent callable.
        """
        # Load tools from registered providers if any are configured
        await self._load_providers_tools()

        # Load tools from MCP if configured
        await self._load_mcp_tools()

        # Build middleware list
        middleware = self._build_middleware()

        # Create the deep agent
        self.agent = create_deep_agent(
            self.model,
            tools=self.tools,
            middleware=middleware,
        )
        self.model_params = self.model_factory.config.get_model_params(self.model_name)
        return self.agent

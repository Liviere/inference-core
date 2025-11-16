import logging
from contextlib import ExitStack
from datetime import UTC, datetime
from typing import Any, Callable, Optional

from deepagents import create_deep_agent
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from inference_core.core.config import get_settings
from inference_core.llm.models import get_model_factory
from inference_core.llm.tools import get_registered_providers, load_tools_for_agent


class AgentMetadata(BaseModel):
    model_name: str
    tools_used: list[str]
    start_time: datetime
    end_time: Optional[datetime] = None


class AgentResponse(BaseModel):
    result: dict[str, Any]
    steps: list[dict[str, Any]]
    metadata: AgentMetadata


class AgentService:
    def __init__(
        self,
        agent_name: Optional[str] = "default_agent",
        tools: Optional[list[Callable]] = None,
        use_checkpoints: bool = False,
        checkpoint_config: Optional[dict[str, Any]] = None,
        context_schema: Optional[Any] = None,
    ):
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

    async def create_agent(self) -> Callable:
        # Load tools from registered providers if any are configured
        await self._load_providers_tools()

        # Create the agent
        self.agent = create_agent(
            self.model,
            tools=self.tools,
            checkpointer=self.checkpointer,
            context_schema=self.context_schema,
        )
        self.model_params = self.model_factory.config.get_model_params(self.model_name)
        return self.agent

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

        steps = []
        start_time = datetime.now(UTC)

        messages = [{"role": "user", "content": user_input}]
        configurable = {}

        if self.checkpoint_config:
            configurable.update(self.checkpoint_config)

        for chunk in self.agent.stream(
            {"messages": messages},
            {"configurable": configurable},
            stream_mode="updates",
            context=context,
        ):
            for step, data in chunk.items():
                steps.append({"name": step, "data": data})

        result = steps[-1]["data"] if steps else {}
        tools_used = [step["name"] for step in steps]
        metadata = AgentMetadata(
            model_name=self.model_name,
            tools_used=tools_used,
            start_time=start_time,
            end_time=datetime.now(UTC),
        )
        return AgentResponse(
            result=result,
            steps=steps,
            metadata=metadata,
        )


class DeepAgentService(AgentService):

    def __init__(
        self,
        agent_name: str,
        tools: Optional[list[Callable]] = None,
        subagents: Optional[list["AgentService"]] = None,
        use_checkpoints: bool = False,
        checkpoint_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            agent_name,
            tools,
            use_checkpoints=use_checkpoints,
            checkpoint_config=checkpoint_config,
        )
        self.subagents = [s.agent for s in subagents] if subagents else []
        self.agent = create_deep_agent(self.model, tools=self.tools)

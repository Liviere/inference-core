from datetime import UTC, datetime
from typing import Any, Callable, Optional

from deepagents import create_deep_agent
from langchain.agents import create_agent
from pydantic import BaseModel

from inference_core.llm.models import get_model_factory


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
    def __init__(self, task_name: str, tools: Optional[list[Callable]] = None):
        self.model_factory = get_model_factory()
        self.tools = tools or []
        self.model = self.model_factory.get_model_for_task(task_name)
        self.model_name = self.model.model_name
        self.agent = create_agent(self.model, tools=self.tools)
        self.model_params = self.model_factory.config.get_model_params(self.model_name)

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

    def run_agent_steps(self, user_input: str) -> AgentResponse:

        steps = []
        start_time = datetime.now(UTC)

        for chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="updates",
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
        task_name: str,
        tools: Optional[list[Callable]] = None,
        subagents: Optional[list[AgentService]] = None,
    ):
        super().__init__(task_name, tools)
        self.subagents = [s.agent for s in subagents] if subagents else []
        self.agent = create_deep_agent(self.model, tools=self.tools)

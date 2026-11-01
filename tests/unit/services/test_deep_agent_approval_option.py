"""``approval_middleware``: who asks about a subagent's tool calls.

deepagents makes a ``HumanInTheLoopMiddleware`` of a spec's ``interrupt_on``
and puts it last. An application with an approval middleware of its own gives
it here, and it takes that place. Without it, a spec stays as it is.
"""

from typing import Any, List, Optional

import pytest
from deepagents import create_deep_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import PrivateAttr

from inference_core.agents.middleware import RETURNED_STATE_KEYS
from inference_core.services.agents_service import DeepAgentService


def _service(**kwargs: Any) -> DeepAgentService:
    service = DeepAgentService.__new__(DeepAgentService)
    service._approval_middleware = kwargs.get("approval_middleware")
    return service


def test_without_the_option_a_spec_stays_as_it_is():
    spec = {"name": "a", "interrupt_on": {"send": True}, "middleware": ["m"]}

    assert _service()._with_own_approvals(spec) is spec


def test_the_option_takes_the_last_place_in_the_list():
    made = []

    def approvals(rules):
        made.append(rules)
        return "gate"

    spec = {"name": "a", "interrupt_on": {"send": True}, "middleware": ["m"]}
    changed = _service(approval_middleware=approvals)._with_own_approvals(spec)

    assert made == [{"send": True}]
    assert changed["middleware"] == ["m", "gate"]
    assert "interrupt_on" not in changed
    # What the caller gave is left alone.
    assert spec["middleware"] == ["m"] and "interrupt_on" in spec


@pytest.mark.parametrize("spec", [{"name": "a"}, {"name": "a", "interrupt_on": {}}])
def test_a_spec_that_asks_about_nothing_gets_nothing(spec):
    def approvals(rules):
        raise AssertionError("nothing to ask about")

    assert _service(approval_middleware=approvals)._with_own_approvals(spec) is spec


def test_the_default_keys_are_the_middlewares():
    import inspect

    default = inspect.signature(DeepAgentService.__init__).parameters[
        "returned_state_keys"
    ]
    assert default.default == RETURNED_STATE_KEYS


class _Model(BaseChatModel):
    _script: list = PrivateAttr(default_factory=list)

    def __init__(self, script: List[AIMessage]) -> None:
        super().__init__()
        self._script = list(script)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_Model":
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kw: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._script.pop(0))])


@tool
def send(text: str) -> str:
    """Send a text."""
    return "sent"


class _Gate(HumanInTheLoopMiddleware):
    asked: List[str] = []

    def after_model(self, state: Any, runtime: Any) -> Any:
        _Gate.asked.append("gate")
        return super().after_model(state, runtime)


@pytest.mark.asyncio
async def test_a_subagent_compiled_with_the_option_asks_through_it():
    _Gate.asked = []
    service = _service(approval_middleware=lambda rules: _Gate(interrupt_on=rules))
    spec = service._with_own_approvals(
        {
            "name": "sender",
            "description": "Sends.",
            "system_prompt": "You send.",
            "model": _Model(
                [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "send", "args": {"text": "x"}, "id": "c"}],
                    ),
                ]
            ),
            "tools": [send],
            "interrupt_on": {"send": True},
        }
    )
    agent = create_deep_agent(
        model=_Model(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {"description": "Send.", "subagent_type": "sender"},
                            "id": "t",
                        }
                    ],
                ),
            ]
        ),
        tools=[],
        subagents=[spec],
        checkpointer=InMemorySaver(),
    )

    stopped = await agent.ainvoke(
        {"messages": [("user", "send")]}, {"configurable": {"thread_id": "gate"}}
    )

    assert stopped["__interrupt__"]
    assert _Gate.asked == ["gate"]

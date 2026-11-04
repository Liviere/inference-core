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

from inference_core.agents.middleware import RETURNED_STATE_KEYS, CostTrackingMiddleware
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


def test_cost_tracking_stays_behind_the_approvals():
    """``after_model`` runs last-to-first: cost tracking has to take up the call
    before an approval stops the run, or the resumed run logs it again."""
    cost = CostTrackingMiddleware()
    spec = {
        "name": "a",
        "interrupt_on": {"send": True},
        "middleware": ["m", cost, "inner"],
    }

    changed = _service(approval_middleware=lambda rules: "gate")._with_own_approvals(spec)

    assert changed["middleware"] == ["m", "gate", cost, "inner"]
    assert spec["middleware"] == ["m", cost, "inner"]


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


# ---------------------------------------------------------------------------
# subagent_approvals_ahead_of_cost_tracking: deepagents' own approvals
# ---------------------------------------------------------------------------


def _reply(tokens: int, **kwargs: Any) -> AIMessage:
    """A model's message whose usage says which call it was."""
    return AIMessage(
        content=kwargs.pop("content", ""),
        usage_metadata={
            "input_tokens": tokens,
            "output_tokens": 1,
            "total_tokens": tokens + 1,
        },
        **kwargs,
    )


@pytest.fixture
def logged():
    """The input tokens of each usage log row written, one entry a row."""
    import uuid
    from unittest.mock import patch

    from inference_core.llm.usage_logging import UsageSession

    rows: List[int] = []

    def finalize_sync(session, **kwargs):
        rows.append(kwargs["final_usage"]["input_tokens"])
        return uuid.uuid4()

    async def finalize(session, **kwargs):
        return finalize_sync(session, **kwargs)

    with patch.object(UsageSession, "finalize_sync", finalize_sync), patch.object(
        UsageSession, "finalize", finalize
    ):
        yield rows


SUBAGENT_CALL = 100


def _team(cost: CostTrackingMiddleware, parent: List[AIMessage], sub: List[AIMessage], saver):
    """A parent whose subagent asks before it sends, as deepagents builds it
    from a spec with ``interrupt_on`` and no ``approval_middleware``."""
    spec = _service()._with_own_approvals(
        {
            "name": "sender",
            "description": "Sends.",
            "system_prompt": "You send.",
            "model": _Model(sub),
            "tools": [send],
            "middleware": [cost],
            "interrupt_on": {"send": True},
        }
    )
    return create_deep_agent(
        model=_Model(parent), tools=[], subagents=[spec], checkpointer=saver
    )


async def _run(agent: Any, value: Any, config: dict, sync: bool) -> Any:
    if sync:
        return agent.invoke(value, config)
    return await agent.ainvoke(value, config)


async def _approved_once(logged: List[int], sync: bool) -> List[int]:
    """Rows of a run stopped by the subagent's approval and resumed in what
    stands for another worker: a graph built anew, with a cost tracking of its
    own that has not seen the call before the stop."""
    from langgraph.types import Command

    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "resumed"}}
    task_call = _reply(
        10,
        tool_calls=[
            {
                "name": "task",
                "args": {"description": "Send.", "subagent_type": "sender"},
                "id": "t",
            }
        ],
    )
    send_call = _reply(
        SUBAGENT_CALL,
        tool_calls=[{"name": "send", "args": {"text": "x"}, "id": "c"}],
    )
    stopped = await _run(
        _team(
            CostTrackingMiddleware(pricing_config=None), [task_call], [send_call], saver
        ),
        {"messages": [("user", "send")]},
        config,
        sync,
    )
    assert stopped["__interrupt__"]
    before = list(logged)

    await _run(
        _team(
            CostTrackingMiddleware(pricing_config=None),
            [_reply(20, content="Sent.")],
            [_reply(200, content="Done.")],
            saver,
        ),
        Command(resume={"decisions": [{"type": "approve"}]}),
        config,
        sync,
    )
    return before


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_by_default_a_call_before_deepagents_approval_is_logged_again(
    logged, sync
):
    """What the option is for: deepagents puts its approvals last, they stop the
    run before cost tracking took up the call, and the resumed run logs it from
    the state a second time. The default stays as it was."""
    before = await _approved_once(logged, sync)

    assert before.count(SUBAGENT_CALL) == 1
    assert logged.count(SUBAGENT_CALL) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_with_the_option_a_call_before_the_approval_is_logged_once(
    logged, monkeypatch, sync
):
    monkeypatch.setattr(
        DeepAgentService, "subagent_approvals_ahead_of_cost_tracking", True
    )

    before = await _approved_once(logged, sync)

    assert before.count(SUBAGENT_CALL) == 1
    assert logged.count(SUBAGENT_CALL) == 1
    # The run went on: the subagent's call after the approval is logged too
    # (only the subagent tracks its costs here).
    assert logged == [SUBAGENT_CALL, 200]


def test_with_the_option_deepagents_approvals_go_ahead_of_cost_tracking(monkeypatch):
    monkeypatch.setattr(
        DeepAgentService, "subagent_approvals_ahead_of_cost_tracking", True
    )
    cost = CostTrackingMiddleware()
    spec = {"name": "a", "interrupt_on": {"send": True}, "middleware": ["m", cost]}

    changed = _service()._with_own_approvals(spec)

    [gate] = [m for m in changed["middleware"] if isinstance(m, HumanInTheLoopMiddleware)]
    assert changed["middleware"] == ["m", gate, cost]
    assert gate.interrupt_on.keys() == {"send"}
    assert "interrupt_on" not in changed


def test_an_applications_own_approvals_win_over_the_option(monkeypatch):
    monkeypatch.setattr(
        DeepAgentService, "subagent_approvals_ahead_of_cost_tracking", True
    )
    spec = {"name": "a", "interrupt_on": {"send": True}, "middleware": ["m"]}

    changed = _service(approval_middleware=lambda rules: "gate")._with_own_approvals(
        spec
    )

    assert changed["middleware"] == ["m", "gate"]


class _Stack:
    def __init__(self, calls):
        self.calls = calls

    def close(self):
        self.calls.append("stack")


def test_closing_connections_leaves_the_model_factory_open_and_goes_down_the_nesting():
    """Without a config of its own a service has the process's factory, whose
    models other runs are using."""
    calls = []
    nested = _service()
    nested._nested_services = []
    nested._exit_stack = _Stack(calls)
    nested.model_factory = pytest.fail  # never touched
    service = _service()
    service._nested_services = [nested]
    service._exit_stack = _Stack(calls)
    service.model_factory = pytest.fail

    service.close_connections()

    assert calls == ["stack", "stack"]

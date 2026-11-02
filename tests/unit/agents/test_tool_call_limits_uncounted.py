"""The calls a per-tool limit does not count (``uncounted``), on real graphs."""

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool, ToolException

from inference_core.agents.middleware.tool_call_limits import (
    UncountingToolCallLimitMiddleware,
    build_tool_call_limit_middleware,
)
from inference_core.llm.config import ToolCallLimitsConfig

TIMED_OUT = "Timed out: "
LIMIT = "Tool call limit exceeded."


def _fetch(ok: bool) -> str:
    if not ok:
        raise ToolException(f"{TIMED_OUT}the page took too long")
    return "page"


fetch = StructuredTool.from_function(
    _fetch, name="fetch", description="Fetch a page.", handle_tool_error=True
)


def _timed_out(result: ToolMessage) -> bool:
    return result.status == "error" and str(result.content).startswith(TIMED_OUT)


class _Scripted(BaseChatModel):
    """Asks for the calls of each batch in turn, then answers."""

    batches: list = []
    turn: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_Scripted":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.turn < len(self.batches):
            calls = [
                {"name": "fetch", "args": {"ok": ok}, "id": f"c{self.turn}-{i}"}
                for i, ok in enumerate(self.batches[self.turn])
            ]
            reply = AIMessage("", tool_calls=calls)
        else:
            reply = AIMessage("Done.")
        self.turn += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


# Two calls that time out at once, then three that work, one by one.
BATCHES = [[False, False], [True], [True], [True]]


def _refused(limits: dict, uncounted=None) -> list[str]:
    """The ids of the calls a limit refused (a call two limits refuse has
    an answer from each)."""
    config = ToolCallLimitsConfig(**limits)
    agent = create_agent(
        _Scripted(batches=BATCHES),
        tools=[fetch],
        middleware=build_tool_call_limit_middleware(config, uncounted=uncounted),
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    return sorted(
        {
            m.tool_call_id
            for m in result["messages"]
            if isinstance(m, ToolMessage) and str(m.content).startswith(LIMIT)
        }
    )


PER_TOOL = {"per_tool": [{"tool_name": "fetch", "run_limit": 2}]}


def test_without_the_option_every_call_counts():
    """LangChain's own counting: the two timeouts use up the limit."""
    assert _refused(PER_TOOL) == ["c1-0", "c2-0", "c3-0"]


def test_calls_that_timed_out_are_given_back():
    assert _refused(PER_TOOL, uncounted=_timed_out) == ["c3-0"]


def test_the_global_limit_counts_every_call():
    limits = {"global_limit": {"run_limit": 3}, **PER_TOOL}
    assert _refused(limits, uncounted=_timed_out) == ["c2-0", "c3-0"]


def test_only_per_tool_limits_give_back():
    limits = {"global_limit": {"run_limit": 3}, **PER_TOOL}
    built = build_tool_call_limit_middleware(
        ToolCallLimitsConfig(**limits), uncounted=_timed_out
    )
    assert [type(m).__name__ for m in built] == [
        "ToolCallLimitMiddleware",
        "UncountingToolCallLimitMiddleware",
    ]
    # The same name as LangChain's: one limit of a tool.
    assert built[1].name == "ToolCallLimitMiddleware[fetch]"


def test_a_failing_check_counts_the_call():
    def broken(result: ToolMessage) -> bool:
        raise RuntimeError("no")

    assert _refused(PER_TOOL, uncounted=broken) == ["c1-0", "c2-0", "c3-0"]


def test_the_thread_count_is_given_back_too():
    middleware = UncountingToolCallLimitMiddleware(
        uncounted=_timed_out, tool_name="fetch", thread_limit=5
    )
    timed_out = ToolMessage(
        f"{TIMED_OUT}slow", tool_call_id="a", name="fetch", status="error"
    )
    worked = ToolMessage("page", tool_call_id="b", name="fetch")
    state = {
        "messages": [
            AIMessage(
                "",
                tool_calls=[
                    {"name": "fetch", "args": {}, "id": "a"},
                    {"name": "fetch", "args": {}, "id": "b"},
                ],
            ),
            timed_out,
            worked,
            AIMessage("Done."),
        ],
        "thread_tool_call_count": {"fetch": 2},
        "run_tool_call_count": {"fetch": 2},
    }
    update = middleware.after_model(state, None)
    assert update == {
        "thread_tool_call_count": {"fetch": 1},
        "run_tool_call_count": {"fetch": 1},
    }

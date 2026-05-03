"""
Tutorial: orchestrate inference-core AgentService from a LangGraph graph.

This example demonstrates the bridge between two layers:

1. LangGraph controls deterministic workflow shape with StateGraph nodes.
2. AgentService owns the LangChain agent lifecycle, tools, middleware, and
   optional remote Agent Server delegation.

Run the safe mock version first:

    poetry run python examples/langgraph_agentservice_tutorial.py \
        --mock-agent \
        --prompt "Explain how this bridge works" \
        --outer-stream

Then run the real AgentService version after configuring llm_config.yaml and a
provider API key:

    poetry run python examples/langgraph_agentservice_tutorial.py \
        --agent default_agent \
        --prompt "Explain LangGraph in one paragraph" \
        --outer-stream
"""

import argparse
import asyncio
import json
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_AGENT_NAME = "default_agent"
DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001"
DEFAULT_SESSION_ID = "langgraph-agentservice-tutorial"
DEFAULT_PROMPT = "Explain how LangGraph can orchestrate our AgentService."
TRACE_MODE_CHOICES = ("separate", "nested", "nested-only")
AGENT_EXECUTION_MODE_CHOICES = ("auto", "configured", "local")
TraceMode = Literal["separate", "nested", "nested-only"]
AgentExecutionMode = Literal["auto", "configured", "local"]


class TutorialState(TypedDict, total=False):
    """State shared by the outer LangGraph tutorial workflow."""

    prompt: str
    normalized_prompt: str
    agent_name: str
    user_id: str
    session_id: str
    use_memory: bool
    mock_agent: bool
    trace_mode: TraceMode
    agent_execution_mode: AgentExecutionMode
    agent_result: dict[str, Any]
    agent_steps: list[dict[str, Any]]
    agent_tokens: list[dict[str, Any]]
    cost_metrics: dict[str, Any] | None
    report: str


TokenCallback = Callable[[str, dict[str, Any]], None]
StepCallback = Callable[[str, Any], None]
AgentRunner = Callable[
    [TutorialState, TokenCallback, StepCallback, RunnableConfig | None],
    Awaitable["TutorialAgentRun"],
]


@dataclass(slots=True)
class TutorialAgentRun:
    """Small result object shared by real and mock agent runners.

    WHY: The LangGraph node should not care whether it calls a real
    AgentService instance or a deterministic tutorial double. Keeping one
    shape makes the mock path teach the same integration contract.
    """

    result: dict[str, Any]
    cost_metrics: dict[str, Any] | None = None


def _to_plain_data(value: Any) -> Any:
    """Convert LangChain/Pydantic objects into printable Python data.

    WHY: Agent responses can contain message objects, Pydantic models, UUIDs,
    and datetimes. The tutorial report should stay readable and JSON-friendly
    without depending on provider-specific response classes.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (uuid.UUID, datetime, date)):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(item) for item in value]
    if hasattr(value, "content"):
        return {
            "type": type(value).__name__,
            "content": _to_plain_data(getattr(value, "content")),
        }
    return repr(value)


def _summarize_step_data(data: Any) -> dict[str, Any]:
    """Reduce a LangGraph update payload to a stable tutorial summary.

    WHY: Raw step payloads can contain full message histories and provider
    objects. For learning, the useful signal is which node emitted the update
    and what high-level fields were present.
    """
    plain = _to_plain_data(data)
    if isinstance(plain, dict):
        message_count = 0
        messages = plain.get("messages")
        if isinstance(messages, list):
            message_count = len(messages)
        return {
            "keys": sorted(plain.keys()),
            "message_count": message_count,
        }
    return {"type": type(data).__name__, "value": str(plain)[:240]}


def _truncate(text: str, limit: int = 500) -> str:
    """Keep terminal output compact while preserving the main result.

    WHY: Real agent results and token streams can be long. The tutorial should
    remain scannable even when a provider returns a verbose answer.
    """
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _extract_result_text(result: dict[str, Any]) -> str:
    """Extract a human-readable final answer from common agent result shapes.

    WHY: Local and remote AgentService paths both return dictionaries, but the
    final message can be represented as plain dicts, message objects converted
    to dicts, or custom task payloads.
    """
    messages = result.get("messages")
    if isinstance(messages, list) and messages:
        last_message = messages[-1]
        if isinstance(last_message, dict):
            content = last_message.get("content")
            if isinstance(content, str):
                return content
            if content is not None:
                return json.dumps(content, ensure_ascii=True, indent=2)
    if "output" in result:
        return str(result["output"])
    if "reply" in result:
        return str(result["reply"])
    return json.dumps(result, ensure_ascii=True, indent=2)


def should_force_local_agent_execution(
    trace_mode: TraceMode,
    agent_execution_mode: AgentExecutionMode,
) -> bool:
    """Decide whether the inner AgentService must bypass Agent Server.

    WHY: Nested LangSmith traces require local callback propagation. When the
    configured agent runs through LangGraph Agent Server, execution crosses an
    HTTP boundary and cannot remain a child of the outer tutorial graph trace.
    """
    if agent_execution_mode == "local":
        return True
    if agent_execution_mode == "configured":
        return False
    return trace_mode != "separate"


def build_agentservice_config(
    agent_name: str,
    trace_mode: TraceMode,
    agent_execution_mode: AgentExecutionMode,
) -> Any | None:
    """Build an optional AgentService config override for trace correctness.

    WHY: The tutorial should demonstrate true nested tracing by default. If the
    YAML config marks the agent as remote and Agent Server is enabled, we create
    a copied LLMConfig with only this agent's ``execution_mode`` set to local.
    The repository YAML and global config singleton stay untouched.
    """
    if not should_force_local_agent_execution(trace_mode, agent_execution_mode):
        return None

    from inference_core.llm.config import get_llm_config

    return get_llm_config().with_overrides(
        agent_overrides={agent_name: {"execution_mode": "local"}}
    )


def prepare_request_node(state: TutorialState) -> dict[str, Any]:
    """Normalize user input before the graph crosses into AgentService.

    WHY: Keeping this as a separate LangGraph node makes the boundary visible:
    deterministic workflow preparation happens outside the agent, while the LLM
    reasoning remains inside AgentService.
    """
    prompt = (state.get("prompt") or DEFAULT_PROMPT).strip()
    if not prompt:
        raise ValueError("prompt cannot be empty")

    user_id = str(state.get("user_id") or DEFAULT_USER_ID)
    uuid.UUID(user_id)

    return {
        "normalized_prompt": prompt,
        "agent_name": state.get("agent_name") or DEFAULT_AGENT_NAME,
        "user_id": user_id,
        "session_id": state.get("session_id") or DEFAULT_SESSION_ID,
        "use_memory": bool(state.get("use_memory", False)),
        "trace_mode": state.get("trace_mode") or "separate",
        "agent_execution_mode": state.get("agent_execution_mode") or "auto",
        "agent_steps": [],
        "agent_tokens": [],
    }


async def mock_agent_runner(
    state: TutorialState,
    on_token: TokenCallback,
    on_step: StepCallback,
    parent_runnable_config: RunnableConfig | None,
) -> TutorialAgentRun:
    """Run a deterministic fake agent with the same callback contract.

    WHY: The tutorial should be executable before API keys, provider models, or
    llm_config.yaml are ready. This keeps the LangGraph integration shape real
    while replacing only the inner LLM call.
    """
    del parent_runnable_config

    prompt = state["normalized_prompt"]
    on_step(
        "mock_agent",
        {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "ai", "content": "Mock response prepared."},
            ]
        },
    )

    chunks = [
        "Mock AgentService bridge received the prompt: ",
        prompt,
        ". In the real path this node calls AgentService.create_agent() ",
        "and AgentService.arun_agent_steps().",
    ]
    for chunk in chunks:
        on_token(chunk, {"type": "text", "node": "mock_agent"})

    final_text = "".join(chunks)
    return TutorialAgentRun(
        result={
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "ai", "content": final_text},
            ],
            "mode": "mock",
        },
        cost_metrics={
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "model_call_count": 0,
        },
    )


async def run_agentservice_runner(
    state: TutorialState,
    on_token: TokenCallback,
    on_step: StepCallback,
    parent_runnable_config: RunnableConfig | None,
) -> TutorialAgentRun:
    """Call the real inference-core AgentService from inside a graph node.

    WHY: This is the bridge the tutorial exists to teach. LangGraph owns the
    outer deterministic workflow, while AgentService owns the LangChain agent,
    configured tools, middleware, cost tracking, memory, and optional remote
    Agent Server delegation.
    """
    from inference_core.services.agents_service import AgentService

    trace_mode = state.get("trace_mode", "separate")
    agent_execution_mode = state.get("agent_execution_mode", "auto")
    service = AgentService(
        agent_name=state["agent_name"],
        user_id=uuid.UUID(state["user_id"]),
        session_id=state["session_id"],
        use_memory=state.get("use_memory", False),
        config=build_agentservice_config(
            state["agent_name"],
            trace_mode,
            agent_execution_mode,
        ),
    )
    try:
        await service.create_agent()
        response = await service.arun_agent_steps(
            state["normalized_prompt"],
            on_token=on_token,
            on_step=on_step,
            parent_runnable_config=parent_runnable_config,
            trace_mode=trace_mode,
        )
        return TutorialAgentRun(
            result=_to_plain_data(response.result),
            cost_metrics=_to_plain_data(response.cost_metrics),
        )
    finally:
        service.close()


async def run_agentservice_node(
    state: TutorialState,
    *,
    agent_runner: AgentRunner,
    parent_runnable_config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Execute the selected agent runner and collect inner stream events.

    WHY: LangGraph nodes should exchange state updates, while AgentService uses
    callback-based streaming. This node adapts callbacks into graph state so the
    final report can show both layers of execution.
    """
    tokens: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []

    def on_token(text: str, meta: dict[str, Any]) -> None:
        tokens.append({"text": text, "meta": _to_plain_data(meta)})

    def on_step(name: str, data: Any) -> None:
        steps.append({"name": name, "summary": _summarize_step_data(data)})

    agent_run = await agent_runner(
        state,
        on_token,
        on_step,
        parent_runnable_config,
    )

    return {
        "agent_result": agent_run.result,
        "agent_steps": steps,
        "agent_tokens": tokens,
        "cost_metrics": agent_run.cost_metrics,
    }


def format_report_node(state: TutorialState) -> dict[str, Any]:
    """Build the final human-readable tutorial report.

    WHY: A tutorial should make the execution boundary obvious after the run,
    not require readers to infer it from logs or source code alone.
    """
    result = state.get("agent_result", {})
    tokens = state.get("agent_tokens", [])
    token_text = "".join(str(item.get("text", "")) for item in tokens)
    steps = state.get("agent_steps", [])

    lines = [
        "LangGraph -> AgentService tutorial report",
        "",
        "Outer graph flow:",
        "  START -> prepare_request -> run_core_agent -> format_report -> END",
        "",
        "Bridge point:",
        (
            "  AgentService("
            f"agent_name={state['agent_name']!r}, "
            f"use_memory={state.get('use_memory', False)}"
            ")"
        ),
        "  await service.create_agent()",
        "  await service.arun_agent_steps(..., on_token=..., on_step=...)",
        "",
        "Trace mode:",
        f"  {state.get('trace_mode', 'separate')}",
        "Agent execution mode:",
        f"  {state.get('agent_execution_mode', 'auto')}",
        "",
        "What AgentService still owns:",
        "  model resolution, LangChain create_agent(), tool loading, middleware,",
        "  cost tracking, memory, and optional execution_mode='remote' delegation.",
        "",
        "Agent result:",
        f"  {_truncate(_extract_result_text(result))}",
        "",
        f"Inner agent steps captured: {len(steps)}",
        f"Inner token chunks captured: {len(tokens)}",
        "",
        "Token preview:",
        f"  {_truncate(token_text) if token_text else '(no token callback output)'}",
        "",
        "Cost metrics:",
        f"  {json.dumps(state.get('cost_metrics'), ensure_ascii=True, sort_keys=True)}",
        "",
        "Agent Server comparison:",
        "  This script builds a custom outer StateGraph around AgentService.",
        "  agent_graphs.py and build_agent_graph() compile configured agents for",
        "  langgraph dev/up. true nested tracing requires local inner execution;",
        "  remote Agent Server runs cross an HTTP boundary and remain separate.",
    ]
    return {"report": "\n".join(lines)}


def build_tutorial_graph(agent_runner: AgentRunner | None = None) -> Any:
    """Build the outer LangGraph workflow used by the tutorial.

    WHY: Exposing graph construction as a function keeps the example runnable
    from the CLI and importable from tests without executing any LLM calls at
    import time.
    """
    selected_runner = agent_runner or run_agentservice_runner
    graph = StateGraph(TutorialState)

    async def run_core_agent(
        state: TutorialState,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        return await run_agentservice_node(
            state,
            agent_runner=selected_runner,
            parent_runnable_config=config,
        )

    graph.add_node("prepare_request", prepare_request_node)
    graph.add_node("run_core_agent", run_core_agent)
    graph.add_node("format_report", format_report_node)

    graph.add_edge(START, "prepare_request")
    graph.add_edge("prepare_request", "run_core_agent")
    graph.add_edge("run_core_agent", "format_report")
    graph.add_edge("format_report", END)

    return graph.compile()


def build_initial_state(args: argparse.Namespace) -> TutorialState:
    """Translate CLI arguments into the graph input state.

    WHY: Keeping CLI parsing separate from graph construction makes the graph
    reusable in unit tests, notebooks, and future integration examples.
    """
    return {
        "prompt": args.prompt,
        "agent_name": args.agent,
        "user_id": args.user_id,
        "session_id": args.session_id,
        "use_memory": args.use_memory,
        "mock_agent": args.mock_agent,
        "trace_mode": args.trace_mode,
        "agent_execution_mode": args.agent_execution_mode,
    }


def _print_intro(
    args: argparse.Namespace,
    argv: list[str] | None = None,
) -> None:
    """Print the tutorial framing before graph execution starts.

    WHY: The example is meant for learning, so the terminal output should name
    the two layers before any LangGraph updates appear.
    """
    mode = "mock AgentService double" if args.mock_agent else "real AgentService"
    input_argv = argv if argv is not None else sys.argv[1:]
    print("LangGraph outer graph + inference-core AgentService inner agent")
    print(f"Mode: {mode}")
    print("Input arguments:")
    print(f"  argv={input_argv!r}")
    print(f"  agent={args.agent!r}")
    print(f"  prompt={args.prompt!r}")
    print(f"  session_id={args.session_id!r}")
    print(f"  user_id={args.user_id!r}")
    print(f"  use_memory={args.use_memory!r}")
    print(f"  mock_agent={args.mock_agent!r}")
    print(f"  outer_stream={args.outer_stream!r}")
    print(f"  trace_mode={args.trace_mode!r}")
    print(f"  agent_execution_mode={args.agent_execution_mode!r}")
    print("Docs idea: LangChain agents provide the agent abstraction; LangGraph")
    print("provides the lower-level orchestration runtime around that agent.")
    print()


def _print_outer_update(update: dict[str, Any]) -> None:
    """Print a compact view of LangGraph outer stream updates.

    WHY: Outer graph streaming should show node progress without dumping raw
    message histories or provider payloads into the terminal.
    """
    for node_name, node_update in update.items():
        if isinstance(node_update, dict):
            changed = ", ".join(sorted(node_update.keys()))
        else:
            changed = type(node_update).__name__
        print(f"[outer graph] {node_name}: {changed}")


async def run_graph(
    graph: Any,
    initial_state: TutorialState,
    *,
    outer_stream: bool,
) -> TutorialState:
    """Run the tutorial graph with optional outer LangGraph streaming.

    WHY: The same graph should support a quiet final-result mode and a teaching
    mode where each outer LangGraph node update is visible.
    """
    if not outer_stream:
        return await graph.ainvoke(initial_state)

    final_state: TutorialState = dict(initial_state)
    async for update in graph.astream(initial_state, stream_mode="updates"):
        _print_outer_update(update)
        for node_update in update.values():
            if isinstance(node_update, dict):
                final_state.update(node_update)
    return final_state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the tutorial runner.

    WHY: The example should be easy to run in both mock and real modes without
    editing source files or repository configuration.
    """
    parser = argparse.ArgumentParser(
        description="Run a LangGraph workflow whose core node calls AgentService.",
    )
    parser.add_argument("--agent", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument(
        "--use-memory",
        action="store_true",
        help="Enable AgentService memory integration for the real agent path.",
    )
    parser.add_argument(
        "--mock-agent",
        action="store_true",
        help="Use a deterministic mock runner instead of calling a real LLM.",
    )
    parser.add_argument(
        "--outer-stream",
        action="store_true",
        help="Print LangGraph outer node updates while the graph runs.",
    )
    parser.add_argument(
        "--trace-mode",
        default="separate",
        choices=TRACE_MODE_CHOICES,
        help=(
            "separate: inner AgentService keeps its own trace root; "
            "nested: reuse outer RunnableConfig when available; "
            "nested-only: require parent trace callbacks and fail otherwise."
        ),
    )
    parser.add_argument(
        "--agent-execution-mode",
        default="auto",
        choices=AGENT_EXECUTION_MODE_CHOICES,
        help=(
            "auto: use configured execution for separate traces and force local "
            "for nested traces; configured: honor llm_config.yaml; local: always "
            "bypass Agent Server for this tutorial run."
        ),
    )
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    """Run the CLI tutorial entry point.

    WHY: Keeping the main flow async avoids event-loop nesting and mirrors the
    async AgentService usage recommended for FastAPI and LangGraph code.
    """
    args = parse_args(argv)
    _print_intro(args, argv)

    runner = mock_agent_runner if args.mock_agent else None
    graph = build_tutorial_graph(agent_runner=runner)
    final_state = await run_graph(
        graph,
        build_initial_state(args),
        outer_stream=args.outer_stream,
    )
    print()
    print(final_state["report"])
    return 0


def main() -> None:
    """Execute the async tutorial from a normal Python script entry point."""
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    print("Starting LangGraph AgentService tutorial...")

    main()

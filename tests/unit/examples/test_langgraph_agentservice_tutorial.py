from langchain_core.callbacks.base import BaseCallbackHandler

from examples.langgraph_agentservice_tutorial import (
    TutorialAgentRun,
    build_tutorial_graph,
    mock_agent_runner,
    run_graph,
    should_force_local_agent_execution,
)


async def test_mock_tutorial_graph_produces_report():
    graph = build_tutorial_graph(agent_runner=mock_agent_runner)

    final_state = await graph.ainvoke(
        {
            "prompt": "Explain the bridge",
            "agent_name": "default_agent",
            "user_id": "00000000-0000-0000-0000-000000000001",
            "session_id": "test-session",
            "use_memory": False,
            "mock_agent": True,
        }
    )

    assert "LangGraph -> AgentService tutorial report" in final_state["report"]
    assert final_state["agent_result"]["mode"] == "mock"
    assert final_state["agent_steps"][0]["name"] == "mock_agent"
    assert final_state["agent_tokens"]


async def test_tutorial_graph_accepts_injected_agent_runner():
    async def fake_runner(state, on_token, on_step, parent_runnable_config):
        assert parent_runnable_config is not None
        assert parent_runnable_config["metadata"]["langgraph_node"] == "run_core_agent"
        on_step("fake_inner_agent", {"messages": [{"role": "ai"}]})
        on_token("Injected runner output", {"type": "text", "node": "fake"})
        return TutorialAgentRun(
            result={"reply": f"handled by {state['agent_name']}"},
            cost_metrics={"total_tokens": 0},
        )

    graph = build_tutorial_graph(agent_runner=fake_runner)
    final_state = await graph.ainvoke(
        {
            "prompt": "Use an injected runner",
            "agent_name": "custom_agent",
            "user_id": "00000000-0000-0000-0000-000000000001",
            "session_id": "test-session",
        }
    )

    assert final_state["agent_result"] == {"reply": "handled by custom_agent"}
    assert final_state["agent_steps"] == [
        {
            "name": "fake_inner_agent",
            "summary": {"keys": ["messages"], "message_count": 1},
        }
    ]


async def test_tutorial_graph_passes_parent_config_and_trace_mode():
    seen = {}

    class _TestCallbackHandler(BaseCallbackHandler):
        run_inline = True

    async def fake_runner(state, on_token, on_step, parent_runnable_config):
        seen["trace_mode"] = state["trace_mode"]
        seen["parent_runnable_config"] = parent_runnable_config
        on_step("fake_inner_agent", {"messages": [{"role": "ai"}]})
        on_token("Nested output", {"type": "text", "node": "fake"})
        return TutorialAgentRun(result={"reply": "nested ok"})

    graph = build_tutorial_graph(agent_runner=fake_runner)
    await graph.ainvoke(
        {
            "prompt": "Use nested tracing",
            "agent_name": "custom_agent",
            "user_id": "00000000-0000-0000-0000-000000000001",
            "session_id": "test-session",
            "trace_mode": "nested",
        },
        config={
            "callbacks": [_TestCallbackHandler()],
            "configurable": {"thread_id": "outer-thread"},
        },
    )

    assert seen["trace_mode"] == "nested"
    callback_manager = seen["parent_runnable_config"]["callbacks"]
    assert hasattr(callback_manager, "handlers")
    assert any(
        isinstance(handler, _TestCallbackHandler)
        for handler in callback_manager.handlers
    )
    assert seen["parent_runnable_config"]["configurable"]["thread_id"] == "outer-thread"


def test_agent_execution_mode_auto_forces_local_only_for_nested_traces():
    assert should_force_local_agent_execution("separate", "auto") is False
    assert should_force_local_agent_execution("nested", "auto") is True
    assert should_force_local_agent_execution("nested-only", "auto") is True
    assert should_force_local_agent_execution("nested", "configured") is False
    assert should_force_local_agent_execution("separate", "local") is True


async def test_run_graph_outer_stream_merges_final_state(capsys):
    graph = build_tutorial_graph(agent_runner=mock_agent_runner)

    final_state = await run_graph(
        graph,
        {
            "prompt": "Stream outer updates",
            "agent_name": "default_agent",
            "user_id": "00000000-0000-0000-0000-000000000001",
            "session_id": "test-session",
            "mock_agent": True,
        },
        outer_stream=True,
    )

    captured = capsys.readouterr()
    assert "[outer graph] prepare_request" in captured.out
    assert "[outer graph] run_core_agent" in captured.out
    assert "report" in final_state

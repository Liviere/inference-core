"""How streamed chunks of a tool call reach ``on_token``.

By default the tool's name is a ``calling `name``` reasoning line, as it
always was. With ``tool_call_token_type="tool_call"`` it is a segment of its
own type, for a client that shows tool calls apart from the reasoning.
"""

from langchain_core.messages import AIMessageChunk

from inference_core.services.agents_service import AgentService, DeepAgentService

META = {"langgraph_node": "model", "lc_agent_name": "helper"}


def _chunk() -> AIMessageChunk:
    return AIMessageChunk(
        content=[
            {"type": "reasoning", "reasoning": "Look it up."},
            {"type": "text", "text": "Checking."},
        ],
        tool_call_chunks=[
            {"name": "web_search", "args": '{"q": "x"}', "id": "c1", "index": 0}
        ],
    )


def test_by_default_a_tool_call_is_a_line_of_reasoning():
    segments = AgentService._extract_message_segments(_chunk(), META)

    assert ("calling `web_search`\n", "reasoning") in [
        (text, meta["type"]) for text, meta in segments
    ]
    assert all(meta["type"] != "tool_call" for _, meta in segments)


def test_as_its_own_type_a_tool_call_carries_the_name_only():
    segments = AgentService._extract_message_segments(
        _chunk(), META, tool_call_type="tool_call"
    )

    tool_calls = [
        (text, meta) for text, meta in segments if meta["type"] == "tool_call"
    ]
    assert tool_calls == [
        (
            "web_search",
            {
                "node": "model",
                "agent_name": "helper",
                "type": "tool_call",
                "tool_name": "web_search",
            },
        )
    ]
    # The arguments never reach the stream, and nothing else changes.
    assert all("q" not in text for text, _ in segments)
    unchanged = [
        (text, meta)
        for text, meta in AgentService._extract_message_segments(_chunk(), META)
        if not text.startswith("calling")
    ]
    assert [
        (text, meta) for text, meta in segments if meta["type"] != "tool_call"
    ] == unchanged


def test_the_service_forwards_its_choice():
    received: list[tuple[str, dict]] = []
    service = AgentService.__new__(AgentService)
    service._tool_call_token_type = "tool_call"
    service._process_message_chunk(
        _chunk(),
        META,
        lambda text, meta: received.append((text, meta)),
        tool_call_type=service._tool_call_token_type,
    )
    assert ("web_search", "tool_call") in [(t, m["type"]) for t, m in received]


def test_the_default_of_both_services_is_unchanged():
    import inspect

    for cls in (AgentService, DeepAgentService):
        default = inspect.signature(cls.__init__).parameters["tool_call_token_type"]
        assert default.default == "reasoning"

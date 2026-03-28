"""
LangGraph Agent Server client for remote agent execution.

WHY: Phase 1 of the LangGraph Platform migration — delegates agent runs
to an external Agent Server instead of executing them in-process.  This
keeps the existing Celery/Redis/WebSocket infrastructure intact while
gaining LangGraph Studio observability and managed graph execution.

The client wraps langgraph-sdk and translates between inference-core's
AgentResponse format and the Agent Server's REST/streaming API.
"""

import logging
from typing import Any, Callable, Optional

from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient

from inference_core.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


_client_instance: Optional[LangGraphClient] = None


def get_agent_server_client(settings: Optional[Settings] = None) -> LangGraphClient:
    """Return a singleton LangGraph SDK client.

    Reuse a single HTTP connection pool across all remote agent
    calls.  The client is stateless and thread-safe.
    """
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    settings = settings or get_settings()
    if not settings.agent_server_url:
        raise RuntimeError(
            "AGENT_SERVER_URL is not configured.  Set it in .env or disable "
            "remote execution (AGENT_SERVER_ENABLED=False)."
        )

    _client_instance = get_client(
        url=settings.agent_server_url,
        api_key=settings.agent_server_api_key,
        timeout=settings.agent_server_timeout,
    )
    logger.info(
        "Initialized LangGraph Agent Server client → %s",
        settings.agent_server_url,
    )
    return _client_instance


def reset_agent_server_client() -> None:
    """Reset the singleton client (useful for testing)."""
    global _client_instance
    _client_instance = None


def _resolve_graph_id(agent_name: str, remote_graph_id: Optional[str]) -> str:
    """Determine the graph ID to use on the Agent Server."""
    return remote_graph_id or agent_name


def _build_config(
    checkpoint_config: Optional[dict[str, Any]],
    metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Build the ``config`` dict for an Agent Server run.

    WHY: Middleware on the Agent Server resolves per-request context
    (user_id, session_id, …) from ``runtime.configurable``.  We merge
    checkpoint keys *and* user-level keys from metadata into a single
    configurable block so both checkpointing and middleware work.
    """
    configurable: dict[str, Any] = {}

    if checkpoint_config:
        configurable.update(
            {k: v for k, v in checkpoint_config.items() if k != "thread_id"}
        )

    # Forward middleware-relevant keys from metadata → configurable
    _MW_KEYS = (
        "user_id",
        "session_id",
        "request_id",
        "instance_id",
        "instance_name",
        # Instance-level overrides (model, prompt)
        "primary_model",
        "system_prompt_override",
        "system_prompt_append",
        # Reasoning output toggle (InstanceConfigMiddleware reads this)
        "reasoning_output",
        # Per-subagent overrides (SubagentConfigMiddleware reads these)
        "subagent_configs",
    )
    if metadata:
        for key in _MW_KEYS:
            if key in metadata:
                configurable[key] = metadata[key]

    if configurable:
        logger.debug(
            "_build_config: configurable keys=%s, primary_model=%r, "
            "subagent_configs=%s",
            list(configurable.keys()),
            configurable.get("primary_model"),
            (
                list(configurable["subagent_configs"].keys())
                if "subagent_configs" in configurable
                else "NONE"
            ),
        )

    return {"configurable": configurable} if configurable else {}


async def run_remote(
    *,
    agent_name: str,
    remote_graph_id: Optional[str] = None,
    user_input: str,
    thread_id: Optional[str] = None,
    checkpoint_config: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Execute an agent on the remote Agent Server and wait for the result.

    WHY: Synchronous (non-streaming) remote execution path.  Used by
    `AgentService.arun_agent_steps` when `execution_mode='remote'`.

    Returns the raw response dict from the Agent Server.
    """
    client = get_agent_server_client()
    graph_id = _resolve_graph_id(agent_name, remote_graph_id)

    # Reuse existing thread or create a new one for stateless runs
    effective_thread_id = thread_id
    if effective_thread_id is None and checkpoint_config:
        effective_thread_id = checkpoint_config.get("thread_id")

    run_metadata = dict(metadata or {})
    run_metadata["source"] = "inference_core"
    run_metadata["agent_name"] = agent_name

    config = _build_config(checkpoint_config, metadata)

    logger.info(
        "Remote run → graph=%s thread=%s",
        graph_id,
        effective_thread_id or "(stateless)",
    )

    result = await client.runs.wait(
        thread_id=effective_thread_id,
        assistant_id=graph_id,
        input={"messages": [{"role": "user", "content": user_input}]},
        metadata=run_metadata,
        config=config or None,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )

    return result


async def stream_remote(
    *,
    agent_name: str,
    remote_graph_id: Optional[str] = None,
    user_input: str,
    thread_id: Optional[str] = None,
    checkpoint_config: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
    on_token: Optional[Callable[[str, dict[str, Any]], None]] = None,
    on_step: Optional[Callable[[str, Any], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Stream an agent run from the remote Agent Server.

    Streaming remote execution path.  Translates Agent Server SSE
    events into the same `on_token` / `on_step` callback interface used
    by local execution.

    Returns the final result dict after the stream completes.
    """
    from inference_core.services._cancel import AgentCancelled

    client = get_agent_server_client()
    graph_id = _resolve_graph_id(agent_name, remote_graph_id)

    effective_thread_id = thread_id
    if effective_thread_id is None and checkpoint_config:
        effective_thread_id = checkpoint_config.get("thread_id")

    run_metadata = dict(metadata or {})
    run_metadata["source"] = "inference_core"
    run_metadata["agent_name"] = agent_name

    config = _build_config(checkpoint_config, metadata)

    # Use messages+updates streaming for token-level + step-level events
    stream_mode = ["messages", "updates"] if on_token else "updates"

    logger.info(
        "Remote stream → graph=%s thread=%s mode=%s",
        graph_id,
        effective_thread_id or "(stateless)",
        stream_mode,
    )

    last_result: dict[str, Any] = {}
    run_id: Optional[str] = None

    async for event in client.runs.stream(
        thread_id=effective_thread_id,
        assistant_id=graph_id,
        input={"messages": [{"role": "user", "content": user_input}]},
        metadata=run_metadata,
        config=config or None,
        stream_mode=stream_mode,
        stream_subgraphs=True,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        version="v2",
    ):
        # v2 events are TypedDicts with ["type"], ["data"], ["ns"]
        event_type: str = event["type"]
        data = event["data"]
        ns: list[str] = event.get("ns", [])

        # Check cancellation
        if cancel_check:
            try:
                if cancel_check():
                    if run_id:
                        try:
                            await client.runs.cancel(effective_thread_id, run_id)
                        except Exception:
                            pass
                    raise AgentCancelled(
                        "Agent execution cancelled by user",
                        partial_result=last_result or None,
                    )
            except AgentCancelled:
                raise
            except Exception:
                pass

        # Extract run_id from metadata event for potential cancellation
        if event_type == "metadata" and isinstance(data, dict):
            run_id = data.get("run_id", run_id)

        if event_type == "messages/partial" and on_token:
            _forward_message_event(data, on_token, ns=ns)
        elif event_type == "updates" and on_step:
            _forward_step_event(data, on_step, ns=ns)

        # Track the latest result
        if event_type in ("values", "updates") and isinstance(data, dict):
            if "messages" in data:
                last_result = data
            elif isinstance(data, dict):
                for node_name, node_data in data.items():
                    if isinstance(node_data, dict) and "messages" in node_data:
                        last_result = node_data

    return last_result


def _forward_message_event(
    data: Any,
    on_token: Callable[[str, dict[str, Any]], None],
    *,
    ns: list[str] | None = None,
) -> None:
    """Translate Agent Server message events to on_token callbacks.

    WHY: The Agent Server emits messages/partial events with a list of
    message dicts.  We extract text content and forward it in the same
    format that local AgentService uses.

    ``ns`` carries the subgraph namespace path from v2 streaming so
    consumers can distinguish parent vs subagent tokens.
    """
    if not isinstance(data, list):
        return

    for msg in data:
        if not isinstance(msg, dict):
            continue
        msg_type = msg.get("type", "")
        if msg_type != "ai":
            continue

        content = msg.get("content", "")
        node = msg.get("name", "agent")
        meta: dict[str, Any] = {"node": node, "type": "text"}
        if ns:
            meta["ns"] = ns

        # Handle string content
        if isinstance(content, str) and content:
            on_token(content, meta)
        # Handle structured content blocks
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "text")
                    if block_type == "text" and block.get("text"):
                        on_token(block["text"], {**meta, "type": "text"})
                    elif block_type == "thinking" and block.get("thinking"):
                        on_token(block["thinking"], {**meta, "type": "reasoning"})


def _forward_step_event(
    data: Any,
    on_step: Callable[[str, Any], None],
    *,
    ns: list[str] | None = None,
) -> None:
    """Translate Agent Server update events to on_step callbacks.

    ``ns`` carries the subgraph namespace path from v2 streaming.
    """
    if isinstance(data, dict):
        for step_name, step_data in data.items():
            try:
                if ns:
                    step_data = (
                        {**step_data, "ns": ns}
                        if isinstance(step_data, dict)
                        else step_data
                    )
                on_step(step_name, step_data)
            except Exception:
                pass

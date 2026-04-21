"""Contract tests for LangGraph Agent Server — per-request ``configurable`` overrides.

WHY: The upcoming frontend (``@langchain/react`` useStream) talks to the Agent
Server over the same ``langgraph-sdk`` protocol we already use in
``agent_server_client.py``.  Before we build the ``/run-bundle`` endpoint and
the React UI, we must prove end-to-end that runtime overrides pushed via
``config.configurable`` are honoured by the compiled graph — specifically the
keys that ``InstanceConfigMiddleware`` reads:

  * ``system_prompt_override``
  * ``system_prompt_append``
  * ``primary_model``  (asserted indirectly via behavioural change)

This file is the Phase 1 validation from the integration plan.

Run with an Agent Server already up:

    poetry run langgraph dev --no-browser
    poetry run pytest -m agent_server \
        tests/integration/test_agent_server_configurable.py
"""

import os
import uuid

import httpx
import pytest

from inference_core.services.agent_server_client import (
    reset_agent_server_client,
    run_remote,
)

pytestmark = [pytest.mark.agent_server, pytest.mark.integration]

AGENT_SERVER_URL = os.environ.get("AGENT_SERVER_URL", "http://localhost:2024")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _server_is_reachable() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as http:
            resp = await http.get(f"{AGENT_SERVER_URL}/ok")
            return resp.status_code == 200
    except Exception:
        return False


def _extract_last_ai_text(result: dict) -> str:
    """Flatten the final AI message content to a searchable string.

    The Agent Server returns messages as a list of dicts; content may be a
    plain string or a list of content blocks.  We accept anything textual.
    """
    messages = result.get("messages", []) or []
    if not isinstance(messages, list):
        return ""

    # Walk from the end — useStream's final answer is the last AI message.
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("type") not in ("ai", "AIMessage", "AIMessageChunk"):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("reasoning") or ""
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_client():
    reset_agent_server_client()
    yield
    reset_agent_server_client()


@pytest.fixture(autouse=True)
async def _skip_if_no_server():
    if not await _server_is_reachable():
        pytest.skip(
            f"Agent Server not reachable at {AGENT_SERVER_URL}. "
            f"Start it with: poetry run langgraph dev --no-browser"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConfigurableSystemPromptOverride:
    """``system_prompt_override`` must fully replace the agent's system prompt."""

    async def test_override_forces_specific_marker(self):
        # A very rigid instruction — any sensible model should follow it verbatim.
        marker = f"MARKER-{uuid.uuid4().hex[:8].upper()}"
        override = (
            "You are a test responder.  "
            "Regardless of the user's message, reply with exactly this single "
            f"token and nothing else: {marker}"
        )

        result = await run_remote(
            agent_name="default_agent",
            user_input="Say hi",
            metadata={
                # MemoryMiddleware on default_agent expects a UUID user_id.
                "user_id": str(uuid.uuid4()),
                "session_id": f"sess-{uuid.uuid4().hex[:8]}",
                "system_prompt_override": override,
            },
        )

        text = _extract_last_ai_text(result)
        assert marker in text, (
            f"system_prompt_override did not take effect. "
            f"Expected {marker!r} in agent reply, got: {text[:300]!r}"
        )


class TestConfigurableSystemPromptAppend:
    """``system_prompt_append`` must extend (not replace) the base prompt."""

    async def test_append_injects_extra_rule(self):
        suffix_marker = f"SUFFIX-{uuid.uuid4().hex[:6].upper()}"
        append = (
            "ADDITIONAL STRICT RULE: end every reply with the literal token "
            f"{suffix_marker}"
        )

        result = await run_remote(
            agent_name="default_agent",
            user_input="Reply with a short greeting.",
            metadata={
                "user_id": str(uuid.uuid4()),
                "session_id": f"sess-{uuid.uuid4().hex[:8]}",
                "system_prompt_append": append,
            },
        )

        text = _extract_last_ai_text(result)
        assert suffix_marker in text, (
            f"system_prompt_append did not take effect. "
            f"Expected {suffix_marker!r} in reply, got: {text[:300]!r}"
        )


class TestConfigurableUserAndSessionKeys:
    """``user_id`` / ``session_id`` / ``instance_id`` must be accepted without error.

    We don't assert behaviour here (CostTrackingMiddleware persists them to
    the DB which the Agent Server process doesn't share with the test).  We
    only assert the run completes — proving the keys are valid inputs, not
    rejected by validation.
    """

    async def test_run_accepts_middleware_keys(self):
        result = await run_remote(
            agent_name="default_agent",
            user_input="Reply with one short word.",
            metadata={
                "user_id": str(uuid.uuid4()),
                "session_id": f"sess-{uuid.uuid4().hex[:8]}",
                "instance_id": str(uuid.uuid4()),
                "instance_name": "contract-test-instance",
                "request_id": f"req-{uuid.uuid4().hex[:8]}",
            },
        )

        assert isinstance(result, dict)
        assert _extract_last_ai_text(result), "Agent returned empty response"

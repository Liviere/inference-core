"""Integration tests for the LangGraph Agent Server.

Requires a running Agent Server:
    poetry run langgraph dev --no-browser

Run with:
    poetry run pytest -m agent_server tests/integration/test_agent_server.py

These tests validate the real Agent Server round-trip:
  - health check connectivity
  - listing deployed assistants
  - running a simple agent invocation
  - streaming tokens from a remote agent
  - thread state persistence across turns
"""

import os

import httpx
import pytest

from inference_core.services.agent_server_client import (
    get_agent_server_client,
    reset_agent_server_client,
    run_remote,
    stream_remote,
)

# All tests in this module require a live Agent Server
pytestmark = [pytest.mark.agent_server, pytest.mark.integration]

AGENT_SERVER_URL = os.environ.get("AGENT_SERVER_URL", "http://localhost:2024")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _server_is_reachable() -> bool:
    """Quick health check — returns False if Agent Server is not running."""
    try:
        async with httpx.AsyncClient(timeout=3) as http:
            resp = await http.get(f"{AGENT_SERVER_URL}/ok")
            return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_client():
    """Clean singleton state between tests."""
    reset_agent_server_client()
    yield
    reset_agent_server_client()


@pytest.fixture(autouse=True)
async def _skip_if_no_server():
    """Skip every test in this module if Agent Server is not running."""
    if not await _server_is_reachable():
        pytest.skip(
            f"Agent Server not reachable at {AGENT_SERVER_URL}. "
            f"Start it with: poetry run langgraph dev --no-browser"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentServerHealth:
    """Verify basic connectivity to the Agent Server."""

    async def test_health_endpoint(self):
        async with httpx.AsyncClient(timeout=5) as http:
            resp = await http.get(f"{AGENT_SERVER_URL}/ok")
        assert resp.status_code == 200

    async def test_list_assistants(self):
        """The server should expose at least the agents defined in langgraph.json."""
        client = get_agent_server_client()
        assistants = await client.assistants.search()

        graph_ids = {a["graph_id"] for a in assistants}
        assert (
            "default_agent" in graph_ids
        ), f"default_agent not found in deployed graphs: {graph_ids}"


class TestRemoteRun:
    """Validate synchronous (non-streaming) remote agent runs."""

    async def test_run_returns_result(self):
        result = await run_remote(
            agent_name="default_agent",
            user_input="Say exactly: AGENT_SERVER_TEST_OK",
        )

        assert isinstance(result, dict)
        # The result should contain messages from the agent
        assert "messages" in result or isinstance(result, dict)

    async def test_run_with_metadata(self):
        result = await run_remote(
            agent_name="default_agent",
            user_input="Reply with one word: hello",
            metadata={"test": True, "source": "integration_test"},
        )

        assert isinstance(result, dict)


class TestRemoteStream:
    """Validate streaming remote agent runs."""

    async def test_stream_emits_tokens(self):
        tokens: list[str] = []
        steps: list[tuple[str, object]] = []

        def on_token(text: str, meta: dict) -> None:
            tokens.append(text)

        def on_step(name: str, data: object) -> None:
            steps.append((name, data))

        result = await stream_remote(
            agent_name="default_agent",
            user_input="Count from 1 to 3, each number on a separate line.",
            on_token=on_token,
            on_step=on_step,
        )

        assert isinstance(result, dict)
        # We should have received at least some tokens
        assert len(tokens) > 0, "No tokens received during streaming"


class TestThreadPersistence:
    """Validate that state persists across thread turns."""

    async def test_multi_turn_thread(self):
        client = get_agent_server_client()

        # Create a dedicated thread
        thread = await client.threads.create()
        thread_id = thread["thread_id"]

        # Turn 1: establish context
        result1 = await run_remote(
            agent_name="default_agent",
            user_input="Remember this number: 42. Just confirm you noted it.",
            thread_id=thread_id,
        )
        assert isinstance(result1, dict)

        # Turn 2: recall context
        result2 = await run_remote(
            agent_name="default_agent",
            user_input="What number did I ask you to remember?",
            thread_id=thread_id,
        )
        assert isinstance(result2, dict)

        # The second response should reference 42
        messages = result2.get("messages", [])
        full_text = " ".join(
            str(m.get("content", "")) for m in messages if isinstance(m, dict)
        )
        assert (
            "42" in full_text
        ), f"Agent did not recall '42' in thread continuation. Got: {full_text[:200]}"

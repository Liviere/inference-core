"""Tests for middleware _runtime_context — task-local context vars.

Covers:
- populate_from_configurable sets all vars
- getters return defaults when not populated
- clear resets all vars
- Concurrent async tasks have isolated contexts
"""

import asyncio
import uuid

import pytest

from inference_core.agents.middleware._runtime_context import (
    clear,
    get_instance_id,
    get_instance_name,
    get_memory_session_context_enabled,
    get_memory_tool_instructions_enabled,
    get_primary_model,
    get_request_id,
    get_session_id,
    get_system_prompt_append,
    get_system_prompt_override,
    get_user_id,
    populate_from_configurable,
)


@pytest.fixture(autouse=True)
def _clean_context():
    """Ensure context is clean before and after each test."""
    clear()
    yield
    clear()


class TestPopulateFromConfigurable:
    def test_sets_all_fields(self):
        uid = uuid.uuid4()
        iid = uuid.uuid4()
        populate_from_configurable(
            {
                "user_id": str(uid),
                "session_id": "sess-1",
                "request_id": "req-1",
                "instance_id": str(iid),
                "instance_name": "my-instance",
                "primary_model": "claude-haiku-4-5-20251001",
                "system_prompt_override": "You are a coder.",
                "system_prompt_append": "Be concise.",
            }
        )
        assert get_user_id() == uid
        assert get_session_id() == "sess-1"
        assert get_request_id() == "req-1"
        assert get_instance_id() == iid
        assert get_instance_name() == "my-instance"
        assert get_primary_model() == "claude-haiku-4-5-20251001"
        assert get_system_prompt_override() == "You are a coder."
        assert get_system_prompt_append() == "Be concise."

    def test_accepts_uuid_objects_directly(self):
        uid = uuid.uuid4()
        populate_from_configurable({"user_id": uid})
        assert get_user_id() == uid

    def test_ignores_missing_keys(self):
        populate_from_configurable({"user_id": str(uuid.uuid4())})
        assert get_session_id() is None
        assert get_request_id() is None

    def test_empty_configurable(self):
        populate_from_configurable({})
        assert get_user_id() is None


class TestDefaults:
    def test_all_getters_return_none(self):
        assert get_user_id() is None
        assert get_session_id() is None
        assert get_request_id() is None
        assert get_instance_id() is None
        assert get_instance_name() is None
        assert get_primary_model() is None
        assert get_system_prompt_override() is None
        assert get_system_prompt_append() is None


class TestClear:
    def test_resets_all(self):
        populate_from_configurable(
            {
                "user_id": str(uuid.uuid4()),
                "session_id": "s",
            }
        )
        clear()
        assert get_user_id() is None
        assert get_session_id() is None


class TestAsyncIsolation:
    async def test_concurrent_tasks_isolated(self):
        """Each async task should see its own contextvar values."""
        uid_a = uuid.uuid4()
        uid_b = uuid.uuid4()
        results = {}

        async def task_a():
            populate_from_configurable({"user_id": str(uid_a)})
            await asyncio.sleep(0.01)
            results["a"] = get_user_id()

        async def task_b():
            populate_from_configurable({"user_id": str(uid_b)})
            await asyncio.sleep(0.01)
            results["b"] = get_user_id()

        await asyncio.gather(task_a(), task_b())

        assert results["a"] == uid_a
        assert results["b"] == uid_b


# ---------------------------------------------------------------------------
# Memory configuration context vars
# ---------------------------------------------------------------------------


class TestMemoryConfigContextVars:
    """Verify memory_session_context_enabled and memory_tool_instructions_enabled."""

    def test_defaults_are_none(self):
        """Both memory config getters return None by default."""
        assert get_memory_session_context_enabled() is None
        assert get_memory_tool_instructions_enabled() is None

    def test_populate_sets_memory_session_context(self):
        """populate_from_configurable sets memory_session_context_enabled."""
        populate_from_configurable({"memory_session_context_enabled": False})
        assert get_memory_session_context_enabled() is False

    def test_populate_sets_memory_tool_instructions(self):
        """populate_from_configurable sets memory_tool_instructions_enabled."""
        populate_from_configurable({"memory_tool_instructions_enabled": True})
        assert get_memory_tool_instructions_enabled() is True

    def test_populate_coerces_to_bool(self):
        """Non-bool values are coerced to bool."""
        populate_from_configurable({"memory_session_context_enabled": 0})
        assert get_memory_session_context_enabled() is False

    def test_clear_resets_memory_vars(self):
        """clear() resets memory config vars to None."""
        populate_from_configurable(
            {
                "memory_session_context_enabled": True,
                "memory_tool_instructions_enabled": False,
            }
        )
        clear()
        assert get_memory_session_context_enabled() is None
        assert get_memory_tool_instructions_enabled() is None

    def test_missing_keys_leave_default(self):
        """Missing memory keys leave context vars as None."""
        populate_from_configurable({"user_id": str(uuid.uuid4())})
        assert get_memory_session_context_enabled() is None
        assert get_memory_tool_instructions_enabled() is None

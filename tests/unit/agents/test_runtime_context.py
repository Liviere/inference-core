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
    get_request_id,
    get_session_id,
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
            }
        )
        assert get_user_id() == uid
        assert get_session_id() == "sess-1"
        assert get_request_id() == "req-1"
        assert get_instance_id() == iid
        assert get_instance_name() == "my-instance"

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

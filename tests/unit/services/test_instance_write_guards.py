"""The checks an application adds before an instance is changed or deleted."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_core.database.sql.models.user_agent_instance import UserAgentInstance
from inference_core.services.user_agent_instance_service import (
    UserAgentInstanceService,
    _instance_list_filters,
    _instance_write_guards,
    register_instance_list_filter,
    register_instance_write_guard,
)

USER = uuid.UUID("11111111-1111-1111-1111-111111111111")


@pytest.fixture(autouse=True)
def clean_registries():
    guards, filters = list(_instance_write_guards), list(_instance_list_filters)
    _instance_write_guards.clear()
    _instance_list_filters.clear()
    yield
    _instance_write_guards[:] = guards
    _instance_list_filters[:] = filters


def _service(instance):
    result = MagicMock()
    result.scalar_one_or_none.return_value = instance
    result.scalar_one.return_value = instance
    result.scalars.return_value.all.return_value = [instance]
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    db.delete = AsyncMock()
    return UserAgentInstanceService(db, MagicMock()), db


def _instance():
    return SimpleNamespace(
        id=uuid.uuid4(),
        instance_name="helper",
        base_agent_name="assistant_agent",
        primary_model=None,
        is_deepagent=False,
        is_default=False,
        skills=None,
        subagents=[],
        config_overrides={},
        display_name="Helper",
    )


@pytest.mark.asyncio
async def test_without_a_guard_an_instance_is_changed_and_deleted_as_ever():
    instance = _instance()
    service, db = _service(instance)

    await service.update_instance(USER, instance.id, display_name="New")
    assert instance.display_name == "New"

    assert await service.delete_instance(USER, instance.id) is True
    db.delete.assert_awaited_once_with(instance)


@pytest.mark.asyncio
async def test_a_guard_sees_the_fields_of_an_update_and_can_refuse_it():
    instance = _instance()
    service, db = _service(instance)
    seen = []

    async def guard(_db, user_id, target, updates):
        seen.append((user_id, target, updates))
        raise ValueError("Not this one.")

    register_instance_write_guard(guard)

    with pytest.raises(ValueError, match="Not this one."):
        await service.update_instance(USER, instance.id, display_name="New")

    assert seen == [(USER, instance, {"display_name": "New"})]
    assert instance.display_name == "Helper"
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_guard_is_asked_with_none_before_a_deletion_and_can_refuse_it():
    instance = _instance()
    service, db = _service(instance)
    seen = []

    async def guard(_db, _user_id, _target, updates):
        seen.append(updates)
        raise ValueError("Not this one.")

    register_instance_write_guard(guard)

    with pytest.raises(ValueError, match="Not this one."):
        await service.delete_instance(USER, instance.id)

    assert seen == [None]
    db.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_missing_instance_asks_no_guard():
    service, _db = _service(None)
    guard = AsyncMock()
    register_instance_write_guard(guard)

    assert await service.update_instance(USER, uuid.uuid4(), display_name="N") is None
    assert await service.delete_instance(USER, uuid.uuid4()) is False
    guard.assert_not_awaited()


def test_registering_twice_is_registering_once():
    async def guard(*_args):
        return None

    def list_filter(_user_id):
        return []

    for _ in range(2):
        register_instance_write_guard(guard)
        register_instance_list_filter(list_filter)

    assert _instance_write_guards == [guard]
    assert _instance_list_filters == [list_filter]


@pytest.mark.asyncio
async def test_a_list_filter_adds_its_criteria_to_the_list_only():
    instance = _instance()
    service, db = _service(instance)
    asked = []

    def list_filter(user_id):
        asked.append(user_id)
        return [UserAgentInstance.primary_model.is_(None)]

    await service.list_instances(USER)
    no_filter_sql = str(db.execute.await_args.args[0])

    register_instance_list_filter(list_filter)
    await service.list_instances(USER)
    filtered_sql = str(db.execute.await_args.args[0])

    assert asked == [USER]
    assert "primary_model IS NULL" not in no_filter_sql
    assert "primary_model IS NULL" in filtered_sql

    await service.get_instance(USER, instance.id)
    assert asked == [USER]

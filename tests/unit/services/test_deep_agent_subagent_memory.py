"""A subagent's memory: its own, none, or a copy of its parent's.

The parent's MemoryMiddleware keeps per-run state. Handed to every subagent
as the same object, a subagent's recall replaced what the parent had
recalled, and each task ended with an analysis of its own.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.store.memory import InMemoryStore

from inference_core.agents.middleware.memory import (
    MemoryMiddleware,
    MemoryTurnWritesMiddleware,
    PostRunAnalysisOptions,
)
from inference_core.services.agent_memory_service import (
    AgentMemoryStoreService,
    MemoryOptions,
    MemorySpace,
    TurnWriteLog,
)
from inference_core.services.agents_service import (
    INHERIT_MEMORY_KEY,
    _propagates_to_subagents,
)
from inference_core.services.user_agent_instance_service import (
    UserAgentInstanceService,
    _instance_deleted_hooks,
    _overrides_validators,
    _run_overrides_validators,
    register_config_overrides_validator,
    register_instance_deleted_hook,
)
from tests.unit.services.test_deep_agent_task_brief import _deep_service, _Scripted


def _parent_memory() -> MemoryMiddleware:
    service = AgentMemoryStoreService(store=InMemoryStore(), scope="shared")
    return MemoryMiddleware(memory_service=service, user_id="u1", postrun_analysis=True)


def _spec(**extra):
    return {
        "name": "writer",
        "description": "Writes.",
        "system_prompt": "You write.",
        "model": _Scripted(replies=[]),
        "tools": [],
        **extra,
    }


def _memory_of(spec) -> list:
    return [m for m in spec["middleware"] if isinstance(m, MemoryMiddleware)]


@pytest.mark.asyncio
async def test_a_subagent_gets_a_copy_of_its_parents_memory():
    service = _deep_service(_Scripted(replies=[]))
    parent = _parent_memory()

    spec = await service._resolve_explicit_subagent(
        _spec(), inherited_middleware=[parent]
    )

    [memory] = _memory_of(spec)
    assert memory is not parent
    assert memory.memory_service is parent.memory_service
    assert memory.postrun_analysis is False
    assert parent.postrun_analysis is True


@pytest.mark.asyncio
async def test_a_subagent_with_a_memory_of_its_own_keeps_it():
    service = _deep_service(_Scripted(replies=[]))
    own = MemoryMiddleware(
        memory_service=AgentMemoryStoreService(store=InMemoryStore()), user_id="u1"
    )

    spec = await service._resolve_explicit_subagent(
        _spec(middleware=[own]), inherited_middleware=[_parent_memory()]
    )

    assert _memory_of(spec) == [own]


@pytest.mark.asyncio
async def test_a_subagent_whose_memory_is_off_gets_none():
    service = _deep_service(_Scripted(replies=[]))

    spec = await service._resolve_explicit_subagent(
        _spec(**{INHERIT_MEMORY_KEY: False}), inherited_middleware=[_parent_memory()]
    )

    assert _memory_of(spec) == []


@pytest.mark.asyncio
async def test_a_memory_kit_reads_and_writes_the_subagents_spaces():
    service = _deep_service(_Scripted(replies=[]))
    service.memory_store = InMemoryStore()
    options = MemoryOptions(
        spaces=(
            MemorySpace(key="private", path=("agent:a1",), description="Own work."),
            MemorySpace(key="team", path=("space:t1",), description="The team."),
        ),
        default_write="private",
    )

    memory, middleware, tools = service.build_memory_kit(options)

    assert [s.key for s in memory.spaces] == ["private", "team"]
    assert memory.store is service.memory_store
    assert middleware.memory_service is memory
    assert middleware.postrun_analysis is False
    assert sorted(t.name for t in tools) == sorted(middleware.active_tool_names)
    [save] = [t for t in tools if t.name == "save_memory_store"]
    assert "- team: The team." in save.description


def test_a_memory_kit_writes_to_the_runs_write_log():
    service = _deep_service(_Scripted(replies=[]))
    service.memory_store = InMemoryStore()
    log = TurnWriteLog()
    service._memory_options = MemoryOptions(write_log=log)
    service._memory_analysis_options = PostRunAnalysisOptions(record_writes=True)

    memory, middleware, _ = service.build_memory_kit(
        MemoryOptions(spaces=(MemorySpace(key="private", path=("agent:a1",)),))
    )

    assert memory.write_log is log
    # The parent keeps the subagent's writes in its thread, not the subagent.
    assert middleware.analysis_options.record_writes is False
    assert middleware.turn_writes_keeper() is None


def test_the_keeper_of_the_writes_is_not_handed_to_subagents():
    keeper = MemoryTurnWritesMiddleware(_parent_memory())
    assert _propagates_to_subagents(keeper) is False


# --------------------------------------------------------------------------- #
# An instance's life
# --------------------------------------------------------------------------- #


@pytest.fixture
def clean_hooks():
    validators, hooks = list(_overrides_validators), list(_instance_deleted_hooks)
    yield
    _overrides_validators[:] = validators
    _instance_deleted_hooks[:] = hooks


@pytest.mark.asyncio
async def test_an_application_can_refuse_config_overrides(clean_hooks):
    async def no_unknown_spaces(db, user_id, overrides):
        if "nowhere" in (overrides.get("memory") or {}).get("read", []):
            raise ValueError("Unknown memory space: nowhere")

    register_config_overrides_validator(no_unknown_spaces)

    await _run_overrides_validators(None, uuid.uuid4(), {"memory": {"read": ["general"]}})
    with pytest.raises(ValueError, match="nowhere"):
        await _run_overrides_validators(
            None, uuid.uuid4(), {"memory": {"read": ["nowhere"]}}
        )


@pytest.mark.asyncio
async def test_a_deleted_instance_is_reported_after_the_commit(clean_hooks):
    calls = []

    async def failing(user_id, instance_id):
        raise RuntimeError("store down")

    async def forget(user_id, instance_id):
        calls.append((user_id, instance_id, db.commit.await_count))

    register_instance_deleted_hook(failing)
    register_instance_deleted_hook(forget)
    db = MagicMock()
    db.delete = AsyncMock()
    db.commit = AsyncMock()
    service = UserAgentInstanceService(db, MagicMock())
    user_id, instance_id = uuid.uuid4(), uuid.uuid4()
    instance = MagicMock(instance_name="writer")

    with patch.object(service, "get_instance", AsyncMock(return_value=instance)):
        assert await service.delete_instance(user_id, instance_id) is True

    assert calls == [(user_id, instance_id, 1)]


# --------------------------------------------------------------------------- #
# The run's writes (PostRunAnalysisOptions.record_writes)
# --------------------------------------------------------------------------- #


def _with_memory(**overrides):
    service = _deep_service(_Scripted(replies=[]), **overrides)
    service.use_memory = True
    service._memory_service = AgentMemoryStoreService(
        store=InMemoryStore(), spaces=[MemorySpace(key="general")]
    )
    settings = MagicMock(agent_memory_max_results=5, agent_memory_model=None)
    with patch(
        "inference_core.services.agents_service.get_settings", return_value=settings
    ):
        return service, service._build_middleware()


def test_with_record_writes_the_run_has_one_write_log_and_a_keeper():
    service, middleware = _with_memory(
        memory_options=MemoryOptions(),
        memory_analysis_options=PostRunAnalysisOptions(record_writes=True),
    )

    assert isinstance(service._memory_options.write_log, TurnWriteLog)
    [memory] = [m for m in middleware if isinstance(m, MemoryMiddleware)]
    keepers = [m for m in middleware if isinstance(m, MemoryTurnWritesMiddleware)]
    assert len(keepers) == 1 and keepers[0].memory is memory
    assert middleware.index(keepers[0]) == middleware.index(memory) + 1


def test_without_options_there_is_no_write_log_and_no_keeper():
    service, middleware = _with_memory(memory_options=MemoryOptions())

    assert service._memory_options.write_log is None
    assert not any(isinstance(m, MemoryTurnWritesMiddleware) for m in middleware)
    [memory] = [m for m in middleware if isinstance(m, MemoryMiddleware)]
    assert memory.analysis_options == PostRunAnalysisOptions()

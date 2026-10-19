"""What an instance is checked for when it is created or changed."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_core.llm.config import LLMConfig, MCPConfig, MCPProfileConfig
from inference_core.services.user_agent_instance_service import (
    InstanceCheck,
    UserAgentInstanceService,
    _extra_models_sources,
    _instance_validator_triggers,
    _instance_validators,
    register_extra_models_source,
    register_instance_validator,
)

USER = uuid.UUID("11111111-1111-1111-1111-111111111111")


class _Resolved:
    def __init__(self):
        selectable = SimpleNamespace(user_selectable=True)
        self.agents = {"assistant_agent": selectable, "email_agent": selectable}
        self.available_models = ["model-a"]


class _LLMConfigService:
    async def get_resolved_config(self, user_id):
        return _Resolved()


def _llm_config(*, enabled=True, require_superuser=False):
    config = object.__new__(LLMConfig)
    config.agent_configs = {
        "assistant_agent": SimpleNamespace(mcp_profile="web"),
        "email_agent": SimpleNamespace(mcp_profile=None),
    }
    config.mcp_config = MCPConfig(
        enabled=enabled,
        require_superuser=require_superuser,
        profiles={
            "web": MCPProfileConfig(servers=["s"]),
            "admin": MCPProfileConfig(servers=["s"], agents=["assistant_agent"]),
        },
    )
    return config


@pytest.fixture
def llm_config():
    holder = {"config": _llm_config()}
    with patch(
        "inference_core.services.user_agent_instance_service.get_llm_config",
        side_effect=lambda: holder["config"],
    ):
        yield holder


@pytest.fixture
def clean_validators():
    saved = list(_instance_validators)
    saved_triggers = dict(_instance_validator_triggers)
    saved_sources = list(_extra_models_sources)
    yield
    _instance_validators[:] = saved
    _instance_validator_triggers.clear()
    _instance_validator_triggers.update(saved_triggers)
    _extra_models_sources[:] = saved_sources


def _service(db=None):
    return UserAgentInstanceService(db=db, llm_config_service=_LLMConfigService())


def _db_with(instance):
    db = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = instance
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    return db


def _instance(base="assistant_agent", overrides=None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        base_agent_name=base,
        config_overrides=overrides or {},
        primary_model=None,
        skills=None,
        is_default=False,
        is_deepagent=False,
        subagents=[],
    )


async def _create(
    service, overrides, *, base="assistant_agent", superuser=None, **fields
):
    # Refusals come before the uniqueness check, which needs the database.
    with patch.object(
        service, "get_instance_by_name", AsyncMock(return_value=object())
    ):
        await service.create_instance(
            user_id=USER,
            instance_name="x",
            display_name="X",
            base_agent_name=base,
            config_overrides=overrides,
            actor_is_superuser=superuser,
            **fields,
        )


@pytest.mark.asyncio
async def test_an_instance_may_not_select_a_profile_of_another_type(llm_config):
    with pytest.raises(ValueError, match="not available to agent type 'email_agent'"):
        await _create(_service(), {"mcp_profile": "web"}, base="email_agent")


@pytest.mark.asyncio
async def test_an_instance_may_select_a_profile_meant_for_its_type(llm_config):
    # The check passes and the create goes on to the uniqueness check.
    with pytest.raises(ValueError, match="already exists"):
        await _create(_service(), {"mcp_profile": "admin"})


@pytest.mark.asyncio
async def test_a_profile_that_takes_a_superuser_is_refused_to_anyone_else(llm_config):
    llm_config["config"] = _llm_config(require_superuser=True)
    with pytest.raises(ValueError, match="only be selected by a superuser"):
        await _create(_service(), {"mcp_profile": "admin"}, superuser=False)
    with pytest.raises(ValueError, match="already exists"):
        await _create(_service(), {"mcp_profile": "admin"}, superuser=True)


@pytest.mark.asyncio
async def test_a_change_of_type_alone_is_checked_again(llm_config):
    instance = _instance(overrides={"mcp_profile": "web"})
    service = _service(_db_with(instance))

    with pytest.raises(ValueError, match="not available to agent type 'email_agent'"):
        await service.update_instance(USER, instance.id, base_agent_name="email_agent")


@pytest.mark.asyncio
async def test_a_profile_the_instance_already_has_is_kept(llm_config):
    """A later change of the configuration does not make the instance unsavable."""
    instance = _instance(overrides={"mcp_profile": "web", "temperature": 0.1})
    service = _service(_db_with(instance))
    llm_config["config"] = _llm_config(enabled=False)

    await service.update_instance(
        USER,
        instance.id,
        config_overrides={"mcp_profile": "web", "temperature": 0.2},
    )

    assert instance.config_overrides == {"mcp_profile": "web", "temperature": 0.2}


@pytest.mark.asyncio
async def test_an_application_sees_what_the_instance_is_about_to_become(
    llm_config, clean_validators
):
    seen = []

    async def record(db, check):
        seen.append(check)

    register_instance_validator(record)
    instance = _instance(overrides={"memory": {"read": ["general"]}})
    service = _service(_db_with(instance))

    await service.update_instance(
        USER, instance.id, actor_is_superuser=True, base_agent_name="email_agent"
    )

    [check] = seen
    assert check == InstanceCheck(
        user_id=USER,
        instance_id=instance.id,
        base_agent_name="email_agent",
        config_overrides={"memory": {"read": ["general"]}},
        previous_base_agent_name="assistant_agent",
        previous_overrides={"memory": {"read": ["general"]}},
        actor_is_superuser=True,
    )
    assert check.type_changed
    assert not check.changed("memory")


@pytest.mark.asyncio
async def test_an_application_can_refuse_a_new_instance(llm_config, clean_validators):
    async def refuse(db, check):
        if check.instance_id is None and "connectors" in check.config_overrides:
            raise ValueError("No connectors here")

    register_instance_validator(refuse)

    with pytest.raises(ValueError, match="No connectors here"):
        await _create(_service(), {"connectors": []})


@pytest.mark.asyncio
async def test_a_change_of_other_fields_runs_no_check(llm_config, clean_validators):
    async def refuse(db, check):
        raise ValueError("checked")

    register_instance_validator(refuse)
    instance = _instance(overrides={"mcp_profile": "web"})
    instance.display_name = "Old"
    service = _service(_db_with(instance))

    await service.update_instance(USER, instance.id, display_name="New")

    assert instance.display_name == "New"


@pytest.mark.asyncio
async def test_a_model_the_configuration_lacks_is_refused(llm_config):
    with pytest.raises(ValueError, match="Model 'own-model' not found"):
        await _create(_service(), {}, primary_model="own-model")


@pytest.mark.asyncio
async def test_a_source_may_admit_a_users_own_model(llm_config, clean_validators):
    asked = []

    async def own_models(db, user_id):
        asked.append(user_id)
        return ["own-model"]

    register_extra_models_source(own_models)

    with pytest.raises(ValueError, match="already exists"):
        await _create(_service(), {}, primary_model="own-model")
    with pytest.raises(ValueError, match="Model 'other' not found"):
        await _create(_service(), {}, primary_model="other")
    # A configured model needs no source.
    asked.clear()
    with pytest.raises(ValueError, match="already exists"):
        await _create(_service(), {}, primary_model="model-a")
    assert asked == []


@pytest.mark.asyncio
async def test_a_source_admits_a_model_on_update_too(llm_config, clean_validators):
    async def own_models(db, user_id):
        return ["own-model"]

    instance = _instance()
    service = _service(_db_with(instance))

    with pytest.raises(ValueError, match="Model 'own-model' not found"):
        await service.update_instance(USER, instance.id, primary_model="own-model")

    register_extra_models_source(own_models)
    await service.update_instance(USER, instance.id, primary_model="own-model")

    assert instance.primary_model == "own-model"


@pytest.mark.asyncio
async def test_a_new_instance_shows_its_model_and_subagents(
    llm_config, clean_validators
):
    seen = []

    async def record(db, check):
        seen.append(check)

    register_instance_validator(record)
    subagent = uuid.uuid4()

    with pytest.raises(ValueError, match="already exists"):
        await _create(
            _service(),
            {},
            primary_model="model-a",
            skills=[{"name": "s"}],
            is_deepagent=True,
            subagent_ids=[subagent],
        )

    [check] = seen
    assert check.primary_model == "model-a"
    assert check.previous_primary_model is None
    assert check.skills == ({"name": "s"},)
    assert check.is_deepagent is True
    assert check.subagent_ids == (subagent,)


@pytest.mark.asyncio
async def test_a_change_of_model_alone_runs_no_check_by_default(
    llm_config, clean_validators
):
    async def refuse(db, check):
        raise ValueError("checked")

    register_instance_validator(refuse)
    instance = _instance()
    service = _service(_db_with(instance))

    await service.update_instance(USER, instance.id, primary_model="model-a")

    assert instance.primary_model == "model-a"


@pytest.mark.asyncio
async def test_a_validator_may_ask_to_see_a_change_of_model(
    llm_config, clean_validators
):
    seen = []

    async def record(db, check):
        seen.append(check)

    async def refuse(db, check):
        raise ValueError("not asked")

    register_instance_validator(record, also_on={"primary_model"})
    register_instance_validator(refuse, also_on={"skills"})
    instance = _instance()
    service = _service(_db_with(instance))

    await service.update_instance(USER, instance.id, primary_model="model-a")

    [check] = seen
    assert check.primary_model == "model-a"
    assert check.previous_primary_model is None
    assert not check.type_changed


@pytest.mark.asyncio
async def test_a_validator_may_ask_to_see_new_subagents(llm_config, clean_validators):
    seen = []

    async def record(db, check):
        seen.append(check.subagent_ids)

    register_instance_validator(record, also_on={"subagents"})
    instance = _instance()
    instance.is_deepagent = True
    subagent = SimpleNamespace(id=uuid.uuid4())
    db = _db_with(instance)
    found = MagicMock()
    found.scalars.return_value.all.return_value = [subagent]
    db.execute = AsyncMock(side_effect=[db.execute.return_value, found, found])
    service = _service(db)

    await service.update_instance(USER, instance.id, subagent_ids=[subagent.id])

    assert seen == [(subagent.id,)]


def test_an_unknown_change_is_refused(clean_validators):
    async def record(db, check):
        pass

    with pytest.raises(ValueError, match="Unknown instance changes"):
        register_instance_validator(record, also_on={"display_name"})

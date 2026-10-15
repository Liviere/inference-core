"""Which agent types may use an MCP profile, and who may select one for an instance."""

from types import SimpleNamespace

import pytest

from inference_core.llm.config import LLMConfig, MCPConfig, MCPProfileConfig


def _mcp(*, enabled=True, require_superuser=False, **profile):
    return MCPConfig(
        enabled=enabled,
        require_superuser=require_superuser,
        profiles={"web": MCPProfileConfig(servers=["s"], **profile)},
    )


def _problem(mcp, *, agent_type="assistant_agent", type_profile="web", superuser=False):
    return mcp.profile_problem(
        "web", agent_type=agent_type, type_profile=type_profile, is_superuser=superuser
    )


def test_the_type_that_names_a_profile_may_use_it():
    assert _problem(_mcp()) is None


def test_another_type_may_not():
    problem = _problem(_mcp(), agent_type="email_agent", type_profile=None)
    assert problem == "MCP profile 'web' is not available to agent type 'email_agent'"


def test_a_profile_may_list_more_types():
    mcp = _mcp(agents=["email_agent"])
    assert _problem(mcp, agent_type="email_agent", type_profile=None) is None


def test_nothing_is_available_while_mcp_is_off():
    assert _problem(_mcp(enabled=False)) == "MCP is disabled"


def test_an_unknown_profile_is_named():
    problem = _mcp().profile_problem(
        "nowhere", agent_type="assistant_agent", type_profile="nowhere", is_superuser=True
    )
    assert problem == "MCP profile 'nowhere' does not exist"


@pytest.mark.parametrize(
    ("global_flag", "profile_flag", "superuser", "allowed"),
    [
        (True, None, False, False),
        (True, None, True, True),
        (True, None, None, False),  # who acts was not said: not a superuser
        (False, True, False, False),  # the profile asks for more than the global flag
        (True, False, False, True),  # and may ask for less
        (False, None, False, True),
    ],
)
def test_selecting_a_profile_may_take_a_superuser(
    global_flag, profile_flag, superuser, allowed
):
    mcp = _mcp(require_superuser=global_flag)
    mcp.profiles["web"].require_superuser = profile_flag
    problem = _problem(mcp, superuser=superuser)
    assert (problem is None) is allowed
    if not allowed:
        assert problem == "MCP profile 'web' can only be selected by a superuser"


def test_the_loaded_configuration_knows_which_profile_a_type_names():
    config = object.__new__(LLMConfig)
    config.agent_configs = {
        "assistant_agent": SimpleNamespace(mcp_profile="web"),
        "email_agent": SimpleNamespace(mcp_profile=None),
    }
    config.mcp_config = _mcp()

    assert config.mcp_profile_problem("web", "assistant_agent") is None
    assert "not available" in config.mcp_profile_problem("web", "email_agent")
    assert "not available" in config.mcp_profile_problem("web", "no_such_type")


def test_profiles_from_before_still_load():
    profile = MCPProfileConfig(servers=["s"])
    assert profile.agents is None
    assert profile.require_superuser is None
    assert profile.load_timeout_seconds == 15

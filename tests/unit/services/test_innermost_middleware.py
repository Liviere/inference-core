"""Middleware placed inside everything else of an agent's model calls."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents.middleware import AgentMiddleware

from inference_core.agents.middleware.cost_tracking import CostTrackingMiddleware
from inference_core.agents.middleware.system_prompt_breakpoint import (
    SystemPromptBreakpointMiddleware,
)
from inference_core.services.agents_service import AgentService, DeepAgentService


class _Outer(AgentMiddleware):
    pass


class _Inner(AgentMiddleware):
    pass


@pytest.fixture
def factory():
    with patch("inference_core.services.agents_service.get_model_factory") as get:
        made = MagicMock()
        agent_config = MagicMock()
        agent_config.mcp_profile = None
        agent_config.local_tool_providers = []
        agent_config.allowed_tools = None
        agent_config.fallback = None
        agent_config.response_format = None
        made.config.get_specific_agent_config.return_value = agent_config
        made.config.get_model_params.return_value = {}
        get.return_value = made
        yield made


async def _middleware_of(**kwargs) -> list:
    service = AgentService(agent_name="test_agent", **kwargs)
    with (
        patch("inference_core.services.agents_service.create_agent") as create,
        patch.object(service, "_load_providers_tools", new=AsyncMock()),
        patch.object(service, "_load_mcp_tools", new=AsyncMock()),
        patch.object(service, "_load_memory_tools", new=AsyncMock()),
    ):
        await service.create_agent(system_prompt="You help.")
    return create.call_args.kwargs["middleware"]


async def test_it_stands_inside_the_rest_and_cost_tracking(factory):
    inner = _Inner()

    middleware = await _middleware_of(
        middleware=[_Outer()], innermost_middleware=[inner]
    )

    assert middleware[-1] is inner
    assert isinstance(middleware[-2], CostTrackingMiddleware)
    assert isinstance(middleware[0], _Outer)


async def test_by_default_there_is_none(factory):
    middleware = await _middleware_of(middleware=[_Outer()])

    assert isinstance(middleware[-1], CostTrackingMiddleware)


def test_a_deep_agent_takes_it_too(factory):
    """Its constructor names what it passes on: a keyword it does not name is a
    TypeError for every deep agent."""
    inner = _Inner()
    service = DeepAgentService(agent_name="test_agent", innermost_middleware=[inner])
    assert service._innermost_middleware == [inner]


# The cache breakpoint at the end of the system prompt
# (``system_prompt_breakpoint``): last of all, so it marks the system prompt
# as it is sent, on the model a fallback switched to.


async def test_the_system_prompts_breakpoint_is_last_of_all(factory):
    inner = _Inner()

    middleware = await _middleware_of(
        middleware=[_Outer()],
        innermost_middleware=[inner],
        system_prompt_breakpoint=True,
    )

    assert isinstance(middleware[-1], SystemPromptBreakpointMiddleware)
    assert middleware[-2] is inner


async def test_by_default_there_is_no_breakpoint(factory):
    middleware = await _middleware_of(middleware=[_Outer()])

    assert not any(isinstance(m, SystemPromptBreakpointMiddleware) for m in middleware)


def test_a_deep_agent_passes_the_breakpoint_on(factory):
    service = DeepAgentService(agent_name="test_agent", system_prompt_breakpoint=True)

    assert service._system_prompt_breakpoint is True


def test_a_subagents_spec_gets_its_own_last(factory):
    service = DeepAgentService(agent_name="test_agent", system_prompt_breakpoint=True)
    cost = CostTrackingMiddleware()
    spec = {"name": "helper", "model": MagicMock(), "middleware": [_Outer(), cost]}

    given = service._with_system_prompt_breakpoint(spec)

    assert isinstance(given["middleware"][-1], SystemPromptBreakpointMiddleware)
    assert given["middleware"][:2] == [spec["middleware"][0], cost]
    assert len(spec["middleware"]) == 2
    # Once.
    again = service._with_system_prompt_breakpoint(given)
    assert len(again["middleware"]) == 3


def test_a_compiled_subagent_and_an_agent_without_the_option_are_left_alone(factory):
    compiled = {"name": "nested", "description": "d", "runnable": MagicMock()}
    on = DeepAgentService(agent_name="test_agent", system_prompt_breakpoint=True)
    off = DeepAgentService(agent_name="test_agent")
    spec = {"name": "helper", "model": MagicMock(), "middleware": [_Outer()]}

    assert on._with_system_prompt_breakpoint(compiled) is compiled
    assert off._with_system_prompt_breakpoint(spec) is spec

"""Middleware placed inside everything else of an agent's model calls."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents.middleware import AgentMiddleware

from inference_core.agents.middleware.cost_tracking import CostTrackingMiddleware
from inference_core.services.agents_service import AgentService


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

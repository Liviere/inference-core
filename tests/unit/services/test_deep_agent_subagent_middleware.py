"""What a subagent is given of its parent's middleware.

Cost tracking and memory were handed on and nothing else, so a middleware
that watches what tools return (a guard against instructions planted in a
web page, say) watched the parent's tools and none of its subagents'. A
middleware with ``copy_for_subagent()`` is handed on as what that returns.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents.middleware import AgentMiddleware

from inference_core.services.agents_service import (
    _for_subagent,
    _propagates_to_subagents,
)
from tests.unit.services.test_deep_agent_task_brief import _deep_service, _Scripted


class _Watcher(AgentMiddleware):
    """Copies itself for a subagent, with a note of whose it is."""

    def __init__(self, owner: str = "parent") -> None:
        self.owner = owner

    def copy_for_subagent(self) -> "_Watcher":
        return _Watcher(owner="subagent")


class _Plain(AgentMiddleware):
    pass


def _spec(**extra):
    return {
        "name": "reader",
        "description": "Reads.",
        "system_prompt": "You read.",
        "model": _Scripted(replies=[]),
        "tools": [],
        **extra,
    }


def _watchers(middleware) -> list:
    return [m for m in middleware if isinstance(m, _Watcher)]


def test_a_middleware_that_copies_itself_is_handed_on():
    assert _propagates_to_subagents(_Watcher()) is True
    assert _propagates_to_subagents(_Plain()) is False


def test_a_mocks_made_up_attribute_does_not_count():
    assert _propagates_to_subagents(MagicMock()) is False
    mock = MagicMock()
    assert _for_subagent(mock) is mock


@pytest.mark.asyncio
async def test_an_explicit_subagent_gets_the_copy():
    service = _deep_service(_Scripted(replies=[]))
    parent = _Watcher()

    spec = await service._resolve_explicit_subagent(
        _spec(), inherited_middleware=[parent]
    )

    [watcher] = _watchers(spec["middleware"])
    assert watcher is not parent
    assert watcher.owner == "subagent"


@pytest.mark.asyncio
async def test_a_subagent_with_one_of_its_own_keeps_it():
    service = _deep_service(_Scripted(replies=[]))
    own = _Watcher(owner="own")

    spec = await service._resolve_explicit_subagent(
        _spec(middleware=[own]), inherited_middleware=[_Watcher()]
    )

    assert _watchers(spec["middleware"]) == [own]


@pytest.mark.asyncio
async def test_a_subagent_of_the_configuration_gets_the_copy():
    service = _deep_service(_Scripted(replies=[]))
    service.model_factory.config.get_specific_agent_config.return_value.subagents = None

    with patch.object(service, "_load_tools_for_subagent", AsyncMock(return_value=[])):
        spec = await service._build_subagent_from_config(
            "reader", inherited_middleware=[_Watcher()]
        )

    [watcher] = _watchers(spec["middleware"])
    assert watcher.owner == "subagent"

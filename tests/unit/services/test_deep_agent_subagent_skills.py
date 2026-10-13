"""A declarative subagent's own skills.

``SubAgentMiddleware`` compiles a subagent from its spec and does not look at
skills, so ``DeepAgentService`` gives a subagent what it gives the parent: the
SKILL.md files in the run's store, the list of them in the system prompt and a
reader for the instructions. The last test runs a real graph, with stand-ins
for the two models only, since most of this is in how the pieces meet: the
subagent's graph reads the store of the graph that called it.
"""

import uuid
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents.middleware import SkillsMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.store.memory import InMemoryStore
from pydantic import Field

from inference_core.agents.middleware import CostTrackingMiddleware
from inference_core.agents.tools.skill_reader import READ_SKILL_FILE_TOOL_NAME
from inference_core.services.agents_service import (
    SUBAGENT_SKILLS_ROOT,
    DeepAgentService,
)


def _skill(name: str, description: str, body: str) -> dict[str, str]:
    return {
        "name": name,
        "description": description,
        "content": f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
    }


RESEARCH = _skill(
    "web-research",
    "How the researcher looks for news.",
    "# Steps\n1. Search three sources and keep the links.",
)
ROUTING = _skill(
    "request-routing",
    "How the coordinator decides who gets a task.",
    "# Steps\n1. Hand research to the researcher.",
)


class _Scripted(BaseChatModel):
    """Answers with its replies in order and keeps what it was sent."""

    replies: List[Any]
    seen: List[List[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_Scripted":
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        reply = self.replies[len(self.seen) - 1]
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _deep_service(model: BaseChatModel, **overrides: Any) -> DeepAgentService:
    """A ``DeepAgentService`` on *model*, with no config, database or keys."""
    factory = MagicMock()
    factory.get_agent_model_name.return_value = "scripted"
    factory.get_model_for_agent.return_value = model
    factory.aclose = AsyncMock()
    factory.config.models = {}
    factory.config.get_model_params.return_value = {}
    factory.config.get_specific_agent_config.return_value = MagicMock(
        local_tool_providers=[],
        mcp_profile=None,
        allowed_tools=None,
        tool_model_overrides=None,
        description="A coordinator",
        skills=None,
        subagents=None,
        interrupt_on=None,
        memory_tools=None,
        memory_session_context_enabled=None,
        memory_tool_instructions_enabled=None,
        tool_call_limits=None,
        response_format=None,
        fallback=None,
        generation_params=None,
        reasoning_output=False,
    )
    arguments = {
        "agent_name": "coordinator",
        "enable_cost_tracking": False,
        "user_id": uuid.uuid4(),
        "session_id": "sess-1",
        "request_id": "req-1",
    }
    arguments.update(overrides)
    with (
        patch(
            "inference_core.services.agents_service.get_model_factory",
            return_value=factory,
        ),
        patch(
            "inference_core.services.agents_service.get_llm_config",
            return_value=MagicMock(),
        ),
        patch(
            "inference_core.services.agents_service.get_settings",
            return_value=MagicMock(
                database_url="sqlite+aiosqlite:///test.db",
                agent_memory_enabled=False,
                llm_emulation_enabled=False,
                agent_snapshot_capture_enabled=False,
                agent_snapshot_replay_enabled=False,
                vector_dim=8,
            ),
        ),
    ):
        return DeepAgentService(**arguments)


def _text(message: BaseMessage) -> str:
    return message.text if hasattr(message, "text") else str(message.content)


# --------------------------------------------------------------------------- #
# The spec
# --------------------------------------------------------------------------- #


def test_a_spec_carries_the_subagents_own_skills():
    spec = DeepAgentService._build_declarative_subagent_spec(
        name="researcher",
        description="Researches",
        system_prompt="You research.",
        model=MagicMock(),
        tools=[],
        user_skills=[RESEARCH],
    )

    assert spec["user_skills"] == [RESEARCH]


def test_a_spec_without_skills_carries_no_key_for_them():
    spec = DeepAgentService._build_declarative_subagent_spec(
        name="researcher",
        description="Researches",
        system_prompt="You research.",
        model=MagicMock(),
        tools=[],
        user_skills=[],
    )

    assert "user_skills" not in spec


def test_the_skills_are_taken_off_the_specs_deepagents_is_given():
    with_skills = {"name": "researcher", "user_skills": [RESEARCH]}
    without = {"name": "writer", "user_skills": []}
    compiled = {"name": "compiled", "runnable": object(), "user_skills": [RESEARCH]}

    taken = DeepAgentService._take_subagent_skills([with_skills, without, compiled])

    assert taken == [(with_skills, [RESEARCH])]
    assert "user_skills" not in with_skills
    assert "user_skills" not in without
    # A compiled subagent is somebody else's: it is handed on as it came.
    assert compiled["user_skills"] == [RESEARCH]


# --------------------------------------------------------------------------- #
# What a subagent with skills is given
# --------------------------------------------------------------------------- #


def _files(store: InMemoryStore) -> set[str]:
    return {item.key for item in store.search(("filesystem",), limit=100)}


def test_a_subagent_gets_its_files_its_list_and_its_reader():
    service = _deep_service(_Scripted(replies=[]))
    store = InMemoryStore()
    cost_tracking = CostTrackingMiddleware(
        pricing_config=None, user_id=None, session_id="s", request_id="r"
    )
    spec = {"name": "researcher", "middleware": [cost_tracking], "tools": []}

    service._give_subagents_their_skills(
        [(spec, [RESEARCH])], store, lambda runtime: None
    )

    directory = f"{SUBAGENT_SKILLS_ROOT}researcher/"
    assert _files(store) == {f"{directory}web-research/SKILL.md"}
    skills = [m for m in spec["middleware"] if isinstance(m, SkillsMiddleware)]
    assert [m.sources for m in skills] == [[directory]]
    # Cost tracking stays last, as it does for the parent.
    assert spec["middleware"][-1] is cost_tracking
    assert [tool.name for tool in spec["tools"]] == [READ_SKILL_FILE_TOOL_NAME]


def test_two_subagents_keep_their_skills_apart():
    service = _deep_service(_Scripted(replies=[]))
    store = InMemoryStore()
    first = {"name": "a b", "tools": []}
    second = {"name": "a_b", "tools": []}

    service._give_subagents_their_skills(
        [(first, [RESEARCH]), (second, [ROUTING])], store, lambda runtime: None
    )

    # Both names make the same directory name: the second gets one of its own.
    assert _files(store) == {
        f"{SUBAGENT_SKILLS_ROOT}a_b/web-research/SKILL.md",
        f"{SUBAGENT_SKILLS_ROOT}a_b-2/request-routing/SKILL.md",
    }


def test_nothing_is_given_without_a_store():
    service = _deep_service(_Scripted(replies=[]))
    spec = {"name": "researcher", "tools": []}

    service._give_subagents_their_skills([], None, lambda runtime: None)

    assert spec == {"name": "researcher", "tools": []}


# --------------------------------------------------------------------------- #
# A run
# --------------------------------------------------------------------------- #


def _coordinator_and_researcher(parent_skills):
    researcher = _Scripted(
        replies=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": READ_SKILL_FILE_TOOL_NAME,
                        "args": {
                            "file_path": (
                                f"{SUBAGENT_SKILLS_ROOT}researcher/"
                                "web-research/SKILL.md"
                            )
                        },
                        "id": "read-1",
                    }
                ],
            ),
            AIMessage(content="Three sources, with links."),
        ]
    )
    coordinator = _Scripted(
        replies=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "Today's news",
                            "prompt": "Find today's news.",
                            "subagent_type": "researcher",
                        },
                        "id": "task-1",
                    }
                ],
            ),
            AIMessage(content="Here is the news."),
        ]
    )
    service = _deep_service(
        coordinator,
        user_skills=parent_skills,
        subagents=[
            {
                "name": "researcher",
                "description": "Looks things up.",
                "system_prompt": "You are the researcher.",
                "model": researcher,
                "tools": [],
                "user_skills": [RESEARCH],
            }
        ],
    )
    return service, coordinator, researcher


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_skills", [[ROUTING], None])
async def test_a_subagent_lists_and_reads_its_own_skills(parent_skills):
    service, coordinator, researcher = _coordinator_and_researcher(parent_skills)
    agent = await service.create_agent(system_prompt="You coordinate.")

    state = await agent.ainvoke({"messages": [("user", "What is new?")]})

    assert _text(state["messages"][-1]) == "Here is the news."

    # The subagent is told of its own skill, and of nobody else's.
    told = _text(researcher.seen[0][0])
    assert "You are the researcher." in told
    assert "web-research" in told
    assert "How the researcher looks for news." in told
    assert "request-routing" not in told

    # It read the instructions through its reader.
    read = [m for m in researcher.seen[1] if isinstance(m, ToolMessage)]
    assert len(read) == 1
    assert "Search three sources and keep the links." in _text(read[0])

    # The parent is told of its own skills only, and has no way to the
    # subagent's: its reader, when it has one, is tied to its own directory.
    parent_told = _text(coordinator.seen[0][0])
    assert "web-research" not in parent_told
    assert ("request-routing" in parent_told) is bool(parent_skills)
    readers = [t for t in service.tools if t.name == READ_SKILL_FILE_TOOL_NAME]
    assert len(readers) == (1 if parent_skills else 0)
    for reader in readers:
        refused = reader.invoke(
            {"file_path": (f"{SUBAGENT_SKILLS_ROOT}researcher/web-research/SKILL.md")}
        )
        assert refused.startswith("Error:")

    # The subagent's answer came back to the parent as the task's result.
    answered = [m for m in coordinator.seen[1] if isinstance(m, ToolMessage)]
    assert [_text(m) for m in answered] == ["Three sources, with links."]

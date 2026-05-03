import pytest
from deepagents.backends import StoreBackend
from deepagents.backends.utils import create_file_data
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolRuntime, _get_all_injected_args
from langgraph.store.memory import InMemoryStore

from inference_core.agents.tools.skill_reader import (
    READ_SKILL_FILE_TOOL_NAME,
    create_skill_reader_tool,
    create_skills_middleware,
)


def _skill_backend() -> StoreBackend:
    store = InMemoryStore()
    store.put(
        namespace=("filesystem",),
        key="/skills/test-skill/SKILL.md",
        value=create_file_data(
            "---\nname: test-skill\ndescription: Test skill\n---\n\n# Test Skill\n"
        ),
    )
    return StoreBackend(store=store, namespace=lambda _runtime: ("filesystem",))


def test_read_skill_file_reads_configured_skill() -> None:
    tool = create_skill_reader_tool(backend=_skill_backend(), sources=["/skills/"])

    result = tool.invoke({"file_path": "/skills/test-skill/SKILL.md", "limit": 1000})

    assert tool.name == READ_SKILL_FILE_TOOL_NAME
    assert "# Test Skill" in result
    assert "     1\t---" in result


@pytest.mark.parametrize(
    "file_path",
    [
        "/skills/test-skill/helper.py",
        "/skills/test-skill/nested/SKILL.md",
        "/user-files/test-skill/SKILL.md",
        "/skills/test-skill/../secret/SKILL.md",
    ],
)
def test_read_skill_file_rejects_non_skill_paths(file_path: str) -> None:
    tool = create_skill_reader_tool(backend=_skill_backend(), sources=["/skills/"])

    result = tool.invoke({"file_path": file_path})

    assert result.startswith("Error:")


def test_read_skill_file_accepts_relative_source_paths() -> None:
    tool = create_skill_reader_tool(backend=_skill_backend(), sources=["skills/"])

    result = tool.invoke({"file_path": "skills/test-skill/SKILL.md"})

    assert "# Test Skill" in result


def test_read_skill_file_accepts_skill_relative_path_without_source_prefix() -> None:
    tool = create_skill_reader_tool(backend=_skill_backend(), sources=["/skills/"])

    result = tool.invoke({"file_path": "test-skill/SKILL.md"})

    assert "# Test Skill" in result


def test_read_skill_file_accepts_skill_name_only() -> None:
    tool = create_skill_reader_tool(backend=_skill_backend(), sources=["/skills/"])

    result = tool.invoke({"file_path": "test-skill"})

    assert "# Test Skill" in result


def test_read_skill_file_supports_runtime_injected_backend_factory() -> None:
    tool = create_skill_reader_tool(
        backend=lambda _runtime: _skill_backend(),
        sources=["/skills/"],
    )
    tool_node = ToolNode([tool])
    injected_args = _get_all_injected_args(tool)
    runtime = ToolRuntime(
        state={"messages": []},
        context={},
        config={},
        stream_writer=lambda *_args, **_kwargs: None,
        tool_call_id="call-1",
        store=None,
    )

    result = tool_node._run_one(
        {
            "name": READ_SKILL_FILE_TOOL_NAME,
            "args": {"file_path": "/skills/test-skill/SKILL.md", "limit": 1000},
            "id": "call-1",
            "type": "tool_call",
        },
        "dict",
        runtime,
    )

    assert injected_args.runtime == "runtime"
    assert "# Test Skill" in result.content


def test_skills_middleware_prompt_uses_restricted_reader() -> None:
    middleware = create_skills_middleware(
        backend=_skill_backend(),
        sources=["/skills/"],
    )

    assert "read_skill_file" in middleware.system_prompt_template
    assert "`read_file`" not in middleware.system_prompt_template

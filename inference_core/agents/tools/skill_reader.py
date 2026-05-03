"""Restricted tools for reading agent skill instructions."""

from collections.abc import Callable, Sequence
from pathlib import PurePosixPath
from typing import Annotated, Any

from deepagents.backends.protocol import BackendProtocol, ReadResult
from deepagents.backends.utils import (
    check_empty_content,
    format_content_with_line_numbers,
    validate_path,
)
from deepagents.middleware import SkillsMiddleware
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel, Field

READ_SKILL_FILE_TOOL_NAME = "read_skill_file"
DEFAULT_SKILL_READ_OFFSET = 0
DEFAULT_SKILL_READ_LIMIT = 1000
MAX_SKILL_READ_LIMIT = 2000

BackendLike = BackendProtocol | Callable[[ToolRuntime[Any, Any]], BackendProtocol]


SKILLS_SYSTEM_PROMPT = """

## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

{skills_locations}

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a progressive disclosure pattern. You see each skill's name and description above, and you should read the full instructions only when the user's task matches a listed skill.

1. Check whether the user's task matches a skill's description.
2. Read the skill's full instructions with `read_skill_file(file_path="<path from the skill list>", limit=1000)`.
3. Follow the workflow in `SKILL.md` exactly enough to preserve the skill's intent.

`read_skill_file` is intentionally limited to `SKILL.md` files from the skill locations listed above. It does not read arbitrary filesystem paths, user files, helper scripts, or non-skill documents.

**When to Use Skills:**
- The user's request matches a skill's domain.
- You need a structured workflow or specialized project guidance.
- A skill provides proven patterns for a complex task.

Remember: Skills make you more capable and consistent. When in doubt, check if a skill exists for the task.
"""


class ReadSkillFileSchema(BaseModel):
    """Input schema for the restricted skill reader."""

    file_path: str = Field(
        description="Path to a SKILL.md file exactly as shown in the available skills list."
    )
    offset: int = Field(
        default=DEFAULT_SKILL_READ_OFFSET,
        ge=0,
        description="Line number to start reading from (0-indexed).",
    )
    limit: int = Field(
        default=DEFAULT_SKILL_READ_LIMIT,
        ge=1,
        le=MAX_SKILL_READ_LIMIT,
        description="Maximum number of lines to read.",
    )


def create_skills_middleware(
    *, backend: BackendLike, sources: list[str]
) -> SkillsMiddleware:
    """Create SkillsMiddleware that points agents at the restricted reader."""
    middleware = SkillsMiddleware(backend=backend, sources=sources)
    middleware.system_prompt_template = SKILLS_SYSTEM_PROMPT
    return middleware


def add_skill_reader_tool(
    tools: list[Any], *, backend: BackendLike, sources: list[str]
) -> None:
    """Append the restricted skill reader unless a tool with the same name exists."""
    if any(getattr(tool, "name", None) == READ_SKILL_FILE_TOOL_NAME for tool in tools):
        return
    tools.append(create_skill_reader_tool(backend=backend, sources=sources))


def create_skill_reader_tool(*, backend: BackendLike, sources: list[str]) -> BaseTool:
    """Create a read-only tool that can only read configured `SKILL.md` files."""
    allowed_prefixes = _normalize_source_prefixes(sources)

    def _get_backend(runtime: ToolRuntime[Any, Any] | None) -> BackendProtocol:
        if callable(backend):
            if runtime is None:
                raise ValueError(
                    "read_skill_file requires tool runtime for this backend"
                )
            return backend(runtime)
        return backend

    def sync_read_skill_file(
        file_path: Annotated[str, "Path to a SKILL.md file from the skills list."],
        runtime: ToolRuntime[Any, Any] = None,
        offset: Annotated[
            int, "Line number to start reading from (0-indexed)."
        ] = DEFAULT_SKILL_READ_OFFSET,
        limit: Annotated[
            int, "Maximum number of lines to read."
        ] = DEFAULT_SKILL_READ_LIMIT,
    ) -> str:
        """Read a configured skill instruction file."""
        try:
            validated_path = _validate_skill_file_path(file_path, allowed_prefixes)
            validated_limit = _validate_limit(limit)
            read_result = _get_backend(runtime).read(
                validated_path,
                offset=offset,
                limit=validated_limit,
            )
        except ValueError as exc:
            return f"Error: {exc}"
        return _format_read_result(read_result, validated_path, offset)

    async def async_read_skill_file(
        file_path: Annotated[str, "Path to a SKILL.md file from the skills list."],
        runtime: ToolRuntime[Any, Any] = None,
        offset: Annotated[
            int, "Line number to start reading from (0-indexed)."
        ] = DEFAULT_SKILL_READ_OFFSET,
        limit: Annotated[
            int, "Maximum number of lines to read."
        ] = DEFAULT_SKILL_READ_LIMIT,
    ) -> str:
        """Async wrapper for reading a configured skill instruction file."""
        try:
            validated_path = _validate_skill_file_path(file_path, allowed_prefixes)
            validated_limit = _validate_limit(limit)
            read_result = await _get_backend(runtime).aread(
                validated_path,
                offset=offset,
                limit=validated_limit,
            )
        except ValueError as exc:
            return f"Error: {exc}"
        return _format_read_result(read_result, validated_path, offset)

    return StructuredTool.from_function(
        name=READ_SKILL_FILE_TOOL_NAME,
        description=(
            "Read the full instructions for an available agent skill. "
            "Only SKILL.md files under the configured skill sources are allowed."
        ),
        func=sync_read_skill_file,
        coroutine=async_read_skill_file,
        infer_schema=False,
        args_schema=ReadSkillFileSchema,
    )


def _normalize_source_prefixes(sources: Sequence[str]) -> list[str]:
    prefixes: list[str] = []
    for source in sources:
        normalized = validate_path(source)
        if normalized != "/" and not normalized.endswith("/"):
            normalized = f"{normalized}/"
        if normalized not in prefixes:
            prefixes.append(normalized)
    return prefixes


def _validate_skill_file_path(file_path: str, allowed_prefixes: Sequence[str]) -> str:
    if not allowed_prefixes:
        raise ValueError("no skill sources are configured")

    try:
        validated_path = validate_path(file_path, allowed_prefixes=allowed_prefixes)
    except ValueError:
        validated_path = _resolve_skill_relative_path(file_path, allowed_prefixes)

    _require_skill_markdown_path(validated_path)
    if _is_direct_skill_path(validated_path, allowed_prefixes):
        return validated_path

    raise ValueError(
        "skill path must be a direct SKILL.md child of a configured skill source"
    )


def _resolve_skill_relative_path(
    file_path: str, allowed_prefixes: Sequence[str]
) -> str:
    raw_path = file_path.strip()
    candidate_inputs = [raw_path]
    if raw_path and not raw_path.endswith("SKILL.md"):
        candidate_inputs.append(f"{raw_path.rstrip('/')}/SKILL.md")

    resolved_candidates: list[str] = []
    seen_candidates: set[str] = set()

    for candidate_input in candidate_inputs:
        normalized_candidate = validate_path(candidate_input)
        candidate_suffix = normalized_candidate.lstrip("/")

        for prefix in allowed_prefixes:
            prefixed_candidate = (
                f"{prefix}{candidate_suffix}"
                if prefix != "/"
                else f"/{candidate_suffix}"
            )
            normalized_prefixed = validate_path(
                prefixed_candidate,
                allowed_prefixes=allowed_prefixes,
            )
            if not _is_direct_skill_path(normalized_prefixed, allowed_prefixes):
                continue
            if normalized_prefixed in seen_candidates:
                continue
            seen_candidates.add(normalized_prefixed)
            resolved_candidates.append(normalized_prefixed)

    if len(resolved_candidates) == 1:
        return resolved_candidates[0]
    if len(resolved_candidates) > 1:
        raise ValueError(
            "skill path is ambiguous across configured skill sources; use the full path from the skill list"
        )

    raise ValueError(
        f"Path must start with one of {list(allowed_prefixes)} or be a skill-relative path like 'skill_name/SKILL.md': {file_path}"
    )


def _is_direct_skill_path(path: str, allowed_prefixes: Sequence[str]) -> bool:
    if PurePosixPath(path).name != "SKILL.md":
        return False

    for prefix in allowed_prefixes:
        relative_path = _relative_to_prefix(path, prefix)
        if relative_path is None:
            continue
        parts = PurePosixPath(relative_path).parts
        if len(parts) == 2 and parts[1] == "SKILL.md" and parts[0]:
            return True

    return False


def _require_skill_markdown_path(path: str) -> None:
    if PurePosixPath(path).name != "SKILL.md":
        raise ValueError("read_skill_file can only read SKILL.md files")


def _relative_to_prefix(path: str, prefix: str) -> str | None:
    if prefix == "/":
        return path.lstrip("/")
    if not path.startswith(prefix):
        return None
    return path[len(prefix) :]


def _validate_limit(limit: int) -> int:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    return min(limit, MAX_SKILL_READ_LIMIT)


def _format_read_result(
    read_result: ReadResult | str, file_path: str, offset: int
) -> str:
    if isinstance(read_result, str):
        return read_result

    if read_result.error:
        return f"Error: {read_result.error}"

    if read_result.file_data is None:
        return f"Error: no data returned for '{file_path}'"

    encoding = read_result.file_data.get("encoding", "utf-8")
    if encoding != "utf-8":
        return f"Error: '{file_path}' is not a text skill file"

    content = read_result.file_data.get("content", "")
    empty_msg = check_empty_content(content)
    if empty_msg:
        return empty_msg
    return format_content_with_line_numbers(content, start_line=offset + 1)


__all__ = [
    "READ_SKILL_FILE_TOOL_NAME",
    "add_skill_reader_tool",
    "create_skill_reader_tool",
    "create_skills_middleware",
]

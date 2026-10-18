"""
Memory Tools (Store-based) – CoALA Architecture

Provides LangChain BaseTool subclasses for agent memory CRUD operations,
organized following the CoALA (Cognitive Architectures for Language Agents)
framework with semantic / episodic / procedural memory categories.

Uses run_async_safely() to reuse the Celery worker loop when available,
avoiding creation of conflicting event loops in nested tool calls.

Source: CoALA whitepaper – arxiv:2309.02427
"""

import logging
from typing import Any, List, Optional, Sequence

from langchain_core.tools import BaseTool
from pydantic import Field

from inference_core.celery.async_utils import run_async_safely
from inference_core.services.agent_memory_service import (
    HELD,
    NOT_FOUND,
    READ_ONLY,
    REFUSED,
    REPEATED,
    REPEATED_REASONS,
    SAVED,
    VIA_TOOL,
    AgentMemoryStoreService,
    MemorySpaceError,
    MemoryType,
    MemoryWriteOutcome,
    format_memory_types_for_description,
    get_category_for_type,
    validate_memory_type,
)

logger = logging.getLogger(__name__)


def _shows_spaces(memory_service: Any) -> bool:
    """Whether the model is told about spaces: when it has a choice of them."""
    spaces = getattr(memory_service, "spaces", None) or ()
    return len(spaces) > 1


def _write_outcome_text(outcome: MemoryWriteOutcome, done: str) -> str:
    """What the model is told became of an update or a delete."""
    memory_id = outcome.memory_id
    if outcome.status == NOT_FOUND:
        return f"✗ Memory {memory_id} was not found."
    if outcome.status == READ_ONLY:
        space = outcome.space.key if outcome.space else "another space"
        return (
            f"✗ Memory {memory_id} is in space '{space}', which you can only "
            "read. It was not changed."
        )
    if outcome.status == REFUSED:
        return f"✗ Memory {memory_id} was not changed. {outcome.reason}".rstrip()
    if outcome.status == HELD:
        return (
            f"✓ The change to memory {memory_id} is kept for the user's approval. "
            "The memory stays as it was until they approve it."
        )
    return f"✓ {done} (id: {memory_id})"


def _space_note(memory_service: Any) -> str:
    """The writable spaces, for the save tool's description."""
    spaces = [
        space
        for space in getattr(memory_service, "spaces", None) or ()
        if space.writable
    ]
    if len(spaces) < 2:
        return ""
    lines = ["\nSpaces you can save to (argument `space`):"]
    for space in spaces:
        about = f": {space.description}" if space.description else ""
        lines.append(f"  - {space.key}{about}")
    return "\n".join(lines) + "\n"


def _resolve_tool_user_id(user_id: Optional[str]) -> str:
    """Return user_id or resolve it from the current runtime.

    WHY: On the Agent Server, tools are instantiated at graph-build time
    (module load) without a concrete user_id.  The actual value is injected
    per-request via ``RunnableConfig.configurable`` — we check that source
    directly because LangGraph tool-nodes may run in a different asyncio
    task than the middleware hook that populates ``_runtime_context``
    contextvars (e.g. subagent graphs, which skip InstanceConfigMiddleware
    by design), making the contextvar fallback unreliable.

    Resolution order:
      1. Explicit ``user_id`` passed to the tool at construction time.
      2. ``get_config()["configurable"]["user_id"]`` — always available
         inside a LangGraph run, unaffected by task boundaries.
      3. ``_runtime_context._user_id`` contextvar — legacy fallback for
         callers that populate it but do not set configurable.

    Raises:
        RuntimeError: When no user_id is available from any source.
    """
    if user_id:
        return user_id

    # Source 2: RunnableConfig.configurable — robust across task boundaries.
    try:
        from langgraph.config import get_config

        cfg = get_config() or {}
        configurable = cfg.get("configurable", {}) if isinstance(cfg, dict) else {}
        cfg_uid = configurable.get("user_id")
        if cfg_uid:
            return str(cfg_uid)
    except (RuntimeError, ImportError):
        # get_config() raises RuntimeError outside a LangGraph run; ImportError
        # guards against unexpected langgraph packaging changes.
        pass

    # Source 3: legacy contextvar set by InstanceConfigMiddleware.before_agent.
    from inference_core.agents.middleware._runtime_context import (
        get_user_id as _ctx_get_user_id,
    )

    ctx_uid = _ctx_get_user_id()
    if ctx_uid is not None:
        return str(ctx_uid)

    raise RuntimeError(
        "Memory tool requires user_id but none was provided, "
        "RunnableConfig.configurable['user_id'] is missing, and the "
        "runtime contextvar is empty.  Ensure user_id is forwarded "
        "in configurable metadata."
    )


class SaveMemoryStoreTool(BaseTool):
    """Tool for persisting user information into store-backed memory (CoALA-aware)."""

    name: str = "save_memory_store"
    description: str = f"""Save important information about the user to long-term memory.
Use when the user shares preferences, facts, instructions, or experiences to remember.

Arguments:
  - content (required): The information to save
  - memory_type (optional): Fine-grained type of memory. Default: 'general'
  - topic (optional): Subject or theme tag for easier retrieval
  - category (optional): CoALA category override (semantic/episodic/procedural).
                         Auto-resolved from memory_type when omitted.
  - space (optional): The memory space to save to, when you have more than one.
                      Without it, the memory goes to your default space.

{format_memory_types_for_description()}
"""

    memory_service: Any = Field(exclude=True)
    user_id: Optional[str] = Field(default=None, exclude=True)
    session_id: Optional[str] = Field(default=None, exclude=True)
    # Who writes through this tool: the model in its run, or the analysis
    # after it. The write policy and the memory's origin are told.
    via: str = Field(default=VIA_TOOL, exclude=True)

    def _run(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.GENERAL,
        topic: Optional[str] = None,
        category: Optional[str] = None,
        space: Optional[str] = None,
    ) -> str:
        try:
            uid = _resolve_tool_user_id(self.user_id)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to save memory: %s", exc)
            return f"✗ Failed to save memory: {exc}"
        return run_async_safely(
            self._save(uid, content, memory_type, topic, category, space)
        )

    async def _arun(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.GENERAL,
        topic: Optional[str] = None,
        category: Optional[str] = None,
        space: Optional[str] = None,
    ) -> str:
        try:
            uid = _resolve_tool_user_id(self.user_id)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to save memory: %s", exc)
            return f"✗ Failed to save memory: {exc}"
        return await self._save(uid, content, memory_type, topic, category, space)

    async def _save(
        self,
        uid: str,
        content: str,
        memory_type: Any,
        topic: Optional[str],
        category: Optional[str],
        space: Optional[str],
    ) -> str:
        try:
            validated_type = validate_memory_type(memory_type)
            resolved_cat = category or get_category_for_type(validated_type).value
            target = self.memory_service.space_for_write(resolved_cat, space)
            outcome = await self.memory_service.write_memory(
                user_id=uid,
                content=content,
                memory_type=validated_type,
                session_id=self.session_id,
                topic=topic,
                category=category,
                space=target.key,
                via=self.via,
            )
            if outcome.status == REFUSED:
                return f"✗ Memory not saved. {outcome.reason}".rstrip()
            memory_id = outcome.memory_id
            if outcome.status == REPEATED:
                text = f"✓ Not saved again. {outcome.reason} (id: {memory_id})"
                if outcome.reason == REPEATED_REASONS[SAVED]:
                    text += (
                        " If what you meant to save changes what it says, call "
                        "update_memory_store with that id."
                    )
                return text
            if outcome.status == HELD:
                return (
                    f"✓ Memory kept for the user's approval (id: {memory_id[:8]}...). "
                    "It is not used until the user approves it."
                )
            where = (
                f", space: {target.key}" if _shows_spaces(self.memory_service) else ""
            )
            return (
                f"✓ Memory saved (id: {memory_id[:8]}..., category: {resolved_cat}"
                f"{where})"
            )
        except MemorySpaceError as exc:
            return f"✗ {exc}"
        except ValueError as exc:
            logger.error("Invalid memory type: %s", exc)
            return f"✗ {exc}"
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to save memory: %s", exc)
            return f"✗ Failed to save memory: {exc}"


class RecallMemoryStoreTool(BaseTool):
    """Tool for retrieving store-backed user memories by semantic search (CoALA-aware)."""

    name: str = "recall_memories_store"
    description: str = f"""Search long-term memory for user preferences, facts, instructions, or experiences.
You can scope the search to a specific CoALA category for more precise results.

Arguments:
  - query (required): Search query describing what to find
  - memory_type (optional): Filter by specific type. If omitted, searches all types
  - max_results (optional): Maximum number of memories to return
  - category (optional): CoALA category to search (semantic/episodic/procedural).
                         If omitted, searches all categories.

{format_memory_types_for_description()}
"""

    memory_service: Any = Field(exclude=True)
    user_id: Optional[str] = Field(default=None, exclude=True)
    default_max_results: int = Field(default=5, exclude=True)

    def _run(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        max_results: Optional[int] = None,
        category: Optional[str] = None,
    ) -> str:
        try:
            uid = _resolve_tool_user_id(self.user_id)
            if memory_type:
                memory_type = validate_memory_type(memory_type)
            memories = run_async_safely(
                self.memory_service.recall_memories(
                    user_id=uid,
                    query=query,
                    k=max_results or self.default_max_results,
                    memory_type=memory_type,
                    category=category,
                )
            )
            return self._format_results(memories)
        except ValueError as exc:
            logger.error("Invalid memory type: %s", exc)
            return f"✗ {exc}"
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to recall memories: %s", exc)
            return f"✗ Failed to recall memories: {exc}"

    async def _arun(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        max_results: Optional[int] = None,
        category: Optional[str] = None,
    ) -> str:
        try:
            uid = _resolve_tool_user_id(self.user_id)
            if memory_type:
                memory_type = validate_memory_type(memory_type)
            memories = await self.memory_service.recall_memories(
                user_id=uid,
                query=query,
                k=max_results or self.default_max_results,
                memory_type=memory_type,
                category=category,
            )
            return self._format_results(memories)
        except ValueError as exc:
            logger.error("Invalid memory type: %s", exc)
            return f"✗ {exc}"
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to recall memories: %s", exc)
            return f"✗ Failed to recall memories: {exc}"

    def _format_results(self, memories: list) -> str:
        """Format recalled memories with temporal and category context."""
        if not memories:
            return "No relevant memories found."

        lines = [f"Found {len(memories)} relevant memories:"]
        for idx, mem in enumerate(memories, 1):
            score_str = (
                f" (relevance: {mem.score:.2f})" if getattr(mem, "score", None) else ""
            )
            mem_type = mem.memory_type or "general"
            mem_cat = getattr(mem, "memory_category", None) or ""
            mem_topic = mem.topic
            mem_id = getattr(mem, "id", "unknown")

            # Include created_at timestamp for temporal context
            created_at = getattr(mem, "created_at", None)
            created_str = ""
            if created_at:
                if hasattr(created_at, "strftime"):
                    created_str = f" [created: {created_at.strftime('%Y-%m-%d %H:%M')}]"
                elif isinstance(created_at, str):
                    created_str = f" [created: {created_at[:16]}]"

            cat_label = f" [{mem_cat}]" if mem_cat else ""
            space_label = ""
            space = getattr(mem, "space", None)
            if space is not None and _shows_spaces(self.memory_service):
                access = "" if space.writable else ", read only"
                space_label = f" [space: {space.key}{access}]"
            mem_string = (
                f"{idx}. [id: {mem_id}]{cat_label} [{mem_type}]{space_label}"
                f"{created_str} - "
            )
            if mem_topic:
                mem_string += f"({mem_topic}) "
            mem_string += f"{score_str}: {mem.content}"
            render = getattr(self.memory_service, "render_entry", None)
            if callable(render) and hasattr(mem, "saved_at"):
                rendered = render(mem, mem_string)
                if isinstance(rendered, str):
                    mem_string = rendered
            lines.append(mem_string)
        return "\n".join(lines)


class UpdateMemoryStoreTool(BaseTool):
    """Tool for updating existing user memories in store-backed memory (CoALA-aware)."""

    name: str = "update_memory_store"
    description: str = f"""Update an existing memory entry with new content or details.
Use when correcting information or adding details to a specific memory ID found via recall_memories_store.

Arguments:
  - memory_id (required): The unique identifier of the memory to update
  - content (required): The new information to save
  - memory_type (optional): Fine-grained type of memory. Default: 'general'
  - topic (optional): Subject or theme tag
  - category (optional): CoALA category override (semantic/episodic/procedural)

{format_memory_types_for_description()}
"""

    memory_service: Any = Field(exclude=True)
    user_id: Optional[str] = Field(default=None, exclude=True)
    session_id: Optional[str] = Field(default=None, exclude=True)
    via: str = Field(default=VIA_TOOL, exclude=True)

    def _run(
        self,
        memory_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.GENERAL,
        topic: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        try:
            uid = _resolve_tool_user_id(self.user_id)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to update memory: %s", exc)
            return f"✗ Failed to update memory: {exc}"
        return run_async_safely(
            self._update(uid, memory_id, content, memory_type, topic, category)
        )

    async def _arun(
        self,
        memory_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.GENERAL,
        topic: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        try:
            uid = _resolve_tool_user_id(self.user_id)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to update memory: %s", exc)
            return f"✗ Failed to update memory: {exc}"
        return await self._update(uid, memory_id, content, memory_type, topic, category)

    async def _update(
        self,
        uid: str,
        memory_id: str,
        content: str,
        memory_type: Any,
        topic: Optional[str],
        category: Optional[str],
    ) -> str:
        """Change the memory where it lies; never copy it elsewhere."""
        try:
            validated_type = validate_memory_type(memory_type)
            outcome = await self.memory_service.update_memory(
                user_id=uid,
                memory_id=memory_id,
                content=content,
                memory_type=validated_type,
                topic=topic,
                session_id=self.session_id,
                category=category,
                via=self.via,
            )
            return _write_outcome_text(outcome, "Memory updated successfully")
        except ValueError as exc:
            logger.error("Invalid memory type: %s", exc)
            return f"✗ {exc}"
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to update memory: %s", exc)
            return f"✗ Failed to update memory: {exc}"


class DeleteMemoryStoreTool(BaseTool):
    """Tool for deleting user memories from store-backed memory."""

    name: str = "delete_memory_store"
    description: str = """Delete a specific memory entry.
Use when information is incorrect, outdated, or when user requests to forget something.

Arguments:
  - memory_id (required): The unique identifier of the memory to delete (from recall_memories_store)
"""

    memory_service: Any = Field(exclude=True)
    user_id: Optional[str] = Field(default=None, exclude=True)

    def _run(self, memory_id: str) -> str:
        try:
            uid = _resolve_tool_user_id(self.user_id)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to delete memory: %s", exc)
            return f"✗ Failed to delete memory: {exc}"
        return run_async_safely(self._delete(uid, memory_id))

    async def _arun(self, memory_id: str) -> str:
        try:
            uid = _resolve_tool_user_id(self.user_id)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to delete memory: %s", exc)
            return f"✗ Failed to delete memory: {exc}"
        return await self._delete(uid, memory_id)

    async def _delete(self, uid: str, memory_id: str) -> str:
        """Delete the memory where it lies, if the agent may write there."""
        try:
            outcome = await self.memory_service.remove_memory(
                user_id=uid, memory_id=memory_id
            )
            return _write_outcome_text(outcome, "Memory deleted successfully")
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to delete memory: %s", exc)
            return f"✗ Failed to delete memory: {exc}"


def get_memory_tools(
    memory_service: "AgentMemoryStoreService",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    max_recall_results: int = 5,
    include_tools: Optional[List[str]] = None,
) -> List[BaseTool]:
    """Factory for store-backed memory tools to ease agent wiring.

    WHY: Supports per-agent memory tool filtering — only the tools whose
    names appear in ``include_tools`` are returned.  When ``include_tools``
    is ``None``, all four tools are returned (backward-compatible default).
    An empty list means no model-facing memory tools at all.

    When ``user_id`` is ``None`` (Agent Server), each tool resolves it at
    call time from the ``_runtime_context`` context var.
    """
    if include_tools is not None and len(include_tools) == 0:
        logger.debug("Memory tools disabled (include_tools=[])")
        return []

    save_tool = SaveMemoryStoreTool(
        memory_service=memory_service,
        user_id=user_id,
        session_id=session_id,
    )
    note = _space_note(memory_service)
    if note:
        save_tool.description = save_tool.description + note

    all_tools: dict[str, BaseTool] = {
        "save_memory_store": save_tool,
        "recall_memories_store": RecallMemoryStoreTool(
            memory_service=memory_service,
            user_id=user_id,
            default_max_results=max_recall_results,
        ),
        "update_memory_store": UpdateMemoryStoreTool(
            memory_service=memory_service,
            user_id=user_id,
            session_id=session_id,
        ),
        "delete_memory_store": DeleteMemoryStoreTool(
            memory_service=memory_service,
            user_id=user_id,
        ),
    }

    if include_tools is not None:
        selected = [all_tools[name] for name in include_tools if name in all_tools]
    else:
        selected = list(all_tools.values())

    logger.debug(
        "Created %d store memory tools for user=%s (filter=%s)",
        len(selected),
        user_id,
        include_tools,
    )

    return selected


def _spaces_block(
    spaces: Optional[Sequence[Any]], default_write: Optional[str]
) -> str:
    """The memory spaces an agent has, when it has a choice of them.

    A space's description is what tells the model where a memory belongs.
    """
    spaces = list(spaces or ())
    if len(spaces) < 2:
        return ""
    lines = [
        "### Memory spaces",
        "Your memory is kept in spaces. You read all of them and save only to "
        "the ones marked writable. Save each memory to the space whose "
        "description fits it, with the `space` argument.",
    ]
    for space in spaces:
        access = "writable" if space.writable else "read only"
        about = f" {space.description}" if space.description else ""
        lines.append(f"- `{space.key}` ({space.name}, {access}).{about}")
    if default_write:
        lines.append(f"Without `space`, a memory goes to `{default_write}`.")
    return "\n".join(lines) + "\n"


def generate_memory_tools_system_instructions(
    active_tool_names: Optional[List[str]] = None,
    spaces: Optional[Sequence[Any]] = None,
    default_write: Optional[str] = None,
) -> str:
    """Generate system prompt instructions explaining CoALA memory architecture.

    WHY: When a subset of memory tools is exposed, the instruction block must
    only describe those tools — otherwise the model hallucinates calls to
    tools it cannot invoke.

    Args:
        active_tool_names: Names of memory tools currently exposed to the model.
            When ``None``, instructions for all four tools are generated.
            When empty, returns an empty string.
        spaces: The agent's memory spaces (:class:`MemorySpace`). With more
            than one, the instructions say what each holds and which are
            writable.
        default_write: The space a memory goes to when none is named.

    Returns:
        Formatted instructions or empty string when no tools are active.
    """
    _ALL_TOOL_NAMES = {
        "save_memory_store",
        "recall_memories_store",
        "update_memory_store",
        "delete_memory_store",
    }
    if active_tool_names is not None:
        active = set(active_tool_names) & _ALL_TOOL_NAMES
    else:
        active = _ALL_TOOL_NAMES

    if not active:
        return ""

    # Build per-tool instruction sections
    tool_sections: list[str] = []

    if "save_memory_store" in active:
        tool_sections.append(
            "#### save_memory_store — persist information to long-term memory\n"
            "Arguments: content (required), memory_type (auto-categorized when omitted), topic, category."
        )

    if "recall_memories_store" in active:
        tool_sections.append(
            "#### recall_memories_store — retrieve memories by semantic search\n"
            "When to use:\n"
            "- Start of a new topic to load relevant user context (→ category: semantic)\n"
            "- Before giving personalized recommendations (→ semantic + procedural)\n"
            '- When user says "last time", "before", "we discussed" (→ category: episodic)\n'
            "- Always before calling update_memory_store (to get memory_id)"
        )

    if "update_memory_store" in active:
        tool_sections.append(
            "#### update_memory_store — correct or enrich an existing memory\n"
            "Requires memory_id from recall_memories_store first.\n"
            "Use when: user corrects saved info, preferences change, additional details emerge."
        )

    if "delete_memory_store" in active:
        tool_sections.append(
            "#### delete_memory_store — remove a specific memory\n"
            "Requires memory_id from recall_memories_store first.\n"
            "Use when: user asks to forget something, info is outdated or wrong."
        )

    tools_block = "\n\n".join(tool_sections)

    # Only include save triggers / decision rules when save_memory_store is active
    save_triggers = ""
    if "save_memory_store" in active:
        save_triggers = """
### Mandatory save triggers
You MUST call save_memory_store (BEFORE finalizing your answer) when any of these occur:

**Semantic — save immediately:**
- User states a preference: "I prefer X", "I like/hate X", "please use X format"
- User shares a fact about themselves: name, role, location, employer,
  project, language, stack, team, timezone
- User sets a goal: "I'm building X", "my goal is Y", "I need to achieve Z"
- User names a constraint: "I can't use X", "I only have access to Y"

**Procedural — save immediately:**
- User gives a future instruction: "from now on", "always do X", "never do Y",
  "in the future", "next time", "every time"
- User corrects a mistake you made → save what went wrong and the correct way
- You identify a pattern in how the user works or communicates

**Episodic — save before closing a topic:**
- The conversation reaches a concrete decision, agreed plan, or confirmed outcome
- User explicitly asks you to remember something specific from this session
- You complete a multi-step task and the result is worth preserving

### Proactive saves (even without explicit request)
These MUST be saved even if the user never said "remember this":
- User corrects your factual assumption or output style → procedural/instructions
- User adapts your behavior mid-conversation via feedback → semantic/preferences
- A decision made in this session will clearly affect the next interaction → episodic/interaction
- You successfully apply a non-obvious technique → procedural/skill

### What NOT to save
Do NOT call save_memory_store for:
- Temporary context that is only relevant for the current question
- General knowledge the user is asking about (not about themselves or their preferences)
- Step-by-step task progress that belongs in the response, not memory
- Greeting phrases or social pleasantries with no informational value

### Decision rule (apply before every response)
Before finalizing your answer, ask yourself:
  "Did I learn anything about this user's preferences, facts, goals, instructions,
   or decisions that will matter in a future session?"
  If YES → call save_memory_store first, then respond.
  If NO  → respond directly.
This check must happen EVERY turn, not only when the user asks explicitly.
"""

    spaces_block = _spaces_block(spaces, default_write)
    if spaces_block:
        spaces_block = "\n" + spaces_block

    return f"""## Memory Tools Usage (CoALA Architecture)

### Guiding principle
Memory is part of how you work — not something you announce. Save information silently
and use it naturally, the way a person with good memory would. Never say "I've saved
 that" or "I'm noting this down" unless the user explicitly asks "did you remember/save
that?". The act of saving is invisible to the user.

### Memory categories

**SEMANTIC MEMORY** — Stable facts about the world and the user.
  Types: preferences, facts, goals, general.

**EPISODIC MEMORY** — Sequences of past experiences and interactions.
  Types: context, session_summary, interaction.

**PROCEDURAL MEMORY** — Operational rules, workflows, and skills.
  Types: instructions, workflow, skill.
{spaces_block}{save_triggers}
### Tools

{tools_block}

{format_memory_types_for_description()}

### Example workflow
1. User asks a question → recall semantic memories for context (if relevant)
2. User shares preference or fact → save_memory_store BEFORE responding
3. Use recalled + saved context to personalize the response
4. User corrects info → recall to get memory_id, then update_memory_store
5. Session ends with a clear outcome → save session_summary as episodic memory
"""


__all__ = [
    "SaveMemoryStoreTool",
    "RecallMemoryStoreTool",
    "get_memory_tools",
    "generate_memory_tools_system_instructions",
]

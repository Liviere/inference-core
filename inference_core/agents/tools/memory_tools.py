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
from typing import Any, List, Optional

from langchain_core.tools import BaseTool
from pydantic import Field

from inference_core.celery.async_utils import run_async_safely
from inference_core.services.agent_memory_service import (
    AgentMemoryStoreService,
    MemoryType,
    format_memory_types_for_description,
    get_category_for_type,
    validate_memory_type,
)

logger = logging.getLogger(__name__)


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

{format_memory_types_for_description()}
"""

    memory_service: Any = Field(exclude=True)
    user_id: str = Field(exclude=True)
    session_id: Optional[str] = Field(default=None, exclude=True)

    def _run(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.GENERAL,
        topic: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        try:
            validated_type = validate_memory_type(memory_type)
            memory_id = run_async_safely(
                self.memory_service.save_memory(
                    user_id=self.user_id,
                    content=content,
                    memory_type=validated_type,
                    session_id=self.session_id,
                    topic=topic,
                    category=category,
                )
            )
            resolved_cat = category or get_category_for_type(validated_type).value
            return f"✓ Memory saved (id: {memory_id[:8]}..., category: {resolved_cat})"
        except ValueError as exc:
            logger.error("Invalid memory type: %s", exc)
            return f"✗ {exc}"
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to save memory: %s", exc)
            return f"✗ Failed to save memory: {exc}"

    async def _arun(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.GENERAL,
        topic: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        try:
            validated_type = validate_memory_type(memory_type)
            memory_id = await self.memory_service.save_memory(
                user_id=self.user_id,
                content=content,
                memory_type=validated_type,
                session_id=self.session_id,
                topic=topic,
                category=category,
            )
            resolved_cat = category or get_category_for_type(validated_type).value
            return f"✓ Memory saved (id: {memory_id[:8]}..., category: {resolved_cat})"
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
    user_id: str = Field(exclude=True)
    default_max_results: int = Field(default=5, exclude=True)

    def _run(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        max_results: Optional[int] = None,
        category: Optional[str] = None,
    ) -> str:
        try:
            if memory_type:
                memory_type = validate_memory_type(memory_type)
            memories = run_async_safely(
                self.memory_service.recall_memories(
                    user_id=self.user_id,
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
            if memory_type:
                memory_type = validate_memory_type(memory_type)
            memories = await self.memory_service.recall_memories(
                user_id=self.user_id,
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
            mem_string = (
                f"{idx}. [id: {mem_id}]{cat_label} [{mem_type}]{created_str} - "
            )
            if mem_topic:
                mem_string += f"({mem_topic}) "
            mem_string += f"{score_str}: {mem.content}"
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
    user_id: str = Field(exclude=True)
    session_id: Optional[str] = Field(default=None, exclude=True)

    def _run(
        self,
        memory_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.GENERAL,
        topic: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        try:
            validated_type = validate_memory_type(memory_type)
            updated_id = run_async_safely(
                self.memory_service.save_memory(
                    user_id=self.user_id,
                    content=content,
                    memory_type=validated_type,
                    session_id=self.session_id,
                    topic=topic,
                    memory_id=memory_id,
                    category=category,
                )
            )
            return f"✓ Memory updated successfully (id: {updated_id})"
        except ValueError as exc:
            logger.error("Invalid memory type: %s", exc)
            return f"✗ {exc}"
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to update memory: %s", exc)
            return f"✗ Failed to update memory: {exc}"

    async def _arun(
        self,
        memory_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.GENERAL,
        topic: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        try:
            validated_type = validate_memory_type(memory_type)
            updated_id = await self.memory_service.save_memory(
                user_id=self.user_id,
                content=content,
                memory_type=validated_type,
                session_id=self.session_id,
                topic=topic,
                memory_id=memory_id,
                category=category,
            )
            return f"✓ Memory updated successfully (id: {updated_id})"
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
    user_id: str = Field(exclude=True)

    def _run(self, memory_id: str) -> str:
        try:
            success = run_async_safely(
                self.memory_service.delete_memory(
                    user_id=self.user_id,
                    memory_id=memory_id,
                )
            )
            if success:
                return f"✓ Memory deleted successfully (id: {memory_id})"
            return f"✗ Failed to delete memory (id: {memory_id} not found or error)"
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to delete memory: %s", exc)
            return f"✗ Failed to delete memory: {exc}"

    async def _arun(self, memory_id: str) -> str:
        try:
            success = await self.memory_service.delete_memory(
                user_id=self.user_id,
                memory_id=memory_id,
            )
            if success:
                return f"✓ Memory deleted successfully (id: {memory_id})"
            return f"✗ Failed to delete memory (id: {memory_id} not found or error)"
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to delete memory: %s", exc)
            return f"✗ Failed to delete memory: {exc}"


def get_memory_tools(
    memory_service: "AgentMemoryStoreService",
    user_id: str,
    session_id: Optional[str] = None,
    max_recall_results: int = 5,
) -> List[BaseTool]:
    """Factory for store-backed memory tools to ease agent wiring."""

    save_tool = SaveMemoryStoreTool(
        memory_service=memory_service,
        user_id=user_id,
        session_id=session_id,
    )

    recall_tool = RecallMemoryStoreTool(
        memory_service=memory_service,
        user_id=user_id,
        default_max_results=max_recall_results,
    )

    update_tool = UpdateMemoryStoreTool(
        memory_service=memory_service,
        user_id=user_id,
        session_id=session_id,
    )

    delete_tool = DeleteMemoryStoreTool(
        memory_service=memory_service,
        user_id=user_id,
    )

    logger.debug(
        "Created store memory tools for user=%s",
        user_id,
    )

    return [save_tool, recall_tool, update_tool, delete_tool]


def generate_memory_tools_system_instructions() -> str:
    """Generate system prompt instructions explaining CoALA memory architecture.

    Returns formatted instructions explaining the three memory categories
    and how/when to use each memory tool.
    """
    return f"""## Memory Tools Usage (CoALA Architecture)

NOTE: Memory tools are internal cognitive processes. 
Never verbally confirm, announce, or reference the act of saving or retrieving 
information unless the user explicitly asks "did you remember/save that?". 
Just use the information naturally, the way a human would.

Your long-term memory is organized following the Cognitive Architectures for Language Agents (CoALA) framework
into three categories:

### Memory Categories

**SEMANTIC MEMORY** — Stable facts about the world and the user.
  Types: preferences, facts, goals, general.
  Shared across all agents for this user.

**EPISODIC MEMORY** — Sequences of past experiences and interactions.
  Types: context, session_summary, interaction.
  Scoped per-agent (each agent has its own history).

**PROCEDURAL MEMORY** — Operational rules, workflows, and learned skills.
  Types: instructions, workflow, skill.
  Scoped per-agent (agent-specific capabilities).

### Tools

#### save_memory_store
Save important information to long-term memory.
**When to use:**
- User explicitly asks you to remember something (→ semantic/preferences or facts)
- User shares personal preferences, facts, goals (→ semantic)
- User gives instructions for future interactions (→ procedural/instructions)
- You learn a useful technique or pattern (→ procedural/skill)
- A conversation yields important decisions or outcomes (→ episodic/session_summary)

#### recall_memories_store
Retrieve relevant memories. Use `category` to scope search.
**When to use:**
- At the start of conversations to understand user context (→ category: semantic)
- When providing recommendations (→ semantic + procedural)
- When user references past interactions (→ category: episodic)
- Before updating/deleting memories (to get memory_id)

### update_memory_store
Use this tool to correct or update existing memories.
**When to use:**
- User corrects previously saved information
- Additional details need to be added to an existing memory
- Preferences change over time
- **Requires memory_id obtained from recall_memories_store**

### delete_memory_store
Use this tool to remove specific memories.
**When to use:**
- User asks to forget specific information
- Information is no longer valid or relevant and shouldn't be kept
- **Requires memory_id obtained from recall_memories_store**

**Best practices:**
- Use `category` parameter to scope searches for precision
- Save stable user facts as semantic, interaction history as episodic
- Save learned agent skills and rules as procedural
- Query memories proactively to personalize responses

{format_memory_types_for_description()}

**Example workflow:**
1. User asks a question → recall semantic memories for context
2. Use recalled context to personalize response
3. If user shares new info → save with appropriate type (auto-categorized)
4. If user corrects info → recall to find ID, then update
5. At session end → optionally save session_summary as episodic memory
"""


__all__ = [
    "SaveMemoryStoreTool",
    "RecallMemoryStoreTool",
    "get_memory_tools",
    "generate_memory_tools_system_instructions",
]

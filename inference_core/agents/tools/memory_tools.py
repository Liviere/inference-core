"""
Memory Tools (Store-based)

Replicates the original memory tools but uses LangGraph Store-backed
AgentMemoryStoreService for persistence instead of vector stores.

Uses run_async_safely() to reuse the Celery worker loop when available,
avoiding creation of conflicting event loops in nested tool calls.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from langchain_core.tools import BaseTool
from pydantic import Field

from inference_core.celery.async_utils import run_async_safely
from inference_core.services.agent_memory_service import (
    AgentMemoryStoreService,
    MemoryType,
    format_memory_types_for_description,
    validate_memory_type,
)

logger = logging.getLogger(__name__)


class SaveMemoryStoreTool(BaseTool):
    """Tool for persisting user information into store-backed memory."""

    name: str = "save_memory_store"
    description: str = f"""Save important information about the user to store-backed long-term memory.
Use when the user shares preferences, facts, or instructions to remember.

Arguments:
  - content (required): The information to save
  - memory_type (optional): Category of memory. Default: 'general'
  - topic (optional): Subject or theme tag for easier retrieval

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
    ) -> str:
        try:
            # Runtime validation against canonical enum
            validated_type = validate_memory_type(memory_type)
            memory_id = run_async_safely(
                self.memory_service.save_memory(
                    user_id=self.user_id,
                    content=content,
                    memory_type=validated_type,
                    session_id=self.session_id,
                    topic=topic,
                )
            )
            return f"✓ Memory saved successfully (id: {memory_id[:8]}...)"
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
    ) -> str:
        try:
            # Runtime validation against canonical enum
            validated_type = validate_memory_type(memory_type)
            memory_id = await self.memory_service.save_memory(
                user_id=self.user_id,
                content=content,
                memory_type=validated_type,
                session_id=self.session_id,
                topic=topic,
            )
            return f"✓ Memory saved successfully (id: {memory_id[:8]}...)"
        except ValueError as exc:
            logger.error("Invalid memory type: %s", exc)
            return f"✗ {exc}"
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to save memory: %s", exc)
            return f"✗ Failed to save memory: {exc}"


class RecallMemoryStoreTool(BaseTool):
    """Tool for retrieving store-backed user memories by semantic search."""

    name: str = "recall_memories_store"
    description: str = f"""Search store-backed long-term memory for user preferences, facts, or context.

Arguments:
  - query (required): Search query describing what to find
  - memory_type (optional): Filter by specific category. If omitted, searches all types
  - max_results (optional): Maximum number of memories to return

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
    ) -> str:
        try:
            # Runtime validation if memory_type provided
            if memory_type:
                memory_type = validate_memory_type(memory_type)
            memories = run_async_safely(
                self.memory_service.recall_memories(
                    user_id=self.user_id,
                    query=query,
                    k=max_results or self.default_max_results,
                    memory_type=memory_type,
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
    ) -> str:
        try:
            # Runtime validation if memory_type provided
            if memory_type:
                memory_type = validate_memory_type(memory_type)
            memories = await self.memory_service.recall_memories(
                user_id=self.user_id,
                query=query,
                k=max_results or self.default_max_results,
                memory_type=memory_type,
            )
            return self._format_results(memories)
        except ValueError as exc:
            logger.error("Invalid memory type: %s", exc)
            return f"✗ {exc}"
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to recall memories: %s", exc)
            return f"✗ Failed to recall memories: {exc}"

    def _format_results(self, memories: list) -> str:
        """Format recalled memories with temporal context.

        Includes created_at timestamp for each memory to give the agent
        awareness of when information was saved.
        """
        if not memories:
            return "No relevant memories found."

        lines = [f"Found {len(memories)} relevant memories:"]
        for idx, mem in enumerate(memories, 1):
            score_str = (
                f" (relevance: {mem.score:.2f})" if getattr(mem, "score", None) else ""
            )
            mem_type = mem.memory_type or "general"
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

            mem_string = f"{idx}. [id: {mem_id}] [{mem_type}]{created_str} - "
            if mem_topic:
                mem_string += f"({mem_topic}) "
            mem_string += f"{score_str}: {mem.content}"
            lines.append(mem_string)
        return "\n".join(lines)


class UpdateMemoryStoreTool(BaseTool):
    """Tool for updating existing user memories in store-backed memory."""

    name: str = "update_memory_store"
    description: str = f"""Update an existing memory entry with new content or details.
Use when correcting information or adding details to a specific memory ID found via recall_memories_store.

Arguments:
  - memory_id (required): The unique identifier of the memory to update
  - content (required): The new information to save
  - memory_type (optional): Category of memory. Default: 'general'
  - topic (optional): Subject or theme tag

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
    ) -> str:
        try:
            # Runtime validation against canonical enum
            validated_type = validate_memory_type(memory_type)
            updated_id = run_async_safely(
                self.memory_service.save_memory(
                    user_id=self.user_id,
                    content=content,
                    memory_type=validated_type,
                    session_id=self.session_id,
                    topic=topic,
                    memory_id=memory_id,
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
    ) -> str:
        try:
            # Runtime validation against canonical enum
            validated_type = validate_memory_type(memory_type)
            updated_id = await self.memory_service.save_memory(
                user_id=self.user_id,
                content=content,
                memory_type=validated_type,
                session_id=self.session_id,
                topic=topic,
                memory_id=memory_id,
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
    """Generate system prompt instructions for memory tools usage.

    Returns formatted instructions explaining when and how to use
    save_memory_store, recall_memories_store, update_memory_store, and delete_memory_store tools.

    Use this to append to agent system prompts when memory is enabled.
    """
    return f"""## Memory Tools Usage

You have access to long-term memory tools to remember important information about the user:

### save_memory_store
Use this tool to save important information that should be remembered for future conversations.
**When to use:**
- User explicitly asks you to remember something
- User shares personal preferences (communication style, format preferences, etc.)
- User provides factual information (location, job, interests, etc.)
- User gives specific instructions for future interactions
- User mentions goals or objectives they're working towards

**Do NOT use for:**
- Temporary context within the same conversation
- General knowledge or facts not specific to the user
- Information that's already common knowledge

### recall_memories_store
Use this tool to retrieve relevant memories before responding to user requests.
**When to use:**
- At the start of conversations to understand user preferences
- Before providing recommendations or suggestions
- When context from previous interactions would be helpful
- When user references something from past conversations
- **Before updating or deleting memories, to get the correct memory_id**

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
- Query memories proactively to personalize responses
- Use specific queries to find relevant information and IDs
- Filter by memory_type when looking for specific categories

{format_memory_types_for_description()}

**Example workflow:**
1. User asks a question → recall relevant memories first
2. Use recalled context to personalize your response
3. If user shares new information → save it with appropriate memory_type
4. If user corrects information → recall to find ID, then update or delete
5. Continue conversation with personalized context
"""


__all__ = [
    "SaveMemoryStoreTool",
    "RecallMemoryStoreTool",
    "get_memory_tools",
    "generate_memory_tools_system_instructions",
]

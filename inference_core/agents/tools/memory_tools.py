"""
Memory Tools for LangChain v1 Agents

Provides tools for agents to save and recall long-term memories.
These tools integrate with AgentMemoryService for vector-based storage.

Tools:
- save_memory: Save important information about the user
- recall_memories: Search for relevant memories

Usage:
    from inference_core.agents.tools.memory_tools import get_memory_tools

    # In AgentService
    tools = get_memory_tools(
        memory_service=memory_service,
        user_id="user-uuid",
    )
    agent = create_agent(model, tools=tools, ...)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import TYPE_CHECKING, Any, List, Literal, Optional

from langchain_core.tools import BaseTool
from pydantic import Field

if TYPE_CHECKING:
    from inference_core.services.agent_memory_service import AgentMemoryService

logger = logging.getLogger(__name__)


def _run_async_in_thread(coro):
    """Run an async coroutine in a dedicated thread with its own event loop.

    This is necessary because LangChain tool _run methods are synchronous,
    but our memory service is async. Using asyncio.run() doesn't work when
    already inside an event loop (e.g., Jupyter notebooks).
    """

    def _run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_thread)
        return future.result(timeout=30.0)


class SaveMemoryTool(BaseTool):
    """
    Tool for agents to save important information to long-term memory.

    The agent can use this tool to remember user preferences, facts,
    or other important context for future conversations.
    """

    name: str = "save_memory"
    description: str = """Save important information about the user to long-term memory.
Use this tool when the user shares preferences, personal facts, or instructions
that should be remembered for future conversations.

Args:
    content: The information to remember (be specific and clear)
    memory_type: Category - one of: preference, fact, context, instruction, general
    topic: Optional topic/category for organization (e.g., "food", "work", "health")

Examples:
- User says they prefer dark mode → save_memory("User prefers dark mode UI", "preference", "ui")
- User mentions their name is John → save_memory("User's name is John", "fact", "personal")
- User asks to always use formal language → save_memory("User wants formal language", "instruction")
"""

    # Configuration fields (not part of tool args)
    memory_service: Any = Field(exclude=True)  # AgentMemoryService
    user_id: str = Field(exclude=True)
    session_id: Optional[str] = Field(default=None, exclude=True)
    upsert_mode: bool = Field(default=False, exclude=True)

    def _run(
        self,
        content: str,
        memory_type: Literal[
            "preference", "fact", "context", "instruction", "general"
        ] = "general",
        topic: Optional[str] = None,
    ) -> str:
        """Save memory synchronously using ThreadPoolExecutor."""
        try:
            memory_id = _run_async_in_thread(
                self.memory_service.save_memory(
                    user_id=self.user_id,
                    content=content,
                    memory_type=memory_type,
                    session_id=self.session_id,
                    topic=topic,
                    upsert_by_similarity=self.upsert_mode,
                )
            )
            return f"✓ Memory saved successfully (id: {memory_id[:8]}...)"
        except Exception as e:
            logger.error("Failed to save memory: %s", e)
            return f"✗ Failed to save memory: {str(e)}"

    async def _arun(
        self,
        content: str,
        memory_type: Literal[
            "preference", "fact", "context", "instruction", "general"
        ] = "general",
        topic: Optional[str] = None,
    ) -> str:
        """Save memory asynchronously."""
        try:
            memory_id = await self.memory_service.save_memory(
                user_id=self.user_id,
                content=content,
                memory_type=memory_type,
                session_id=self.session_id,
                topic=topic,
                upsert_by_similarity=self.upsert_mode,
            )
            return f"✓ Memory saved successfully (id: {memory_id[:8]}...)"
        except Exception as e:
            logger.error("Failed to save memory: %s", e)
            return f"✗ Failed to save memory: {str(e)}"


class RecallMemoryTool(BaseTool):
    """
    Tool for agents to search and recall relevant memories.

    The agent can use this tool to retrieve previously stored information
    about the user based on semantic similarity.
    """

    name: str = "recall_memories"
    description: str = """Search long-term memory for relevant information about the user.
Use this tool when you need to recall user preferences, facts, or previous context.

Args:
    query: What to search for (describe what information you need)
    memory_type: Optional filter by type (preference, fact, context, instruction, general)
    max_results: How many memories to retrieve (default: 5)

Examples:
- Need user's UI preferences → recall_memories("UI preferences", "preference")
- Need user's name → recall_memories("user name", "fact")
- General context about user → recall_memories("user background")
"""

    # Configuration fields
    memory_service: Any = Field(exclude=True)  # AgentMemoryService
    user_id: str = Field(exclude=True)
    default_max_results: int = Field(default=5, exclude=True)

    def _run(
        self,
        query: str,
        memory_type: Optional[
            Literal["preference", "fact", "context", "instruction", "general"]
        ] = None,
        max_results: Optional[int] = None,
    ) -> str:
        """Recall memories synchronously using ThreadPoolExecutor."""
        try:
            memories = _run_async_in_thread(
                self.memory_service.recall_memories(
                    user_id=self.user_id,
                    query=query,
                    k=max_results or self.default_max_results,
                    memory_type=memory_type,
                )
            )
            return self._format_results(memories)
        except Exception as e:
            logger.error("Failed to recall memories: %s", e)
            return f"✗ Failed to recall memories: {str(e)}"

    async def _arun(
        self,
        query: str,
        memory_type: Optional[
            Literal["preference", "fact", "context", "instruction", "general"]
        ] = None,
        max_results: Optional[int] = None,
    ) -> str:
        """Recall memories asynchronously."""
        try:
            memories = await self.memory_service.recall_memories(
                user_id=self.user_id,
                query=query,
                k=max_results or self.default_max_results,
                memory_type=memory_type,
            )
            return self._format_results(memories)
        except Exception as e:
            logger.error("Failed to recall memories: %s", e)
            return f"✗ Failed to recall memories: {str(e)}"

    def _format_results(self, memories: list) -> str:
        """Format memory results for agent consumption."""
        if not memories:
            return "No relevant memories found."

        lines = [f"Found {len(memories)} relevant memories:"]
        for i, mem in enumerate(memories, 1):
            score_str = f" (relevance: {mem.score:.2f})" if mem.score else ""
            mem_type = mem.metadata.get("memory_type", "unknown")
            lines.append(f"{i}. [{mem_type}]{score_str}: {mem.content}")

        return "\n".join(lines)


def get_memory_tools(
    memory_service: "AgentMemoryService",
    user_id: str,
    session_id: Optional[str] = None,
    upsert_mode: bool = False,
    max_recall_results: int = 5,
) -> List[BaseTool]:
    """
    Factory function to create memory tools for an agent.

    Args:
        memory_service: AgentMemoryService instance
        user_id: User identifier for namespace isolation
        session_id: Optional session identifier for metadata
        upsert_mode: If True, check similarity before saving (avoid duplicates)
        max_recall_results: Default max results for recall queries

    Returns:
        List of configured memory tools [SaveMemoryTool, RecallMemoryTool]
    """
    save_tool = SaveMemoryTool(
        memory_service=memory_service,
        user_id=user_id,
        session_id=session_id,
        upsert_mode=upsert_mode,
    )

    recall_tool = RecallMemoryTool(
        memory_service=memory_service,
        user_id=user_id,
        default_max_results=max_recall_results,
    )

    logger.debug(
        "Created memory tools for user=%s, upsert=%s",
        user_id,
        upsert_mode,
    )

    return [save_tool, recall_tool]


__all__ = [
    "SaveMemoryTool",
    "RecallMemoryTool",
    "get_memory_tools",
]

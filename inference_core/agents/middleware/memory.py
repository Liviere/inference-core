"""
Memory Middleware for LangChain v1 Agents.

This middleware provides automatic memory context injection for agents using
the LangChain v1 middleware API. It hooks into the before_agent event to
recall relevant user memories and inject them as context.

Key features:
- Auto-recalls relevant memories based on user input
- Injects memory context into agent state
- Tracks memory operations for observability
- Uses ThreadPoolExecutor for async memory operations in sync hooks
  (compatible with Jupyter notebooks and other async contexts)

Usage:
    from langchain.agents import create_agent
    from inference_core.agents.middleware import MemoryMiddleware

    middleware = MemoryMiddleware(
        memory_service=memory_service,
        user_id="user-uuid",
        auto_recall=True,
    )

    agent = create_agent(
        model="gpt-4o",
        tools=[...],
        middleware=[middleware],
    )

Source references:
  - LangChain v1 middleware docs: context-engineering, long-term-memory
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from inference_core.services.agent_memory_service import AgentMemoryService

logger = logging.getLogger(__name__)


class MemoryState(AgentState):
    """Extended agent state for memory middleware.

    This state schema extends the base AgentState with fields for tracking
    memory operations and injected context.

    Attributes:
        memory_context: Formatted memory context string injected into conversation
        memories_recalled: Number of memories retrieved during recall
        memory_recall_latency_ms: Time taken for memory recall in milliseconds
        memory_types_recalled: List of memory types that were retrieved
    """

    memory_context: NotRequired[str]
    memories_recalled: NotRequired[int]
    memory_recall_latency_ms: NotRequired[float]
    memory_types_recalled: NotRequired[List[str]]


@dataclass
class _MemoryMiddlewareContext:
    """Internal context for tracking memory operations across hooks."""

    recall_start_time: Optional[float] = None
    recalled_memories: List[Any] = field(default_factory=list)
    context_injected: bool = False


class MemoryMiddleware(AgentMiddleware[MemoryState]):
    """Middleware for automatic memory context injection.

    This middleware automatically recalls relevant memories for the user
    at the start of agent execution and injects them as context.

    The middleware uses the following hooks:
    - before_agent: Recall relevant memories and inject context

    Attributes:
        state_schema: The custom state schema with memory tracking fields
        memory_service: AgentMemoryService instance for memory operations
        user_id: User ID for memory namespace isolation
        auto_recall: Whether to automatically recall memories in before_agent
        max_recall_results: Maximum number of memories to recall
        include_memory_types: Memory types to include in context recall
    """

    state_schema = MemoryState

    def __init__(
        self,
        memory_service: "AgentMemoryService",
        user_id: str,
        auto_recall: bool = True,
        max_recall_results: int = 5,
        include_memory_types: Optional[List[str]] = None,
    ):
        """Initialize the memory middleware.

        Args:
            memory_service: AgentMemoryService instance for memory operations.
            user_id: User ID for memory namespace isolation.
            auto_recall: Whether to automatically recall memories in before_agent.
                        If False, memories can still be accessed via tools.
            max_recall_results: Maximum number of memories to recall.
            include_memory_types: Memory types to include in context.
                                 Defaults to ["preference", "fact", "instruction"].
        """
        self.memory_service = memory_service
        self.user_id = user_id
        self.auto_recall = auto_recall
        self.max_recall_results = max_recall_results
        self.include_memory_types = include_memory_types or [
            "preference",
            "fact",
            "instruction",
        ]

        # Per-invocation context
        self._ctx: Optional[_MemoryMiddlewareContext] = None

    # -------------------------------------------------------------------------
    # Node-style hooks
    # -------------------------------------------------------------------------

    def before_agent(
        self, state: MemoryState, runtime: Runtime
    ) -> Dict[str, Any] | None:
        """Recall relevant memories and inject context at start of execution.

        This hook:
        1. Extracts the user's input from the messages
        2. Recalls relevant memories based on the input
        3. Formats memories as context string
        4. Returns state updates with memory context

        Args:
            state: Current agent state with messages
            runtime: Agent runtime context

        Returns:
            State updates with memory_context and tracking metrics,
            or None if auto_recall is disabled or no memories found.
        """
        if not self.auto_recall:
            logger.debug("Memory auto-recall disabled, skipping")
            return None

        # Initialize context
        self._ctx = _MemoryMiddlewareContext()
        self._ctx.recall_start_time = time.monotonic()

        # Extract user input from messages
        user_input = self._extract_user_input(state)
        if not user_input:
            logger.debug("No user input found in state, skipping memory recall")
            return None

        try:
            # Recall memories using asyncio.run() since hooks are synchronous
            memory_context, metrics = self._recall_and_format(user_input)

            if not memory_context:
                logger.debug("No relevant memories found for user=%s", self.user_id)
                return {
                    "memories_recalled": 0,
                    "memory_recall_latency_ms": metrics.get("latency_ms", 0),
                }

            self._ctx.context_injected = True

            logger.info(
                "Injected memory context: %d memories for user=%s (%.1fms)",
                metrics.get("count", 0),
                self.user_id,
                metrics.get("latency_ms", 0),
            )

            return {
                "memory_context": memory_context,
                "memories_recalled": metrics.get("count", 0),
                "memory_recall_latency_ms": metrics.get("latency_ms", 0),
                "memory_types_recalled": metrics.get("types", []),
            }

        except Exception as e:
            logger.error("Failed to recall memories: %s", e, exc_info=True)
            return {
                "memories_recalled": 0,
                "memory_context": "",
            }

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _extract_user_input(self, state: MemoryState) -> Optional[str]:
        """Extract the latest user input from state messages.

        Args:
            state: Agent state with messages

        Returns:
            User input string or None if not found
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        # Find the last human/user message
        for msg in reversed(messages):
            # Handle different message types
            if hasattr(msg, "type") and msg.type == "human":
                return msg.content
            elif hasattr(msg, "role") and msg.role == "user":
                return msg.content
            elif isinstance(msg, dict):
                if msg.get("type") == "human" or msg.get("role") == "user":
                    return msg.get("content", "")

        # Fallback: use last message content if it exists
        last_msg = messages[-1]
        if hasattr(last_msg, "content"):
            return last_msg.content
        elif isinstance(last_msg, dict):
            return last_msg.get("content", "")

        return None

    def _recall_and_format(self, user_input: str) -> tuple[str, Dict[str, Any]]:
        """Recall memories and format as context string.

        Uses ThreadPoolExecutor with a dedicated event loop to execute async
        memory operations from sync hook. This approach is compatible with
        Jupyter notebooks and other environments with existing event loops.

        Args:
            user_input: User's query for relevance-based recall

        Returns:
            Tuple of (formatted_context, metrics_dict)
        """
        start_time = time.monotonic()

        async def _async_recall():
            """Async function to perform memory recall operations."""
            context = await self.memory_service.format_context_for_prompt(
                user_id=self.user_id,
                query=user_input,
                include_types=self.include_memory_types,
            )

            memories_by_type = await self.memory_service.get_user_context(
                user_id=self.user_id,
                memory_types=self.include_memory_types,
            )

            return context, memories_by_type

        def _run_in_thread():
            """Run the async recall in a new event loop in this thread."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_async_recall())
            finally:
                loop.close()

        # Use a thread pool to run the async code (avoids nested event loop issues)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_thread)
            try:
                context, memories_by_type = future.result(timeout=10.0)
            except concurrent.futures.TimeoutError:
                logger.error("Memory recall timed out after 10 seconds")
                return "", {"count": 0, "latency_ms": 0, "types": []}
            except Exception as e:
                logger.error("Memory recall failed: %s", e)
                return "", {"count": 0, "latency_ms": 0, "types": []}

        latency_ms = (time.monotonic() - start_time) * 1000

        total_count = sum(len(mems) for mems in memories_by_type.values())
        types_with_memories = [t for t, mems in memories_by_type.items() if mems]

        metrics = {
            "count": total_count,
            "latency_ms": latency_ms,
            "types": types_with_memories,
        }

        return context, metrics


# ============================================================================
# Factory function
# ============================================================================


def create_memory_middleware(
    memory_service: "AgentMemoryService",
    user_id: str,
    auto_recall: bool = True,
    max_recall_results: int = 5,
    include_memory_types: Optional[List[str]] = None,
) -> MemoryMiddleware:
    """
    Factory function to create MemoryMiddleware instance.

    Args:
        memory_service: AgentMemoryService instance
        user_id: User ID for memory namespace
        auto_recall: Enable automatic memory recall in before_agent
        max_recall_results: Max memories to recall
        include_memory_types: Memory types to include

    Returns:
        Configured MemoryMiddleware instance
    """
    return MemoryMiddleware(
        memory_service=memory_service,
        user_id=user_id,
        auto_recall=auto_recall,
        max_recall_results=max_recall_results,
        include_memory_types=include_memory_types,
    )


__all__ = [
    "MemoryState",
    "MemoryMiddleware",
    "create_memory_middleware",
]

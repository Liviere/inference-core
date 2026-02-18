"""
Memory Middleware for LangChain v1 Agents (CoALA Architecture).

This middleware provides automatic memory context injection for agents using
the LangChain v1 middleware API. It uses wrap_model_call to inject CoALA-structured
memory context (semantic / episodic / procedural) into the system prompt
before each model call.

Key features:
- Auto-recalls relevant memories across CoALA categories
- Injects CoALA-structured XML context into system prompt via wrap_model_call
- Tracks memory operations for observability
- Uses ThreadPoolExecutor for async memory operations in sync hooks

Source: CoALA whitepaper – arxiv:2309.02427
"""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.messages import SystemMessage
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

from inference_core.celery.async_utils import run_async_safely

if TYPE_CHECKING:
    from inference_core.services.agent_memory_service import (
        AgentMemoryStoreService,
    )

from inference_core.services.agent_memory_service import MemoryCategory, MemoryType

logger = logging.getLogger(__name__)


class MemoryState(AgentState):
    """Extended agent state for memory middleware (CoALA-aware).

    Attributes:
        memory_context: Formatted CoALA-structured memory context string
        memories_recalled: Number of memories retrieved during recall
        memory_recall_latency_ms: Time taken for memory recall in milliseconds
        memory_types_recalled: List of memory types that were retrieved
        memory_categories_recalled: List of CoALA categories that had results
    """

    memory_context: NotRequired[str]
    memories_recalled: NotRequired[int]
    memory_recall_latency_ms: NotRequired[float]
    memory_types_recalled: NotRequired[List[str]]
    memory_categories_recalled: NotRequired[List[str]]


@dataclass
class _MemoryMiddlewareContext:
    """Internal context for tracking memory operations across hooks."""

    recall_start_time: Optional[float] = None
    recalled_memories: List[Any] = field(default_factory=list)
    context_injected: bool = False


class MemoryMiddleware(AgentMiddleware[MemoryState]):
    """Middleware for automatic CoALA memory context injection into system prompt.

    Recalls memories across configured CoALA categories (semantic, episodic,
    procedural) and injects structured XML context before each model call.

    Attributes:
        state_schema: The custom state schema with memory tracking fields
        memory_service: AgentMemoryStoreService instance for memory operations
        user_id: User ID for memory namespace isolation
        auto_recall: Whether to automatically recall memories in before_agent
        max_recall_results: Maximum number of memories to recall per category
        include_memory_types: Memory types to include in context recall
        include_categories: CoALA categories to recall (defaults to all three)
    """

    state_schema = MemoryState

    def __init__(
        self,
        memory_service: "AgentMemoryStoreService",
        user_id: str,
        auto_recall: bool = True,
        max_recall_results: int = 5,
        include_memory_types: Optional[List[str]] = None,
        include_categories: Optional[List[str]] = None,
    ):
        """Initialize the memory middleware.

        Args:
            memory_service: AgentMemoryStoreService instance for memory operations.
            user_id: User ID for memory namespace isolation.
            auto_recall: Whether to automatically recall memories in before_agent.
            max_recall_results: Maximum number of memories to recall per category.
            include_memory_types: Memory types to include in context.
                                 Defaults to preferences, facts, instructions, goals, context.
            include_categories: CoALA categories to recall.
                               Defaults to all three (semantic, episodic, procedural).
        """
        self.memory_service = memory_service
        self.user_id = user_id
        self.auto_recall = auto_recall
        self.max_recall_results = max_recall_results
        self.include_memory_types = include_memory_types or [
            MemoryType.PREFERENCES.value,
            MemoryType.FACTS.value,
            MemoryType.INSTRUCTION.value,
            MemoryType.GOALS.value,
            MemoryType.CONTEXT.value,
        ]
        self.include_categories = include_categories or [
            MemoryCategory.SEMANTIC.value,
            MemoryCategory.EPISODIC.value,
            MemoryCategory.PROCEDURAL.value,
        ]

        # Per-invocation context (persists across hooks within same invocation)
        self._ctx: Optional[_MemoryMiddlewareContext] = None
        # Cached memory context for injection into system prompt
        self._cached_memory_context: Optional[str] = None

    # -------------------------------------------------------------------------
    # Node-style hooks
    # -------------------------------------------------------------------------

    def before_agent(
        self, state: MemoryState, runtime: Runtime
    ) -> Dict[str, Any] | None:
        """Recall relevant memories and store in state at start of execution.

        This hook:
        1. Extracts the user's input from the messages
        2. Recalls relevant memories based on the input
        3. Caches memory context for wrap_model_call injection
        4. Returns state updates with memory tracking metrics

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
        # Reset cached context for new invocation
        self._cached_memory_context = None

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

            # Cache context for wrap_model_call to inject into system prompt
            self._cached_memory_context = memory_context
            self._ctx.context_injected = True

            logger.info(
                "Recalled memory context: %d memories for user=%s (%.1fms)",
                metrics.get("count", 0),
                self.user_id,
                metrics.get("latency_ms", 0),
            )

            return {
                "memory_context": memory_context,
                "memories_recalled": metrics.get("count", 0),
                "memory_recall_latency_ms": metrics.get("latency_ms", 0),
                "memory_types_recalled": metrics.get("types", []),
                "memory_categories_recalled": metrics.get("categories", []),
            }

        except Exception as e:
            logger.error("Failed to recall memories: %s", e, exc_info=True)
            return {
                "memories_recalled": 0,
                "memory_context": "",
            }

    # -------------------------------------------------------------------------
    # Wrap-style hooks
    # -------------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject memory context into system prompt before model call.

        This hook modifies the system message to include recalled memory context,
        ensuring the model has access to relevant user memories when generating
        responses.

        Args:
            request: ModelRequest containing messages, model, and system_message
            handler: Function to call the underlying model

        Returns:
            ModelResponse from the handler
        """
        # If no memory context was recalled, pass through unchanged
        if not self._cached_memory_context:
            return handler(request)

        try:
            # Get current system message content blocks
            current_blocks = list(request.system_message.content_blocks)

            # Create memory context block to prepend
            memory_block = {
                "type": "text",
                "text": f"\n<user_memory_context>\n{self._cached_memory_context}\n</user_memory_context>\n\n",
            }

            # Prepend memory context to system message
            new_content = current_blocks + [memory_block]
            new_system_message = SystemMessage(content=new_content)

            logger.debug(
                "Injected memory context into system prompt for user=%s",
                self.user_id,
            )

            return handler(request.override(system_message=new_system_message))

        except Exception as e:
            logger.error(
                "Failed to inject memory context into system prompt: %s",
                e,
                exc_info=True,
            )
            # Fall back to original request on error
            return handler(request)

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
        """Recall memories across CoALA categories and format as structured context.

        Uses run_async_safely() to reuse the Celery worker loop when available.

        Args:
            user_input: User's query for relevance-based recall

        Returns:
            Tuple of (formatted_context, metrics_dict)
        """
        start_time = time.monotonic()

        async def _async_recall():
            """Async function to perform CoALA multi-category recall."""
            context = await self.memory_service.format_context_for_prompt(
                user_id=self.user_id,
                query=user_input,
                include_types=self.include_memory_types,
                include_categories=self.include_categories,
            )

            memories_by_type = await self.memory_service.get_user_context(
                user_id=self.user_id,
                memory_types=self.include_memory_types,
            )

            return context, memories_by_type

        try:
            context, memories_by_type = run_async_safely(_async_recall(), timeout=10.0)
        except TimeoutError:
            logger.error("Memory recall timed out after 10 seconds")
            return "", {"count": 0, "latency_ms": 0, "types": [], "categories": []}
        except Exception as e:
            logger.error("Memory recall failed: %s", e)
            return "", {"count": 0, "latency_ms": 0, "types": [], "categories": []}

        latency_ms = (time.monotonic() - start_time) * 1000

        total_count = sum(len(mems) for mems in memories_by_type.values())
        types_with_memories = [t for t, mems in memories_by_type.items() if mems]

        # Determine which categories had results
        from inference_core.services.agent_memory_service import get_category_for_type

        categories_with_memories = list(
            {get_category_for_type(t).value for t in types_with_memories}
        )

        metrics = {
            "count": total_count,
            "latency_ms": latency_ms,
            "types": types_with_memories,
            "categories": categories_with_memories,
        }

        return context, metrics


# ============================================================================
# Factory function
# ============================================================================


def create_memory_middleware(
    memory_service: AgentMemoryStoreService,
    user_id: str,
    auto_recall: bool = True,
    max_recall_results: int = 5,
    include_memory_types: Optional[List[str]] = None,
    include_categories: Optional[List[str]] = None,
) -> MemoryMiddleware:
    """
    Factory function to create MemoryMiddleware instance (CoALA-aware).

    Args:
        memory_service: AgentMemoryStoreService instance
        user_id: User ID for memory namespace
        auto_recall: Enable automatic memory recall in before_agent
        max_recall_results: Max memories to recall per category
        include_memory_types: Memory types to include (uses canonical MemoryType values)
        include_categories: CoALA categories to recall (defaults to all three)

    Returns:
        Configured MemoryMiddleware instance
    """
    return MemoryMiddleware(
        memory_service=memory_service,
        user_id=user_id,
        auto_recall=auto_recall,
        max_recall_results=max_recall_results,
        include_memory_types=include_memory_types,
        include_categories=include_categories,
    )


__all__ = [
    "MemoryState",
    "MemoryMiddleware",
    "create_memory_middleware",
]

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

from pydantic import BaseModel, Field

from inference_core.agents.middleware._runtime_context import (
    get_user_id as _ctx_get_user_id,
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
        session_analysis_saved: Whether the post-run hook persisted a memory entry
        session_analysis_latency_ms: Time taken for post-run analysis in milliseconds
    """

    memory_context: NotRequired[str]
    memories_recalled: NotRequired[int]
    memory_recall_latency_ms: NotRequired[float]
    memory_types_recalled: NotRequired[List[str]]
    memory_categories_recalled: NotRequired[List[str]]
    session_analysis_saved: NotRequired[bool]
    session_analysis_latency_ms: NotRequired[float]


class _MemoryExtractionResult(BaseModel):
    """Structured output schema for the post-run memory extraction LLM call.

    WHY this schema: Forces a structured, type-safe response from the model so the
    middleware can act on it programmatically without text parsing.
    """

    worth_saving: bool = Field(
        description=(
            "True when the session contains information worth persisting in "
            "long-term memory for future conversations."
        )
    )
    content: str = Field(
        description=(
            "Compact memory summary (≤300 words) written in the same language as "
            "the user. Empty string when worth_saving is False."
        )
    )
    memory_type: str = Field(
        description=(
            "'session_summary' for preferences, facts, and contextual information; "
            "'interaction' for explicit corrections or user feedback on agent behaviour; "
            "empty string when worth_saving is False."
        )
    )


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
        user_id: Optional[str] = None,
        auto_recall: bool = True,
        max_recall_results: int = 5,
        include_memory_types: Optional[List[str]] = None,
        include_categories: Optional[List[str]] = None,
        postrun_analysis: bool = True,
        postrun_analysis_model: Optional[str] = None,
    ):
        """Initialize the memory middleware.

        Args:
            memory_service: AgentMemoryStoreService instance for memory operations.
            user_id: User ID for memory namespace isolation.  When ``None``
                (Agent Server), resolved at runtime from
                ``runtime.configurable["user_id"]`` via context vars.
            auto_recall: Whether to automatically recall memories in before_agent.
            max_recall_results: Maximum number of memories to recall per category.
            include_memory_types: Memory types to include in context.
                                 Defaults to preferences, facts, instructions, goals, context.
            include_categories: CoALA categories to recall.
                               Defaults to all three (semantic, episodic, procedural).
            postrun_analysis: Whether to run best-effort post-session analysis in
                after_agent, silently persisting session-level memories the main
                agent may have missed.  Controlled by
                ``agent_memory_postrun_analysis_enabled`` in settings.
            postrun_analysis_model: Optional model name for the post-run extraction
                LLM call.  When ``None`` (default), the middleware reuses whichever
                model the agent last used (captured from ``wrap_model_call``).  Set
                to a cheaper/faster model (e.g. ``"gpt-5-nano"``) when you want
                to keep extraction costs low without changing the agent's own model.
        """
        self.memory_service = memory_service
        self.user_id = user_id
        self.auto_recall = auto_recall
        self.max_recall_results = max_recall_results
        self.postrun_analysis = postrun_analysis
        self.postrun_analysis_model = postrun_analysis_model
        # Set in wrap_model_call / awrap_model_call; used by after_agent.
        self._captured_model: Optional[Any] = None
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
    # Deferred user_id resolution
    # -------------------------------------------------------------------------

    def _resolve_user_id(self) -> Optional[str]:
        """Return user_id — instance attribute or context-var fallback.

        WHY: On the Agent Server the middleware singleton is shared across
        concurrent requests so ``self.user_id`` is ``None``.  The actual
        value is set per-request via ``populate_from_configurable()`` which
        writes to ``_runtime_context._user_id``.
        """
        if self.user_id is not None:
            return self.user_id
        ctx_uid = _ctx_get_user_id()
        return str(ctx_uid) if ctx_uid is not None else None

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

        resolved_uid = self._resolve_user_id()
        if not resolved_uid:
            logger.warning(
                "Memory recall skipped — no user_id available (instance=%s, ctx=%s)",
                self.user_id,
                _ctx_get_user_id(),
            )
            return None

        # Extract user input from messages
        user_input = self._extract_user_input(state)
        if not user_input:
            logger.debug("No user input found in state, skipping memory recall")
            return None

        try:
            # Recall memories using asyncio.run() since hooks are synchronous
            memory_context, metrics = self._recall_and_format(user_input, resolved_uid)

            if not memory_context:
                logger.debug("No relevant memories found for user=%s", resolved_uid)
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
                resolved_uid,
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
        # Capture the current model for post-run analysis (before possible injection override).
        self._captured_model = getattr(request, "model", None)

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
                self._resolve_user_id(),
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

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse:
        """Async version of wrap_model_call for use with astream."""
        self._captured_model = getattr(request, "model", None)
        if not self._cached_memory_context:
            return await handler(request)

        try:
            current_blocks = list(request.system_message.content_blocks)
            memory_block = {
                "type": "text",
                "text": f"\n<user_memory_context>\n{self._cached_memory_context}\n</user_memory_context>\n\n",
            }
            new_content = current_blocks + [memory_block]
            new_system_message = SystemMessage(content=new_content)
            return await handler(request.override(system_message=new_system_message))
        except Exception as e:
            logger.error(
                "Failed to inject memory context into system prompt (async): %s",
                e,
                exc_info=True,
            )
            return await handler(request)

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

    def _recall_and_format(
        self,
        user_input: str,
        user_id: Optional[str] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Recall memories across CoALA categories and format as structured context.

        Uses run_async_safely() to reuse the Celery worker loop when available.

        Args:
            user_input: User's query for relevance-based recall
            user_id: Resolved user_id (falls back to self.user_id).

        Returns:
            Tuple of (formatted_context, metrics_dict)
        """
        uid = user_id or self.user_id
        start_time = time.monotonic()

        async def _async_recall():
            """Async function to perform CoALA multi-category recall."""
            context = await self.memory_service.format_context_for_prompt(
                user_id=uid,
                query=user_input,
                include_types=self.include_memory_types,
                include_categories=self.include_categories,
            )

            memories_by_type = await self.memory_service.get_user_context(
                user_id=uid,
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

        return self._build_recall_metrics(context, memories_by_type, start_time)

    async def _arecall_and_format(
        self,
        user_input: str,
        user_id: Optional[str] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Async recall path — used by Agent Server (avoids run_async_safely).

        WHY: On the Agent Server the before_agent hook runs inside an already-
        running event loop.  Using ``run_async_safely`` would spin up a new
        thread+loop, which can trigger MissingGreenlet errors with SQLAlchemy
        async connections.  This method directly awaits the async memory
        service instead.
        """
        uid = user_id or self.user_id
        start_time = time.monotonic()

        try:
            context = await self.memory_service.format_context_for_prompt(
                user_id=uid,
                query=user_input,
                include_types=self.include_memory_types,
                include_categories=self.include_categories,
            )

            memories_by_type = await self.memory_service.get_user_context(
                user_id=uid,
                memory_types=self.include_memory_types,
            )
        except TimeoutError:
            logger.error("Memory recall timed out (async)")
            return "", {"count": 0, "latency_ms": 0, "types": [], "categories": []}
        except Exception as e:
            logger.error("Memory recall failed (async): %s", e)
            return "", {"count": 0, "latency_ms": 0, "types": [], "categories": []}

        return self._build_recall_metrics(context, memories_by_type, start_time)

    def _build_recall_metrics(
        self,
        context: str,
        memories_by_type: Dict[str, Any],
        start_time: float,
    ) -> tuple[str, Dict[str, Any]]:
        """Shared metrics builder for sync and async recall paths."""
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

    # -------------------------------------------------------------------------
    # Post-run analysis helpers
    # -------------------------------------------------------------------------

    def _get_analysis_model(self) -> Optional[Any]:
        """Return the model instance to use for post-run extraction.

        Priority:
        1. ``postrun_analysis_model`` (override by name) → ``init_chat_model(name)``
        2. ``_captured_model`` (last model seen in ``wrap_model_call``) → agent's own model

        Returns ``None`` when neither is available; the caller should skip analysis
        and log a warning rather than failing.
        """
        if self.postrun_analysis_model:
            try:
                from langchain.chat_models import init_chat_model

                return init_chat_model(self.postrun_analysis_model)
            except Exception as e:
                logger.warning(
                    "Failed to init postrun_analysis_model '%s', "
                    "falling back to captured model: %s",
                    self.postrun_analysis_model,
                    e,
                )
                return self._captured_model
        return self._captured_model

    # Prompt template used for every post-run extraction call.
    # Defined at class scope to avoid reconstructing the string on every invocation.
    _EXTRACTION_PROMPT: str = (
        "Analyze the following conversation and decide whether it contains "
        "information worth persisting in long-term memory for future sessions.\n\n"
        "Information worth persisting includes:\n"
        "- User preferences (language, tone, tools, response style, formats)\n"
        "- Personal facts (name, location, profession, goals, current project)\n"
        "- Explicit future instructions ('always do X', 'never do Y', 'from now on…')\n"
        "- Significant corrections to agent behaviour\n"
        "- Project or task context useful beyond this session\n\n"
        "Do NOT persist: single-question factual exchanges, greetings, or generic "
        "conversation with no personal context.\n\n"
        "Write the content field in the SAME language the user spoke.\n\n"
        "Conversation:\n\n{transcript}"
    )

    async def _extract_via_model(
        self,
        state: MemoryState,
        model: Any,
    ) -> tuple[Optional[str], str]:
        """Ask the model to extract what's worth persisting from the completed session.

        WHY model-based extraction: A model call understands any natural language and catches nuanced
        signals — implicit corrections, preferences stated in context, multi-turn intent
        — that deterministic rules cannot.

        Returns:
            (content, memory_type) where content is None when nothing is worth saving.
        """
        messages = state.get("messages", [])
        lines: list[str] = []
        for msg in messages:
            msg_type = getattr(msg, "type", None) or getattr(msg, "role", None)
            content = getattr(msg, "content", "") or ""
            if not isinstance(content, str) or not content.strip():
                continue
            if msg_type in ("human", "user"):
                lines.append(f"User: {content.strip()}")
            elif msg_type in ("ai", "assistant"):
                lines.append(f"Assistant: {content.strip()}")

        if not lines:
            return None, ""

        # Limit transcript length to avoid blowing up context windows.
        transcript = "\n\n".join(lines)[-4000:]

        try:
            structured = model.with_structured_output(_MemoryExtractionResult)
            result: _MemoryExtractionResult = await structured.ainvoke(
                [
                    {
                        "role": "user",
                        "content": self._EXTRACTION_PROMPT.format(
                            transcript=transcript
                        ),
                    }
                ]
            )
            if not result.worth_saving or not result.content.strip():
                return None, ""
            memory_type = result.memory_type.strip() or "session_summary"
            return result.content.strip(), memory_type
        except Exception as e:
            logger.warning("Post-run model extraction failed: %s", e)
            return None, ""

    # -------------------------------------------------------------------------
    # Post-run analysis hooks
    # -------------------------------------------------------------------------

    def after_agent(
        self, state: MemoryState, runtime: Runtime
    ) -> Dict[str, Any] | None:
        """Best-effort post-session analysis: persist what the agent may have missed.

        WHY model-based extraction:
        By reusing the same model the agent used (or a configured cheaper override),
        the analysis understands any language and detects nuanced signals — implicit
        corrections, preferences stated in context

        Behavior:
          - Skips when postrun_analysis is disabled, no user_id is available, or no
            model was captured from wrap_model_call (agent made no model calls).
          - Calls _extract_via_model to get structured (content, memory_type).
          - When the agent already persisted memories explicitly, only a session_summary
            is added on top — interaction duplicates are skipped.
          - Saves via AgentMemoryStoreService (NOT tool calls, which would modify the
            final state the caller has already read).
          - Always fail-open: errors never propagate to the caller.
        """
        if not self.postrun_analysis:
            return None

        resolved_uid = self._resolve_user_id()
        if not resolved_uid:
            logger.debug("Post-run analysis skipped — no user_id available")
            return None

        messages = state.get("messages", [])
        if len(messages) < 2:
            logger.debug(
                "Post-run analysis skipped — too few messages (%d)", len(messages)
            )
            return None

        model = self._get_analysis_model()
        if model is None:
            logger.warning(
                "Post-run analysis skipped — no model available "
                "(wrap_model_call may not have fired this run)"
            )
            return None

        already_saved = self._has_memory_save_in_run(state)
        start_time = time.monotonic()

        try:

            async def _analyse_and_save() -> Optional[str]:
                content, memory_type = await self._extract_via_model(state, model)
                if not content:
                    return None
                if already_saved and memory_type != "session_summary":
                    logger.debug(
                        "Post-run analysis: agent already saved; skipping '%s'",
                        memory_type,
                    )
                    return None
                await self.memory_service.save_memory(
                    user_id=resolved_uid,
                    content=content,
                    memory_type=memory_type,
                    topic="session_analysis",
                )
                return memory_type

            saved_type = run_async_safely(_analyse_and_save(), timeout=15.0)
            if saved_type is None:
                return None

            latency_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "Post-run analysis: saved '%s' for user=%s (%.1fms)",
                saved_type,
                resolved_uid,
                latency_ms,
            )
            return {
                "session_analysis_saved": True,
                "session_analysis_latency_ms": latency_ms,
            }
        except Exception as e:
            logger.warning("Post-run analysis failed (fail-open): %s", e, exc_info=True)
            return None

    async def aafter_agent(
        self, state: MemoryState, runtime: Runtime
    ) -> Dict[str, Any] | None:
        """Async version of after_agent for Agent Server and astream contexts.

        WHY separate async path: the Agent Server runs after_agent inside an existing
        asyncio event loop.  run_async_safely would spin up a new thread+loop, causing
        MissingGreenlet errors with SQLAlchemy async connections.  This method directly
        awaits _extract_via_model and the memory service instead.
        """
        if not self.postrun_analysis:
            return None

        resolved_uid = self._resolve_user_id()
        if not resolved_uid:
            return None

        messages = state.get("messages", [])
        if len(messages) < 2:
            return None

        model = self._get_analysis_model()
        if model is None:
            logger.warning("Post-run analysis (async) skipped — no model available")
            return None

        already_saved = self._has_memory_save_in_run(state)
        start_time = time.monotonic()

        try:
            content, memory_type = await self._extract_via_model(state, model)
            if not content:
                return None
            if already_saved and memory_type != "session_summary":
                return None
            await self.memory_service.save_memory(
                user_id=resolved_uid,
                content=content,
                memory_type=memory_type,
                topic="session_analysis",
            )
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "Post-run analysis (async): saved '%s' for user=%s (%.1fms)",
                memory_type,
                resolved_uid,
                latency_ms,
            )
            return {
                "session_analysis_saved": True,
                "session_analysis_latency_ms": latency_ms,
            }
        except Exception as e:
            logger.warning(
                "Post-run analysis (async) failed (fail-open): %s", e, exc_info=True
            )
            return None

    # -------------------------------------------------------------------------
    # Helper: detect explicit saves during the run
    # -------------------------------------------------------------------------

    def _has_memory_save_in_run(self, state: MemoryState) -> bool:
        """Return True if the agent explicitly called save_memory_store during this run.

        WHY: When the model already persisted something, the post-run hook acts more
        conservatively — only adding a session_summary for substantive conversations
        rather than duplicate interaction entries.
        """
        messages = state.get("messages", [])
        for msg in messages:
            # Tool-result messages carry the tool name
            if getattr(msg, "name", None) == "save_memory_store":
                return True
            # AI messages carry pending tool_calls before results arrive
            for tc in getattr(msg, "tool_calls", None) or []:
                if isinstance(tc, dict):
                    if tc.get("name") == "save_memory_store":
                        return True
                elif getattr(tc, "name", None) == "save_memory_store":
                    return True
        return False


# ============================================================================
# Factory function
# ============================================================================


def create_memory_middleware(
    memory_service: AgentMemoryStoreService,
    user_id: Optional[str] = None,
    auto_recall: bool = True,
    max_recall_results: int = 5,
    include_memory_types: Optional[List[str]] = None,
    include_categories: Optional[List[str]] = None,
    postrun_analysis: bool = True,
    postrun_analysis_model: Optional[str] = None,
) -> MemoryMiddleware:
    """
    Factory function to create MemoryMiddleware instance (CoALA-aware).

    Args:
        memory_service: AgentMemoryStoreService instance
        user_id: User ID for memory namespace.  ``None`` for Agent Server
            (resolved at runtime from configurable).
        auto_recall: Enable automatic memory recall in before_agent
        max_recall_results: Max memories to recall per category
        include_memory_types: Memory types to include (uses canonical MemoryType values)
        include_categories: CoALA categories to recall (defaults to all three)
        postrun_analysis: Enable best-effort post-session analysis in after_agent
        postrun_analysis_model: Optional model name override for the extraction LLM call.
            When None, reuses the agent's own model (captured from wrap_model_call).

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
        postrun_analysis=postrun_analysis,
        postrun_analysis_model=postrun_analysis_model,
    )


__all__ = [
    "MemoryState",
    "MemoryMiddleware",
    "create_memory_middleware",
]

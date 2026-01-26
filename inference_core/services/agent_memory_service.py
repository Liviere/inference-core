"""
Agent Memory Service (Store-based)

Implements long-term memory operations using LangGraph Store backends
instead of vector store providers. Mirrors the public surface of the
vector-based service to simplify migration toward LangGraph-native
persistence (SqliteStore/PostgresStore/InMemoryStore).
"""

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from inference_core.services.agents_service import AgentService

logger = logging.getLogger(__name__)


MEMORIES_STORE_NAME = os.getenv("MEMORIES_STORE_NAME", "memories")
INDEXED_FIELDS = ["content", "topic"]


class MemoryType(str, Enum):
    """Standard memory type categories."""

    PREFERENCES = "preferences"
    FACTS = "facts"
    CONTEXT = "context"
    INSTRUCTION = "instructions"
    GOALS = "goals"
    GENERAL = "general"


MEMORY_TYPES = [mt.value for mt in MemoryType]


def get_memory_type_literal():
    """Generate Literal type hint from MemoryType enum for runtime consistency.

    Use this in tool signatures instead of hardcoding Literal values.
    Example: memory_type: get_memory_type_literal() = MemoryType.GENERAL.value
    """
    from typing import Literal, get_args

    return Literal[tuple(mt.value for mt in MemoryType)]


def validate_memory_type(value: str) -> str:
    """Runtime validator ensuring memory_type matches canonical enum.

    Raises ValueError if invalid type provided.
    Use in Pydantic validators or explicit checks.
    """
    # Unwrap Enum if passed instead of str
    if not isinstance(value, str):
        if hasattr(value, "value"):
            value = value.value

    if value not in MEMORY_TYPES:
        valid_types = ", ".join(MEMORY_TYPES)
        raise ValueError(
            f"Invalid memory_type '{value}'. Must be one of: {valid_types}"
        )
    return value


def format_memory_types_for_description() -> str:
    """Generate formatted list of memory types with descriptions for tool docs.

    Returns multi-line string suitable for embedding in tool description.
    """
    lines = ["Available memory types:"]
    for mtype in MemoryType:
        desc_enum = MemoryTypeDescription[mtype.name]
        # Extract first sentence only for brevity
        first_sentence = desc_enum.value.split(".")[0] + "."
        lines.append(f"  - {mtype.value}: {first_sentence}")
    return "\n".join(lines)


class MemoryTypeDescription(str, Enum):
    """Descriptions for memory type categories."""

    PREFERENCES = """Personal preferences, communication styles, and behavioral patterns that should be remembered for future interactions. These memories help maintain consistency in responses and adapt to individual needs and expectations.
    Examples: 'Prefers concise answers with bullet points', 'Enjoys humor in casual conversations'"""

    FACTS = """Factual information about entities, relationships, locations, or any objective data that remains stable over time. These memories store verifiable information that can be referenced to provide accurate and personalized responses.
    Examples: 'Lives in Warsaw, Poland', 'Company headquarters located in New York'"""

    CONTEXT = """Situational awareness and environmental information that provides background for current or recent activities. These memories capture the immediate circumstances and ongoing situations that may influence decision-making.
    Examples: 'Meeting scheduled for tomorrow at 2 PM', 'Weather forecast shows rain this week'"""

    INSTRUCTION = """Operational guidelines, rules, or specific directions about how tasks should be performed or processes should be followed. These memories ensure consistent execution of established procedures and methodologies.
    Examples: 'Always backup files before making changes', 'Send weekly reports every Friday morning'"""

    GOALS = """Objectives, aspirations, and desired outcomes that represent what someone or something is working towards. These memories help maintain focus on long-term vision and provide context for prioritizing actions and decisions.
    Examples: 'Improve team collaboration and communication', 'Reduce system response time by 30%'"""

    GENERAL = """Miscellaneous observations, notes, and information that may be useful but doesn't fit into other specific categories. These memories serve as a repository for various insights and details that could be relevant in future contexts.
    Examples: 'Noticed increased activity during evening hours', 'Client mentioned budget concerns during last meeting'"""


@dataclass
class MemoryMetadata:
    """Consistent metadata schema for stored memories."""

    user_id: str
    memory_type: str = MemoryType.GENERAL.value
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    session_id: Optional[str] = None
    topic: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Flatten metadata for storage and filtering."""

        result = {
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "created_at": self.created_at,
        }
        if self.session_id:
            result["session_id"] = self.session_id
        if self.topic:
            result["topic"] = self.topic
        if self.extra:
            result.update(self.extra)
        return result


@dataclass
class MemoryStoreDocument:
    """Lightweight record wrapper for store search results.

    Includes temporal fields (created_at, updated_at) extracted from
    LangGraph Store Item metadata for time-aware memory operations.
    """

    id: str
    value: Dict[str, Any]
    metadata: Dict[str, Any]
    score: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def content(self) -> str:
        return self.value.get("content", "")

    @property
    def topic(self) -> Optional[str]:
        return self.value.get("topic", None)

    @property
    def memory_type(self) -> Optional[str]:
        return self.value.get("memory_type", None)

    @property
    def created_at_iso(self) -> Optional[str]:
        """Return created_at as ISO 8601 string or None."""
        if self.created_at:
            return self.created_at.isoformat()
        return None

    @property
    def updated_at_iso(self) -> Optional[str]:
        """Return updated_at as ISO 8601 string or None."""
        if self.updated_at:
            return self.updated_at.isoformat()
        return None


@dataclass
class MemoryData:
    """Structured memory data for storage."""

    content: str
    memory_type: str
    session_id: Optional[str] = None
    topic: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class AgentMemoryStoreService:
    """Store-backed memory service to align with LangGraph persistence."""

    def __init__(
        self,
        store: Any,
        base_namespace: Sequence[str] = (MEMORIES_STORE_NAME,),
        max_results: int = 5,
    ) -> None:
        self.store = store
        self.base_namespace = tuple(base_namespace)
        self.max_results = max_results
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _namespace(self, user_id: str) -> tuple:
        """Build a namespaced tuple for store isolation per user."""

        return (str(user_id), *self.base_namespace)

    async def save_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = MemoryType.GENERAL.value,
        session_id: Optional[str] = None,
        topic: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """Persist a memory item in the configured store."""

        assert (
            memory_type in MemoryType._value2member_map_
        ), f"Invalid memory type: {memory_type}"

        user_id = str(user_id)
        namespace_for_memory = (user_id, MEMORIES_STORE_NAME)

        if not extra_metadata:
            extra_metadata = {}

        if not memory_id:
            memory_id = str(uuid.uuid4())

        memory_data = MemoryData(
            content=content,
            memory_type=memory_type,
            session_id=session_id,
            topic=topic,
            extra_metadata=extra_metadata,
        )

        self.store.put(
            namespace_for_memory,
            memory_id,
            memory_data.__dict__,
            index=INDEXED_FIELDS,
        )

        return memory_id

    async def recall_memories(
        self,
        user_id: str,
        query: str,
        k: Optional[int] = None,
        memory_type: Optional[str] = None,
        include_scores: bool = True,
        sort_by_time: bool = False,
    ) -> List[MemoryStoreDocument]:
        """Retrieve relevant memories from the store for the user.

        Args:
            user_id: User identifier for namespace isolation.
            query: Semantic search query string.
            k: Maximum number of results to return.
            memory_type: Optional filter by memory type.
            include_scores: Whether to include relevance scores.
            sort_by_time: If True, sort results by created_at descending (newest first).
                          Otherwise, results are ordered by semantic relevance.

        Returns:
            List of MemoryStoreDocument with temporal metadata (created_at, updated_at).
        """
        limit = k or self.max_results

        namespace_for_memory = (user_id, MEMORIES_STORE_NAME)
        filters = {}
        if memory_type:
            filters["memory_type"] = memory_type

        results = self.store.search(
            namespace_for_memory,
            query=query,
            limit=limit,
            filter=filters,
        )

        documents: List[MemoryStoreDocument] = []
        for item in results or []:
            # Extract timestamps from Store Item (automatically managed by LangGraph Store)
            item_created_at = getattr(item, "created_at", None)
            item_updated_at = getattr(item, "updated_at", None)

            documents.append(
                MemoryStoreDocument(
                    id=getattr(item, "key", getattr(item, "id", "")),
                    value=getattr(item, "value", {}),
                    metadata=getattr(item, "value", {}).get("extra_metadata", {}),
                    score=getattr(item, "score", None) if include_scores else None,
                    created_at=item_created_at,
                    updated_at=item_updated_at,
                )
            )

        # Sort by time if requested (newest first); fallback None timestamps to end
        if sort_by_time:
            documents.sort(
                key=lambda d: (
                    d.created_at is not None,
                    d.created_at or datetime.min.replace(tzinfo=timezone.utc),
                ),
                reverse=True,
            )

        self.logger.debug(
            "Recalled %d memories for user=%s, query='%s', sort_by_time=%s",
            len(documents),
            user_id,
            query[:50] + "..." if len(query) > 50 else query,
            sort_by_time,
        )
        return documents

    async def get_user_context(
        self,
        user_id: str,
        memory_types: Optional[List[str]] = None,
        max_per_type: int = 3,
    ) -> Dict[str, List[str]]:
        """Aggregate memories per type for prompt construction (legacy, content-only).

        For time-aware context, use get_user_context_with_timestamps() instead.
        """
        if memory_types is None:
            memory_types = [
                MemoryType.PREFERENCES.value,
                MemoryType.FACTS.value,
                MemoryType.INSTRUCTION.value,
            ]

        context: Dict[str, List[str]] = {}
        for mtype in memory_types:
            memories = await self.recall_memories(
                user_id=user_id,
                query=mtype,
                k=max_per_type,
                memory_type=mtype,
            )
            if memories:
                context[mtype] = [m.content for m in memories]

        return context

    async def get_user_context_with_timestamps(
        self,
        user_id: str,
        memory_types: Optional[List[str]] = None,
        max_per_type: int = 10,
    ) -> Dict[str, List[MemoryStoreDocument]]:
        """Aggregate memories per type with full temporal metadata.

        Args:
            user_id: User identifier.
            memory_types: List of memory types to include.
            max_per_type: Maximum memories per type.

        Returns:
            Dict mapping memory_type to list of MemoryStoreDocument objects
            (includes created_at, updated_at for temporal processing).
        """
        if memory_types is None:
            memory_types = [
                MemoryType.PREFERENCES.value,
                MemoryType.FACTS.value,
                MemoryType.INSTRUCTION.value,
            ]

        context: Dict[str, List[MemoryStoreDocument]] = {}
        for mtype in memory_types:
            memories = await self.recall_memories(
                user_id=user_id,
                query=mtype,
                k=max_per_type,
                memory_type=mtype,
                sort_by_time=True,
            )
            if memories:
                context[mtype] = memories

        return context

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Remove a specific memory key from the store."""

        namespace = self._namespace(user_id)
        if not hasattr(self.store, "delete"):
            self.logger.warning("Store does not support delete; skipping")
            return False

        try:
            self.store.delete(namespace, memory_id)
            self.logger.info("Deleted memory id=%s for user=%s", memory_id, user_id)
            return True
        except Exception as exc:  # pragma: no cover - log and continue
            self.logger.error("Failed to delete memory id=%s: %s", memory_id, exc)
            return False

    async def delete_user_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
    ) -> int:
        """Bulk-delete user memories by type or all."""

        namespace = self._namespace(user_id)
        filters: Dict[str, Any] = {}
        if memory_type:
            filters["memory_type"] = memory_type

        try:
            results = self.store.search(namespace, query="", filter=filters)
        except TypeError:
            results = self.store.search(namespace, query=None, filter=filters)

        deleted = 0
        if hasattr(self.store, "delete"):
            for item in results or []:
                key = getattr(item, "key", getattr(item, "id", None))
                if key is None:
                    continue
                try:
                    self.store.delete(namespace, key)
                    deleted += 1
                except Exception as exc:  # pragma: no cover
                    self.logger.error("Failed to delete memory %s: %s", key, exc)

        self.logger.info(
            "Deleted %d memories for user=%s, type=%s",
            deleted,
            user_id,
            memory_type or "all",
        )
        return deleted

    async def format_context_for_prompt(
        self,
        user_id: str,
        query: Optional[str] = None,
        include_types: Optional[List[str]] = None,
        max_memories: int = 30,
    ) -> str:
        """Render stored memories as textual context for prompts with temporal bucketing.

        Memories are organized into time-based buckets (today, yesterday, last 5 days,
        last 90 days, older) to give the agent temporal context about when information
        was remembered. Newer memories are prioritized as more relevant/accurate.

        Args:
            user_id: User identifier.
            query: Optional semantic search query for relevance-based recall.
            include_types: Memory types to include in context.
            max_memories: Maximum total memories to retrieve for bucketing.

        Returns:
            Formatted string with memories organized by temporal buckets.
        """
        include_types = include_types or [
            MemoryType.PREFERENCES.value,
            MemoryType.FACTS.value,
            MemoryType.INSTRUCTION.value,
        ]

        # Collect all memories with timestamps
        all_memories: List[MemoryStoreDocument] = []

        if query:
            # Use semantic search with provided query
            memories = await self.recall_memories(
                user_id=user_id,
                query=query,
                k=max_memories,
                sort_by_time=False,  # Keep relevance order, will bucket later
            )
            # Filter by type if specified
            all_memories = [
                m
                for m in memories
                if m.memory_type in include_types or m.memory_type is None
            ]
        else:
            # Fetch memories by type
            context = await self.get_user_context_with_timestamps(
                user_id=user_id,
                memory_types=include_types,
                max_per_type=(
                    max_memories // len(include_types) if include_types else 10
                ),
            )
            for mtype_memories in context.values():
                all_memories.extend(mtype_memories)

        if not all_memories:
            return ""

        # Bucket memories by time
        buckets = self._bucket_memories_by_time(all_memories)

        # Build formatted output
        return self._format_bucketed_memories(buckets, query)

    def _bucket_memories_by_time(
        self,
        memories: List[MemoryStoreDocument],
    ) -> Dict[str, List[MemoryStoreDocument]]:
        """Group memories into temporal buckets.

        Buckets:
        - today: Memories from today
        - yesterday: Memories from yesterday
        - prev5_before_yesterday: Memories from D-2 to D-6 (5 days before yesterday)
        - prev90_excluding_above: Memories from D-7 to D-96
        - older: Memories older than 96 days

        Args:
            memories: List of MemoryStoreDocument with created_at timestamps.

        Returns:
            Dict mapping bucket name to list of memories, each bucket sorted newest-first.
        """
        now = datetime.now(timezone.utc)
        today = now.date()
        yesterday = today - timedelta(days=1)
        # D-2 to D-6 (5 days counting back from the day before yesterday)
        prev5_start = today - timedelta(days=6)  # D-6
        prev5_end = today - timedelta(days=2)  # D-2
        # D-7 to D-96 (90 days excluding the above)
        prev90_start = today - timedelta(days=96)  # D-96
        prev90_end = today - timedelta(days=7)  # D-7

        buckets: Dict[str, List[MemoryStoreDocument]] = {
            "today": [],
            "yesterday": [],
            "prev5_before_yesterday": [],
            "prev90_excluding_above": [],
            "older": [],
        }

        # De-duplicate by content
        seen_contents: set[str] = set()

        for mem in memories:
            content = mem.content
            if not content or content in seen_contents:
                continue
            seen_contents.add(content)

            # Parse timestamp
            mem_date = self._get_memory_date(mem)

            # Assign to bucket
            if mem_date == today:
                buckets["today"].append(mem)
            elif mem_date == yesterday:
                buckets["yesterday"].append(mem)
            elif mem_date and prev5_start <= mem_date <= prev5_end:
                buckets["prev5_before_yesterday"].append(mem)
            elif mem_date and prev90_start <= mem_date <= prev90_end:
                buckets["prev90_excluding_above"].append(mem)
            else:
                buckets["older"].append(mem)

        # Sort each bucket by timestamp descending (newest first)
        for bucket_name in buckets:
            buckets[bucket_name].sort(
                key=lambda m: (
                    m.created_at is not None,
                    m.created_at or datetime.min.replace(tzinfo=timezone.utc),
                ),
                reverse=True,
            )

        return buckets

    def _get_memory_date(self, mem: MemoryStoreDocument) -> Optional[datetime.date]:
        """Extract date from memory's created_at timestamp.

        Args:
            mem: MemoryStoreDocument instance.

        Returns:
            Date object or None if timestamp unavailable.
        """
        if mem.created_at:
            dt = mem.created_at
            # Ensure timezone-aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.date()
        return None

    def _format_bucketed_memories(
        self,
        buckets: Dict[str, List[MemoryStoreDocument]],
        query: Optional[str] = None,
    ) -> str:
        """Format bucketed memories as text for prompt injection.

        Args:
            buckets: Dict of bucket_name -> list of MemoryStoreDocument.
            query: Optional original query for context header.

        Returns:
            Formatted string with temporal sections, prefixed with current date.
        """
        # Current date/time for temporal context awareness
        now = datetime.now(timezone.utc)
        current_datetime_str = now.isoformat().replace("+00:00", "Z")

        header = (
            f"Current datetime (UTC): {current_datetime_str}\n\n"
            "Relevant information from previous conversations. Each memory includes created_at (UTC) "
            "indicating when the information was remembered/saved. Pay attention to the temporal context: "
            "some information remains valid regardless of age (e.g., user's name, stable preferences, permanent facts), "
            "while other information may be time-sensitive (e.g., current projects, temporary preferences, scheduled events). "
            "When memories conflict, consider both recency AND the nature of the information to determine which is most relevant. "
            "Use the timestamps to understand the timeline of user's interactions and evolving context:"
        )

        sections: List[str] = [header]

        def render_bucket(title: str, items: List[MemoryStoreDocument]) -> None:
            """Render a single bucket section."""
            if not items:
                return
            sections.append(f"\n{title}:")
            for item in items:
                ts_label = ""
                if item.created_at:
                    ts_label = f"[{item.created_at.isoformat()}] "
                type_label = f" (type: {item.memory_type})" if item.memory_type else ""
                topic_label = f" [topic: {item.topic}]" if item.topic else ""
                sections.append(f"- {ts_label}{item.content}{type_label}{topic_label}")

        render_bucket("Data from today", buckets.get("today", []))
        render_bucket("Data from yesterday", buckets.get("yesterday", []))
        render_bucket(
            "Data from the previous 5 days (counting back from yesterday)",
            buckets.get("prev5_before_yesterday", []),
        )
        render_bucket(
            "Data from the previous 90 days (excluding the above)",
            buckets.get("prev90_excluding_above", []),
        )
        render_bucket("Older entries", buckets.get("older", []))

        return "\n".join(sections) if len(sections) > 1 else ""


__all__ = [
    "AgentMemoryStoreService",
    "MemoryMetadata",
    "MemoryStoreDocument",
    "MemoryType",
    "MemoryTypeDescription",
    "validate_memory_type",
    "format_memory_types_for_description",
    "get_memory_type_literal",
]

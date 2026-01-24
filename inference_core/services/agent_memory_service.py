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
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

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
    """Lightweight record wrapper for store search results."""

    id: str
    value: Dict[str, Any]
    metadata: Dict[str, Any]
    score: Optional[float] = None

    @property
    def content(self) -> str:
        return self.value.get("content", "")

    @property
    def topic(self) -> Optional[str]:
        return self.value.get("topic", None)

    @property
    def memory_type(self) -> Optional[str]:
        return self.value.get("memory_type", None)


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
    ) -> List[MemoryStoreDocument]:
        """Retrieve relevant memories from the store for the user."""

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
            documents.append(
                MemoryStoreDocument(
                    id=getattr(item, "key", getattr(item, "id", "")),
                    value=getattr(item, "value", {}),
                    metadata=getattr(item, "value", {}).get("extra_metadata", {}),
                    score=getattr(item, "score", None) if include_scores else None,
                )
            )

        self.logger.debug(
            "Recalled %d memories for user=%s, query='%s'",
            len(documents),
            user_id,
            query[:50] + "..." if len(query) > 50 else query,
        )
        return documents

    async def get_user_context(
        self,
        user_id: str,
        memory_types: Optional[List[str]] = None,
        max_per_type: int = 3,
    ) -> Dict[str, List[str]]:
        """Aggregate memories per type for prompt construction."""

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
    ) -> str:
        """Render stored memories as textual context for prompts."""

        lines: List[str] = []
        include_types = include_types or [
            MemoryType.PREFERENCES.value,
            MemoryType.FACTS.value,
            MemoryType.INSTRUCTION.value,
        ]

        context = await self.get_user_context(user_id, include_types)

        if MemoryType.PREFERENCES.value in context:
            prefs = context[MemoryType.PREFERENCES.value]
            if prefs:
                lines.append("User preferences:")
                lines.extend([f"  - {pref}" for pref in prefs])

        if MemoryType.FACTS.value in context:
            facts = context[MemoryType.FACTS.value]
            if facts:
                lines.append("Known facts about user:")
                lines.extend([f"  - {fact}" for fact in facts])

        if MemoryType.INSTRUCTION.value in context:
            instructions = context[MemoryType.INSTRUCTION.value]
            if instructions:
                lines.append("User-specific instructions:")
                lines.extend([f"  - {instr}" for instr in instructions])

        if query:
            relevant = await self.recall_memories(user_id=user_id, query=query, k=3)
            if relevant:
                lines.append("\nRelevant context from memory:")
                lines.extend([f"  - {mem.content}" for mem in relevant])

        return "\n".join(lines) if lines else ""


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

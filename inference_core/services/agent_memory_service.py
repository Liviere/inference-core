"""
Agent Memory Service

Provides long-term memory capabilities for LangChain v1 agents.
Stores and retrieves user-specific memories using vector similarity search.

Features:
- Save memories with user_id namespace and metadata
- Recall relevant memories via semantic search
- Get user context (preferences, facts) for system prompts
- Upsert by similarity to avoid duplicates (optional)
- Delete memories by ID or user

Usage:
    from inference_core.services.agent_memory_service import get_agent_memory_service

    memory_service = get_agent_memory_service()
    if memory_service:
        # Save a memory
        memory_id = await memory_service.save_memory(
            user_id="user-uuid",
            content="User prefers dark mode UI",
            memory_type="preference",
        )

        # Recall memories
        memories = await memory_service.recall_memories(
            user_id="user-uuid",
            query="UI preferences",
        )
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional

from inference_core.core.config import get_settings
from inference_core.vectorstores.base import (
    BaseVectorStoreProvider,
    VectorStoreDocument,
)
from inference_core.vectorstores.factory import get_vector_store_provider

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Standard memory type categories."""

    PREFERENCE = "preference"  # User preferences (UI, language, etc.)
    FACT = "fact"  # Facts about user (name, location, etc.)
    CONTEXT = "context"  # Conversation context to remember
    INSTRUCTION = "instruction"  # User-specific instructions
    GENERAL = "general"  # General memories


@dataclass
class MemoryMetadata:
    """
    Core metadata schema for memories.
    Ensures consistent structure across all memory operations.
    """

    user_id: str
    memory_type: str = MemoryType.GENERAL.value
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    session_id: Optional[str] = None
    topic: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for vector store metadata."""
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


class AgentMemoryService:
    """
    Service for managing agent long-term memories.

    Uses vector store backend for semantic similarity search.
    Memories are namespaced by user_id for isolation.
    """

    def __init__(
        self,
        provider: BaseVectorStoreProvider,
        collection: str,
        max_results: int = 5,
        upsert_by_similarity: bool = False,
        similarity_threshold: float = 0.85,
    ):
        """
        Initialize AgentMemoryService.

        Args:
            provider: Vector store provider instance
            collection: Collection name for memories
            max_results: Default max results for recall queries
            upsert_by_similarity: Check similarity before adding to avoid duplicates
            similarity_threshold: Threshold for considering memories as duplicates (0.0-1.0)
        """
        self.provider = provider
        self.collection = collection
        self.max_results = max_results
        self.upsert_by_similarity = upsert_by_similarity
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def save_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = MemoryType.GENERAL.value,
        session_id: Optional[str] = None,
        topic: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        upsert_by_similarity: Optional[bool] = None,
    ) -> str:
        """
        Save a memory for a user.

        Args:
            user_id: User identifier (namespace)
            content: Memory content text
            memory_type: Type of memory (preference, fact, context, etc.)
            session_id: Optional session identifier
            topic: Optional topic/category
            extra_metadata: Additional metadata fields
            upsert_by_similarity: Override global setting for this call

        Returns:
            Memory ID (existing ID if duplicate found during upsert)
        """
        # Determine upsert behavior
        should_upsert = (
            upsert_by_similarity
            if upsert_by_similarity is not None
            else self.upsert_by_similarity
        )

        # Check for similar existing memory if upsert mode
        if should_upsert:
            existing = await self._find_similar_memory(user_id, content)
            if existing:
                self.logger.debug(
                    "Found similar memory (score=%.3f), skipping duplicate: %s",
                    existing.score,
                    existing.id,
                )
                return existing.id

        # Build metadata
        metadata = MemoryMetadata(
            user_id=user_id,
            memory_type=memory_type,
            session_id=session_id,
            topic=topic,
            extra=extra_metadata or {},
        )

        # Generate ID
        memory_id = str(uuid.uuid4())

        # Add to vector store
        ids = await self.provider.add_texts(
            texts=[content],
            metadatas=[metadata.to_dict()],
            ids=[memory_id],
            collection=self.collection,
        )

        self.logger.info(
            "Saved memory for user=%s, type=%s, id=%s",
            user_id,
            memory_type,
            ids[0] if ids else memory_id,
        )

        return ids[0] if ids else memory_id

    async def recall_memories(
        self,
        user_id: str,
        query: str,
        k: Optional[int] = None,
        memory_type: Optional[str] = None,
        topic: Optional[str] = None,
        include_scores: bool = True,
    ) -> List[VectorStoreDocument]:
        """
        Recall relevant memories for a user based on semantic query.

        Args:
            user_id: User identifier (namespace filter)
            query: Search query text
            k: Max number of results (uses default if None)
            memory_type: Filter by memory type
            topic: Filter by topic
            include_scores: Include similarity scores in results

        Returns:
            List of matching memory documents
        """
        k = k or self.max_results

        # Build filters
        filters: Dict[str, Any] = {"user_id": user_id}
        if memory_type:
            filters["memory_type"] = memory_type
        if topic:
            filters["topic"] = topic

        memories = await self.provider.similarity_search(
            query=query,
            k=k,
            collection=self.collection,
            filters=filters,
        )

        self.logger.debug(
            "Recalled %d memories for user=%s, query='%s'",
            len(memories),
            user_id,
            query[:50] + "..." if len(query) > 50 else query,
        )

        return memories

    async def get_user_context(
        self,
        user_id: str,
        memory_types: Optional[List[str]] = None,
        max_per_type: int = 3,
    ) -> Dict[str, List[str]]:
        """
        Get user context organized by memory type.
        Useful for building system prompts with user preferences/facts.

        Args:
            user_id: User identifier
            memory_types: Types to fetch (defaults to preference, fact, instruction)
            max_per_type: Max memories per type

        Returns:
            Dictionary mapping memory_type to list of content strings
        """
        if memory_types is None:
            memory_types = [
                MemoryType.PREFERENCE.value,
                MemoryType.FACT.value,
                MemoryType.INSTRUCTION.value,
            ]

        context: Dict[str, List[str]] = {}

        for memory_type in memory_types:
            # Use a generic query to get recent memories of this type
            memories = await self.provider.similarity_search(
                query=f"user {memory_type}",  # Generic query
                k=max_per_type,
                collection=self.collection,
                filters={
                    "user_id": user_id,
                    "memory_type": memory_type,
                },
            )
            if memories:
                context[memory_type] = [m.content for m in memories]

        return context

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: Memory document ID

        Returns:
            True if deleted successfully
        """
        try:
            # Check if provider has delete method
            if hasattr(self.provider, "delete_documents"):
                await self.provider.delete_documents(
                    ids=[memory_id],
                    collection=self.collection,
                )
                self.logger.info("Deleted memory id=%s", memory_id)
                return True
            else:
                self.logger.warning(
                    "Provider does not support delete_documents, memory not deleted"
                )
                return False
        except Exception as e:
            self.logger.error("Failed to delete memory id=%s: %s", memory_id, e)
            return False

    async def delete_user_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
    ) -> int:
        """
        Delete all memories for a user (optionally filtered by type).

        Args:
            user_id: User identifier
            memory_type: Optional type filter

        Returns:
            Number of memories deleted
        """
        try:
            # Check if provider supports filtered deletion
            if hasattr(self.provider, "delete_by_filter"):
                filters = {"user_id": user_id}
                if memory_type:
                    filters["memory_type"] = memory_type
                count = await self.provider.delete_by_filter(
                    filters=filters,
                    collection=self.collection,
                )
                self.logger.info(
                    "Deleted %d memories for user=%s, type=%s",
                    count,
                    user_id,
                    memory_type or "all",
                )
                return count

            # Fallback: list and delete individually
            if hasattr(self.provider, "list_documents"):
                filters = {"user_id": user_id}
                if memory_type:
                    filters["memory_type"] = memory_type
                docs = await self.provider.list_documents(
                    collection=self.collection,
                    filters=filters,
                    limit=1000,
                )
                if docs and hasattr(self.provider, "delete_documents"):
                    ids = [d.id for d in docs]
                    await self.provider.delete_documents(
                        ids=ids,
                        collection=self.collection,
                    )
                    self.logger.info(
                        "Deleted %d memories for user=%s", len(ids), user_id
                    )
                    return len(ids)

            self.logger.warning("Provider does not support bulk deletion")
            return 0

        except Exception as e:
            self.logger.error("Failed to delete user memories: %s", e)
            return 0

    async def _find_similar_memory(
        self,
        user_id: str,
        content: str,
    ) -> Optional[VectorStoreDocument]:
        """
        Find existing memory similar to content (for upsert deduplication).

        Args:
            user_id: User identifier
            content: Content to check for similarity

        Returns:
            Existing similar memory if found above threshold, None otherwise
        """
        results = await self.provider.similarity_search(
            query=content,
            k=1,
            collection=self.collection,
            filters={"user_id": user_id},
        )

        if results and results[0].score is not None:
            # Score interpretation depends on distance metric
            # For cosine similarity: higher is more similar (0-1)
            # For cosine distance: lower is more similar (0-2)
            score = results[0].score

            # Qdrant returns cosine distance as score, convert to similarity
            # cosine_similarity = 1 - cosine_distance
            # But some providers return similarity directly
            # We assume score > threshold means similar enough
            # For cosine distance: score < (1 - threshold) means similar
            # Let's use a heuristic: if score < 0.5, treat as distance

            if score < 0.5:
                # Likely cosine distance, convert to similarity
                similarity = 1 - score
            else:
                # Likely similarity score
                similarity = score

            if similarity >= self.similarity_threshold:
                return results[0]

        return None

    async def format_context_for_prompt(
        self,
        user_id: str,
        query: Optional[str] = None,
        include_types: Optional[List[str]] = None,
    ) -> str:
        """
        Format user memories as context string for system prompt injection.

        Args:
            user_id: User identifier
            query: Optional query for relevance-based recall
            include_types: Memory types to include

        Returns:
            Formatted context string ready for prompt injection
        """
        lines = []

        # Get structured context by type
        if include_types is None:
            include_types = [
                MemoryType.PREFERENCE.value,
                MemoryType.FACT.value,
                MemoryType.INSTRUCTION.value,
            ]

        context = await self.get_user_context(user_id, include_types)

        # Format preferences
        if MemoryType.PREFERENCE.value in context:
            prefs = context[MemoryType.PREFERENCE.value]
            if prefs:
                lines.append("User preferences:")
                for pref in prefs:
                    lines.append(f"  - {pref}")

        # Format facts
        if MemoryType.FACT.value in context:
            facts = context[MemoryType.FACT.value]
            if facts:
                lines.append("Known facts about user:")
                for fact in facts:
                    lines.append(f"  - {fact}")

        # Format instructions
        if MemoryType.INSTRUCTION.value in context:
            instructions = context[MemoryType.INSTRUCTION.value]
            if instructions:
                lines.append("User-specific instructions:")
                for instr in instructions:
                    lines.append(f"  - {instr}")

        # Add query-relevant memories if query provided
        if query:
            relevant = await self.recall_memories(
                user_id=user_id,
                query=query,
                k=3,
            )
            if relevant:
                lines.append("\nRelevant context from memory:")
                for mem in relevant:
                    lines.append(f"  - {mem.content}")

        return "\n".join(lines) if lines else ""


# ============================================================================
# Factory / Singleton
# ============================================================================


@lru_cache(maxsize=1)
def get_agent_memory_service() -> Optional[AgentMemoryService]:
    """
    Get or create singleton AgentMemoryService.

    Returns None if agent memory is not enabled or vector store not configured.
    """
    settings = get_settings()

    if not settings.agent_memory_enabled:
        logger.debug("Agent memory is disabled in settings")
        return None

    if not settings.vector_backend:
        logger.warning(
            "Agent memory enabled but vector_backend not configured. "
            "Set vector_backend='qdrant' or 'memory' to enable."
        )
        return None

    provider = get_vector_store_provider()
    if not provider:
        logger.warning("Agent memory enabled but vector store provider unavailable")
        return None

    return AgentMemoryService(
        provider=provider,
        collection=settings.agent_memory_collection,
        max_results=settings.agent_memory_max_results,
        upsert_by_similarity=settings.agent_memory_upsert_by_similarity,
        similarity_threshold=settings.agent_memory_similarity_threshold,
    )


def clear_memory_service_cache() -> None:
    """Clear the singleton cache (useful for testing)."""
    get_agent_memory_service.cache_clear()

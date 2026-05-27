"""Filesystem-backed snapshot capture and replay helpers for AgentService.

WHY: Debugging complex agent flows benefits from a lightweight, local replay
mechanism that can capture resolved agent configuration, streamed execution
events, and the final response without depending on LangSmith.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from unittest.mock import Mock

from pydantic import BaseModel, Field

from inference_core.core.config import get_settings
from inference_core.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)

_SNAPSHOT_SCHEMA_VERSION = 1
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STORAGE_PATH = _PROJECT_ROOT / "debug" / "agent_run_snapshots"


class AgentSnapshotTokenSegment(BaseModel):
    """Serializable token event emitted during agent streaming."""

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSnapshotTraceEvent(BaseModel):
    """Serializable representation of one streamed event."""

    event_type: Literal["updates", "messages", "custom"]
    payload: Any = None
    segments: list[AgentSnapshotTokenSegment] = Field(default_factory=list)
    namespace: list[str] = Field(default_factory=list)


class AgentSnapshotResolvedConfig(BaseModel):
    """Resolved agent configuration captured for one execution."""

    agent_name: str
    model_name: str
    execution_mode: str = "local"
    system_prompt: str | None = None
    response_format: Any = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    middleware: list[dict[str, Any]] = Field(default_factory=list)
    instance_context: dict[str, Any] | None = None


class AgentSnapshotInput(BaseModel):
    """Normalized runtime input data used for matching and replay."""

    user_input: str
    context: Any = None
    configurable: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str
    query_text: str


class AgentRunSnapshot(BaseModel):
    """Full capture of one local AgentService run."""

    schema_version: int = _SNAPSHOT_SCHEMA_VERSION
    snapshot_id: str
    created_at: datetime
    resolved_config: AgentSnapshotResolvedConfig
    input: AgentSnapshotInput
    trace_events: list[AgentSnapshotTraceEvent] = Field(default_factory=list)
    response: dict[str, Any]


class AgentSnapshotMatch(BaseModel):
    """Best snapshot selected for replay."""

    snapshot: AgentRunSnapshot
    score: float
    match_mode: Literal["exact", "semantic"]


def serialize_snapshot_value(value: Any) -> Any:
    """Convert runtime values into JSON-safe structures.

    WHY: Agent state often contains pydantic models, LangChain message objects,
    UUIDs, and datetimes that should remain inspectable after capture.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()

    if isinstance(value, uuid.UUID):
        return str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, BaseModel):
        return serialize_snapshot_value(value.model_dump(mode="python"))

    if isinstance(value, dict):
        return {
            str(key): serialize_snapshot_value(sub_value)
            for key, sub_value in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [serialize_snapshot_value(item) for item in value]

    if isinstance(value, Mock):
        payload: dict[str, Any] = {"__class__": type(value).__name__}
        for attr in (
            "content",
            "type",
            "id",
            "name",
            "tool_calls",
            "response_metadata",
            "usage_metadata",
            "content_blocks",
            "chunk_position",
        ):
            if attr in value.__dict__:
                payload[attr] = serialize_snapshot_value(value.__dict__[attr])
        return payload if len(payload) > 1 else repr(value)

    if _looks_like_langchain_message(value):
        payload = {
            "type": getattr(value, "type", type(value).__name__),
            "content": serialize_snapshot_value(getattr(value, "content", None)),
        }
        for attr in (
            "id",
            "name",
            "tool_calls",
            "response_metadata",
            "usage_metadata",
            "content_blocks",
            "chunk_position",
        ):
            if hasattr(value, attr):
                payload[attr] = serialize_snapshot_value(getattr(value, attr))
        return payload

    return repr(value)


def build_snapshot_query_text(user_input: str, context: Any = None) -> str:
    """Build the normalized query text used for semantic snapshot matching."""

    normalized_user_input = _normalize_text(user_input)
    normalized_context = _normalize_text(serialize_snapshot_value(context))
    if normalized_context:
        return f"{normalized_user_input}\n{normalized_context}".strip()
    return normalized_user_input


def build_snapshot_fingerprint(
    *,
    agent_name: str,
    model_name: str,
    system_prompt: str | None,
    response_format: Any,
    user_input: str,
    context: Any,
) -> str:
    """Build a stable hash for exact snapshot replay matching."""

    fingerprint_payload = {
        "agent_name": agent_name,
        "model_name": model_name,
        "system_prompt": _normalize_text(system_prompt or ""),
        "response_format": serialize_snapshot_value(response_format),
        "user_input": _normalize_text(user_input),
        "context": serialize_snapshot_value(context),
    }
    encoded = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def rebuild_agent_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe response payload suitable for AgentResponse.model_validate."""

    return json.loads(json.dumps(payload))


class FilesystemAgentSnapshotStore:
    """Filesystem-backed storage for captured agent run snapshots."""

    def __init__(self, base_path: str | Path | None = None) -> None:
        resolved_base_path = base_path or getattr(
            get_settings(),
            "agent_snapshot_storage_path",
            str(_DEFAULT_STORAGE_PATH),
        )
        self.base_path = _resolve_storage_path(resolved_base_path)

    def save_snapshot(self, snapshot: AgentRunSnapshot) -> Path:
        """Persist one snapshot as a JSON file and return its path."""

        self.base_path.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{snapshot.created_at.strftime('%Y%m%dT%H%M%S%f')}_"
            f"{snapshot.snapshot_id}.json"
        )
        path = self.base_path / filename
        path.write_text(
            snapshot.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Saved agent snapshot %s to %s", snapshot.snapshot_id, path)
        return path

    def iter_snapshots(
        self,
        *,
        agent_name: str | None = None,
        model_name: str | None = None,
    ) -> list[AgentRunSnapshot]:
        """Load snapshots from disk ordered newest-first."""

        if not self.base_path.exists():
            return []

        snapshots: list[AgentRunSnapshot] = []
        for path in sorted(self.base_path.glob("*.json"), reverse=True):
            try:
                snapshot = AgentRunSnapshot.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                logger.warning("Skipping unreadable agent snapshot %s: %s", path, exc)
                continue
            if agent_name and snapshot.resolved_config.agent_name != agent_name:
                continue
            if model_name and snapshot.resolved_config.model_name != model_name:
                continue
            snapshots.append(snapshot)
        return snapshots


class AgentSnapshotMatcher:
    """Find the best captured snapshot for one agent invocation."""

    def __init__(
        self,
        store: FilesystemAgentSnapshotStore,
        *,
        settings: Any | None = None,
        embedding_service: Any | None = None,
    ) -> None:
        self.store = store
        self._settings = settings
        self._embedding_service = embedding_service

    def find_best_match(
        self,
        *,
        agent_name: str,
        model_name: str,
        system_prompt: str | None,
        response_format: Any,
        user_input: str,
        context: Any,
        match_mode: str,
        min_score: float,
    ) -> AgentSnapshotMatch | None:
        """Return the strongest matching snapshot or None.

        WHY: Debug replay should prefer exact matches and only fall back to
        semantic similarity when explicitly enabled.
        """

        expected_fingerprint = build_snapshot_fingerprint(
            agent_name=agent_name,
            model_name=model_name,
            system_prompt=system_prompt,
            response_format=response_format,
            user_input=user_input,
            context=context,
        )
        candidates = self.store.iter_snapshots(
            agent_name=agent_name,
            model_name=model_name,
        )

        for snapshot in candidates:
            if snapshot.input.fingerprint == expected_fingerprint:
                return AgentSnapshotMatch(
                    snapshot=snapshot,
                    score=1.0,
                    match_mode="exact",
                )

        if match_mode == "exact":
            return None

        query_text = build_snapshot_query_text(user_input, context)
        if not query_text:
            return None

        semantic_candidates = [
            snapshot for snapshot in candidates if snapshot.input.query_text
        ]
        semantic_scores = self._semantic_similarity_scores(
            query_text,
            [snapshot.input.query_text for snapshot in semantic_candidates],
        )

        best_match: AgentSnapshotMatch | None = None
        for snapshot, score in zip(semantic_candidates, semantic_scores):
            if score < min_score:
                continue
            if best_match is None or score > best_match.score:
                best_match = AgentSnapshotMatch(
                    snapshot=snapshot,
                    score=score,
                    match_mode="semantic",
                )

        return best_match

    def _semantic_similarity_scores(
        self,
        query_text: str,
        candidate_texts: list[str],
    ) -> list[float]:
        if not query_text or not candidate_texts:
            return []

        settings = self._settings or get_settings()
        if getattr(settings, "embedding_backend", "local") == "fake":
            logger.info(
                "Skipping semantic snapshot replay because EMBEDDING_BACKEND=fake"
            )
            return []

        try:
            embedding_service = self._embedding_service or get_embedding_service()
            vectors = embedding_service.embed_texts([query_text, *candidate_texts])
        except Exception:
            logger.exception(
                "Semantic snapshot replay embeddings failed; falling back to exact-only matching"
            )
            return []

        expected_count = len(candidate_texts) + 1
        if len(vectors) != expected_count:
            logger.warning(
                "Semantic snapshot replay embedding batch returned %d vectors for %d texts",
                len(vectors),
                expected_count,
            )
            return []

        query_vector = vectors[0]
        return [
            _cosine_similarity(query_vector, candidate_vector)
            for candidate_vector in vectors[1:]
        ]

    def _semantic_similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        scores = self._semantic_similarity_scores(left, [right])
        if not scores:
            return 0.0
        return scores[0]


def get_agent_snapshot_store(
    settings: Any | None = None,
) -> FilesystemAgentSnapshotStore:
    """Return the snapshot store configured for the current process."""

    base_path = None
    if settings is not None:
        base_path = getattr(settings, "agent_snapshot_storage_path", None)
    return FilesystemAgentSnapshotStore(base_path=base_path)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(serialize_snapshot_value(value), sort_keys=True)
    return " ".join(value.lower().split())


def _resolve_storage_path(base_path: str | Path) -> Path:
    path = Path(base_path)
    if not path.is_absolute():
        return _PROJECT_ROOT / path
    return path


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _looks_like_langchain_message(value: Any) -> bool:
    class_name = type(value).__name__
    if class_name.endswith("Message") or class_name.endswith("MessageChunk"):
        return True
    module_name = getattr(type(value), "__module__", "")
    return module_name.startswith("langchain")

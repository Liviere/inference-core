"""Unit tests for filesystem-backed agent snapshot capture helpers."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from inference_core.services.agent_snapshot_service import (
    AgentRunSnapshot,
    AgentSnapshotInput,
    AgentSnapshotMatcher,
    AgentSnapshotResolvedConfig,
    FilesystemAgentSnapshotStore,
    build_snapshot_fingerprint,
    build_snapshot_query_text,
)


def _build_snapshot(
    *, snapshot_id: str, user_input: str, query_text: str
) -> AgentRunSnapshot:
    resolved_config = AgentSnapshotResolvedConfig(
        agent_name="test_agent",
        model_name="gpt-4o",
        execution_mode="local",
        system_prompt="Be concise.",
        response_format=None,
        tools=[],
        middleware=[],
        instance_context=None,
    )
    return AgentRunSnapshot(
        snapshot_id=snapshot_id,
        created_at=datetime.now(UTC),
        resolved_config=resolved_config,
        input=AgentSnapshotInput(
            user_input=user_input,
            context={"mode": "test"},
            configurable={"thread_id": "t-1"},
            fingerprint=build_snapshot_fingerprint(
                agent_name="test_agent",
                model_name="gpt-4o",
                system_prompt="Be concise.",
                response_format=None,
                user_input=user_input,
                context={"mode": "test"},
            ),
            query_text=query_text,
        ),
        trace_events=[],
        response={
            "result": {"messages": [{"type": "ai", "content": "stored"}]},
            "steps": [],
            "metadata": {
                "model_name": "gpt-4o",
                "tools_used": [],
                "start_time": datetime.now(UTC).isoformat(),
                "end_time": datetime.now(UTC).isoformat(),
            },
            "cost_metrics": None,
            "structured_response": None,
        },
    )


class TestAgentSnapshotMatcher:
    def test_returns_exact_match_before_semantic_fallback(self, tmp_path):
        store = FilesystemAgentSnapshotStore(base_path=tmp_path)
        exact_snapshot = _build_snapshot(
            snapshot_id="exact",
            user_input="hello billing",
            query_text=build_snapshot_query_text("hello billing", {"mode": "test"}),
        )
        semantic_snapshot = _build_snapshot(
            snapshot_id="semantic",
            user_input="different phrase",
            query_text="billing invoices",
        )
        store.save_snapshot(exact_snapshot)
        store.save_snapshot(semantic_snapshot)

        embedding_service = MagicMock()
        matcher = AgentSnapshotMatcher(
            store,
            settings=SimpleNamespace(embedding_backend="local"),
            embedding_service=embedding_service,
        )
        match = matcher.find_best_match(
            agent_name="test_agent",
            model_name="gpt-4o",
            system_prompt="Be concise.",
            response_format=None,
            user_input="hello billing",
            context={"mode": "test"},
            match_mode="exact_or_semantic",
            min_score=0.1,
        )

        assert match is not None
        assert match.match_mode == "exact"
        assert match.snapshot.snapshot_id == "exact"
        embedding_service.embed_texts.assert_not_called()

    def test_batches_semantic_embeddings_and_picks_best_match(self):
        lower_match = _build_snapshot(
            snapshot_id="lower",
            user_input="recorded billing help",
            query_text="billing invoices help",
        )
        stronger_match = _build_snapshot(
            snapshot_id="stronger",
            user_input="recorded invoice copy",
            query_text="invoice copy request",
        )
        store = MagicMock()
        store.iter_snapshots.return_value = [lower_match, stronger_match]
        embedding_service = MagicMock()
        embedding_service.embed_texts.return_value = [
            [1.0, 0.0],
            [0.8, 0.6],
            [1.0, 0.0],
        ]

        matcher = AgentSnapshotMatcher(
            store,
            settings=SimpleNamespace(embedding_backend="local"),
            embedding_service=embedding_service,
        )

        match = matcher.find_best_match(
            agent_name="test_agent",
            model_name="gpt-4o",
            system_prompt="Be concise.",
            response_format=None,
            user_input="need invoice billing help",
            context={"mode": "test"},
            match_mode="exact_or_semantic",
            min_score=0.92,
        )

        assert match is not None
        assert match.match_mode == "semantic"
        assert match.snapshot.snapshot_id == "stronger"
        embedding_service.embed_texts.assert_called_once_with(
            [
                build_snapshot_query_text(
                    "need invoice billing help",
                    {"mode": "test"},
                ),
                "billing invoices help",
                "invoice copy request",
            ]
        )

    def test_semantic_match_respects_min_score(self):
        snapshot = _build_snapshot(
            snapshot_id="semantic",
            user_input="recorded billing help",
            query_text="billing invoices help",
        )
        store = MagicMock()
        store.iter_snapshots.return_value = [snapshot]
        embedding_service = MagicMock()
        embedding_service.embed_texts.return_value = [
            [1.0, 0.0],
            [0.8, 0.6],
        ]

        matcher = AgentSnapshotMatcher(
            store,
            settings=SimpleNamespace(embedding_backend="local"),
            embedding_service=embedding_service,
        )

        match = matcher.find_best_match(
            agent_name="test_agent",
            model_name="gpt-4o",
            system_prompt="Be concise.",
            response_format=None,
            user_input="need invoice billing help",
            context={"mode": "test"},
            match_mode="exact_or_semantic",
            min_score=0.92,
        )

        assert match is None
        embedding_service.embed_texts.assert_called_once()

    def test_skips_semantic_match_when_fake_backend_is_configured(self):
        snapshot = _build_snapshot(
            snapshot_id="semantic",
            user_input="recorded billing help",
            query_text="billing invoices help",
        )
        store = MagicMock()
        store.iter_snapshots.return_value = [snapshot]
        embedding_service = MagicMock()

        matcher = AgentSnapshotMatcher(
            store,
            settings=SimpleNamespace(embedding_backend="fake"),
            embedding_service=embedding_service,
        )

        match = matcher.find_best_match(
            agent_name="test_agent",
            model_name="gpt-4o",
            system_prompt="Be concise.",
            response_format=None,
            user_input="need invoice billing help",
            context={"mode": "test"},
            match_mode="exact_or_semantic",
            min_score=0.92,
        )

        assert match is None
        embedding_service.embed_texts.assert_not_called()

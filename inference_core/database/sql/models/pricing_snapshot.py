"""
LLM Pricing Snapshot Model

Stores unique pricing configurations referenced by LLM request logs to avoid
duplicating pricing JSON blobs on every row.

Uniqueness is enforced via a deterministic SHA256 hash over the canonical
serialization of: provider, model_name, currency, input_cost_per_1k,
output_cost_per_1k, extras (dimension -> cost_per_1k).
"""

import hashlib
import json
import uuid
from typing import Dict, Optional

from sqlalchemy import Index, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base, SmartJSON, TimestampMixin


class LLMPricingSnapshot(Base, TimestampMixin):
    """Pricing snapshot referenced by `LLMRequestLog` rows.

    Columns:
        id: UUID primary key
        snapshot_hash: Deterministic hash of pricing config (hex SHA256)
        provider: LLM provider name
        model_name: Model name
        input_cost_per_1k: Input token cost per 1K tokens
        output_cost_per_1k: Output token cost per 1K tokens
        currency: Currency code (e.g. USD)
        extras: JSON mapping extra dimension -> {"cost_per_1k": float}
    """

    __tablename__ = "llm_pricing_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique identifier",
    )

    snapshot_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
        doc="SHA256 hash of pricing configuration",
    )
    provider: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True, doc="LLM provider"
    )
    model_name: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, doc="Model name"
    )
    input_cost_per_1k: Mapped[float] = mapped_column(
        Numeric(18, 6), nullable=False, doc="Input cost per 1K tokens"
    )
    output_cost_per_1k: Mapped[float] = mapped_column(
        Numeric(18, 6), nullable=False, doc="Output cost per 1K tokens"
    )
    currency: Mapped[str] = mapped_column(
        String(10), nullable=False, doc="Currency code"
    )
    extras: Mapped[Optional[dict]] = mapped_column(
        SmartJSON(), nullable=True, doc="Extra dimensions pricing mapping"
    )

    __table_args__ = (
        UniqueConstraint("snapshot_hash", name="uq_llm_pricing_snapshot_hash"),
        Index("ix_llm_pricing_provider_model", "provider", "model_name"),
    )

    @staticmethod
    def compute_hash(
        provider: str,
        model_name: str,
        currency: str,
        input_cost_per_1k: float,
        output_cost_per_1k: float,
        extras: Optional[Dict[str, Dict[str, float]]],
    ) -> str:
        """Compute deterministic SHA256 hash for pricing config.

        Extras dict is normalized with sorted keys.
        """
        canonical = {
            "provider": provider,
            "model_name": model_name,
            "currency": currency,
            "input_cost_per_1k": float(input_cost_per_1k),
            "output_cost_per_1k": float(output_cost_per_1k),
            "extras": {
                k: {"cost_per_1k": float(v.get("cost_per_1k"))}
                for k, v in sorted((extras or {}).items())
            },
        }
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:  # pragma: no cover - repr utility
        return (
            f"<LLMPricingSnapshot(id={self.id}, provider={self.provider}, model={self.model_name}, "
            f"hash={self.snapshot_hash[:8]}...)>"
        )

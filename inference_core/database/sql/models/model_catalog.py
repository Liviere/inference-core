"""
Model Catalog Models

What providers list as their models, as last read, with a log of what changed
between readings.

- ``llm_catalog_models``: one row per model a provider lists. A model that is
  no longer listed keeps its row, marked removed, so a model that comes back
  is told from a new one.
- ``llm_catalog_events``: what changed, one row per change.
- ``llm_catalog_provider_state``: when each provider was last read and how
  that went.

Columns hold what the catalog acts on. Everything that is only shown (names,
limits, prices) is in ``attributes``.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base, SmartJSON, TimestampMixin


class LLMCatalogModel(Base, TimestampMixin):
    """A model as its provider lists it."""

    __tablename__ = "llm_catalog_models"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4, doc="Unique identifier"
    )
    provider: Mapped[str] = mapped_column(
        String(50), nullable=False, doc="LLM provider"
    )
    model_id: Mapped[str] = mapped_column(
        String(255), nullable=False, doc="The provider's id of the model"
    )
    source: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="list",
        doc="'list' for a model in the listing, 'lookup' for one asked about by id",
    )
    kind: Mapped[str] = mapped_column(
        String(20), nullable=False, default="unknown", doc="What the model is for"
    )
    lifecycle: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="unknown",
        doc="active, deprecated, retired or unknown",
    )
    deprecated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, doc="When the model was deprecated"
    )
    retires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, doc="When the model stops answering"
    )
    replacement: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, doc="The model the provider names instead"
    )
    aliases: Mapped[Optional[List[str]]] = mapped_column(
        SmartJSON(), nullable=True, doc="Other ids that reach this model"
    )
    attributes: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        SmartJSON(), nullable=True, doc="What else the provider says about it"
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, doc="First reading it was in"
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, doc="Latest reading it was in"
    )
    missed_runs: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, doc="Readings in a row it was not in"
    )
    removed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, doc="When it was taken for removed"
    )

    __table_args__ = (
        UniqueConstraint("provider", "model_id", name="uq_llm_catalog_model"),
    )

    def __repr__(self) -> str:  # pragma: no cover - repr utility
        return f"<LLMCatalogModel(provider={self.provider}, model={self.model_id})>"


class LLMCatalogEvent(Base):
    """One change between two readings of a provider's listing."""

    __tablename__ = "llm_catalog_events"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4, doc="Unique identifier"
    )
    provider: Mapped[str] = mapped_column(
        String(50), nullable=False, doc="LLM provider"
    )
    model_id: Mapped[str] = mapped_column(
        String(255), nullable=False, doc="The provider's id of the model"
    )
    event_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        doc="added, removed, reappeared or lifecycle_changed",
    )
    details: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        SmartJSON(), nullable=True, doc="What it was before and after"
    )
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, doc="The reading that showed it"
    )

    __table_args__ = (
        Index("ix_llm_catalog_events_detected", "detected_at"),
        Index("ix_llm_catalog_events_provider", "provider", "detected_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - repr utility
        return (
            f"<LLMCatalogEvent(provider={self.provider}, model={self.model_id}, "
            f"type={self.event_type})>"
        )


class LLMCatalogProviderState(Base):
    """When a provider's listing was last read, and how that went."""

    __tablename__ = "llm_catalog_provider_state"

    provider: Mapped[str] = mapped_column(
        String(50), primary_key=True, doc="LLM provider"
    )
    last_attempt_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Latest reading, whatever came of it",
    )
    last_success_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, doc="Latest reading that worked"
    )
    last_error: Mapped[Optional[str]] = mapped_column(
        String(200),
        nullable=True,
        doc="Why the latest reading failed: the kind of error and the status code",
    )
    model_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, doc="Models in the latest listing"
    )
    baselined_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="First reading that worked; changes are reported from then on",
    )

    def __repr__(self) -> str:  # pragma: no cover - repr utility
        return f"<LLMCatalogProviderState(provider={self.provider})>"

"""
LLM Request Log Model

Database model for persistent LLM usage and cost logging.
"""

import uuid
from datetime import UTC, datetime
from typing import Optional

from sqlalchemy import Boolean, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base, SmartJSON, TimestampMixin


class LLMRequestLog(Base, TimestampMixin):
    """
    LLM Request Log model for persistent usage and cost tracking

    Attributes:
        id: Unique identifier (UUID)
        created_at: Request timestamp
        user_id: Optional user association (nullable FK)
        session_id: Optional session identifier
        task_type: Type of task (completion, chat, batch, etc.)
        request_mode: Request mode (sync, async, streaming)
        model_name: LLM model used
        provider: LLM provider (openai, claude, etc.)
        request_id: Correlation ID (Celery task id, etc.)
        latency_ms: Request latency in milliseconds
        success: Whether request completed successfully
        error_type: Error classification if failed
        error_message: Truncated error message if failed

        # Core token dimensions (first-class columns)
        input_tokens: Input token count
        output_tokens: Output token count
        total_tokens: Total token count (computed server-side)

        # Flexible token dimensions (JSON)
        extra_tokens: Additional token types (reasoning, cache, etc.)

        # Audit and transparency
        usage_raw: Full raw usage response from provider
    pricing_snapshot_id: FK to pricing snapshot row (deduplicated pricing config)

        # Core cost fields
        cost_input_usd: Input token cost in USD
        cost_output_usd: Output token cost in USD
        cost_extras_usd: Sum of extra dimension costs in USD
        cost_total_usd: Total cost in USD

        # Flexible cost breakdown (JSON)
        extra_costs: Detailed cost breakdown for extra dimensions

        # Cost metadata
        cost_estimated: True if some tokens were unpriced
        context_multiplier: Context tier multiplier applied

        # Request characteristics
        streamed: Whether request used streaming
        partial: Whether request was aborted/incomplete

        # Extended details (agents)
        details: Extended execution details for agents (per-step breakdown, etc.)
    """

    __tablename__ = "llm_request_logs"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique identifier",
    )

    # User association (nullable)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        doc="Associated user ID (nullable)",
    )

    # Request metadata
    session_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, doc="Session identifier"
    )
    task_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="Task type (completion, chat, etc.)",
    )
    request_mode: Mapped[str] = mapped_column(
        String(20), nullable=False, doc="Request mode (sync, async, streaming)"
    )
    model_name: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, doc="LLM model name"
    )
    provider: Mapped[str] = mapped_column(
        String(50), nullable=False, doc="LLM provider"
    )
    request_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, doc="Correlation ID (Celery task id, etc.)"
    )

    # Performance and status
    latency_ms: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, doc="Request latency in milliseconds"
    )
    success: Mapped[bool] = mapped_column(
        Boolean, nullable=False, index=True, doc="Whether request succeeded"
    )
    error_type: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, doc="Error classification"
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, doc="Truncated error message"
    )

    # Core token dimensions (first-class columns - required for pricing)
    input_tokens: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, doc="Input token count"
    )
    output_tokens: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, doc="Output token count"
    )
    total_tokens: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, doc="Total token count (computed)"
    )

    # Flexible token dimensions (JSON for extensibility)
    extra_tokens: Mapped[Optional[dict]] = mapped_column(
        SmartJSON(),
        nullable=True,
        doc="Additional token types (reasoning, cache, etc.) as JSON",
    )

    # Audit and transparency
    usage_raw: Mapped[dict] = mapped_column(
        SmartJSON(),
        nullable=False,
        doc="Full raw usage response from provider",
    )
    pricing_snapshot_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("llm_pricing_snapshots.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
        doc="Reference to pricing snapshot",
    )

    # Core cost fields (high precision for financial data)
    cost_input_usd: Mapped[Optional[float]] = mapped_column(
        Numeric(18, 6), nullable=True, doc="Input token cost in USD"
    )
    cost_output_usd: Mapped[Optional[float]] = mapped_column(
        Numeric(18, 6), nullable=True, doc="Output token cost in USD"
    )
    cost_extras_usd: Mapped[Optional[float]] = mapped_column(
        Numeric(18, 6), nullable=True, doc="Extra dimensions cost in USD"
    )
    cost_total_usd: Mapped[Optional[float]] = mapped_column(
        Numeric(18, 6), nullable=True, doc="Total cost in USD"
    )

    # Flexible cost breakdown (JSON)
    extra_costs: Mapped[Optional[dict]] = mapped_column(
        SmartJSON(),
        nullable=True,
        doc="Detailed cost breakdown for extra dimensions",
    )

    # Cost metadata
    cost_estimated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, doc="True if some tokens were unpriced"
    )
    context_multiplier: Mapped[Optional[float]] = mapped_column(
        Numeric(10, 4), nullable=True, doc="Context tier multiplier applied"
    )

    # Request characteristics
    streamed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, doc="Whether request used streaming"
    )
    partial: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="Whether request was aborted/incomplete",
    )

    # Extended details for agent executions (per-step breakdown, tool calls, etc.)
    details: Mapped[Optional[dict]] = mapped_column(
        SmartJSON(),
        nullable=True,
        doc="Extended execution details (agent steps, tool calls, latencies, etc.)",
    )

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_llm_logs_created_at", "created_at"),
        Index("ix_llm_logs_user_created", "user_id", "created_at"),
        Index("ix_llm_logs_model_created", "model_name", "created_at"),
        Index("ix_llm_logs_task_success", "task_type", "success"),
        Index("ix_llm_logs_provider_model", "provider", "model_name"),
    )

    def __repr__(self) -> str:
        return (
            f"<LLMRequestLog(id={self.id}, model={self.model_name}, "
            f"task={self.task_type}, success={self.success})>"
        )

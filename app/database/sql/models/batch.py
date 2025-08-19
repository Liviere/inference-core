"""
Batch Domain Models

Database models for batch processing: BatchJob, BatchItem, and BatchEvent.
Supports various LLM provider batch operations with comprehensive tracking.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import DateTime
from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from ..base import FullAuditModel, SmartJSON


class BatchJobStatus(str, Enum):
    """Batch job status enumeration"""

    CREATED = "created"
    SUBMITTED = "submitted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class BatchItemStatus(str, Enum):
    """Batch item status enumeration"""

    QUEUED = "queued"
    SENT = "sent"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchEventType(str, Enum):
    """Batch event type enumeration"""

    STATUS_CHANGE = "status_change"
    ITEM_UPDATE = "item_update"
    ERROR = "error"
    PROGRESS_UPDATE = "progress_update"
    # Semantic lifecycle markers (non-status-change but important timeline anchors)
    SUBMITTED = "submitted"  # Provider submission finished (provider_batch_id assigned)
    FETCH_COMPLETED = "fetch_completed"  # Results fetched and item statuses updated


class BatchJob(FullAuditModel):
    """
    Batch job model for tracking batch processing operations

    Attributes:
        provider: LLM provider (openai, anthropic, etc.)
        model: Model name used for the batch
        status: Current job status
        provider_batch_id: External batch ID from provider
        mode: Processing mode (chat, embedding, completion, custom)
        submitted_at: When batch was submitted to provider
        completed_at: When batch completed processing
        request_count: Total number of requests in batch
        success_count: Number of successful requests
        error_count: Number of failed requests
        config_json: Batch configuration as JSON
        error_summary: Summary of errors if any
        result_uri: URI where results can be downloaded
        metadata_json: Additional metadata (inherited from FullAuditModel)
    """

    __tablename__ = "batch_jobs"

    def __init__(self, **kwargs):
        # Set Python-level defaults
        kwargs.setdefault("request_count", 0)
        kwargs.setdefault("success_count", 0)
        kwargs.setdefault("error_count", 0)
        super().__init__(**kwargs)

    # Core batch identification
    provider: Mapped[str] = mapped_column(
        String(50), nullable=False, doc="LLM provider name"
    )
    model: Mapped[str] = mapped_column(String(100), nullable=False, doc="Model name")
    # Use SQLAlchemy Enum with native_enum disabled for cross-DB compatibility
    status: Mapped[BatchJobStatus] = mapped_column(
        SAEnum(BatchJobStatus, native_enum=False, length=20),
        nullable=False,
        doc="Current job status (enum)",
    )
    provider_batch_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, doc="External provider batch ID"
    )

    # Batch configuration
    mode: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        doc="Processing mode (chat, embedding, completion, custom)",
    )

    # Timing fields
    submitted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When batch was submitted to provider",
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, doc="When batch completed processing"
    )

    # Progress tracking
    request_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
        doc="Total number of requests",
    )
    success_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
        doc="Number of successful requests",
    )
    error_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
        doc="Number of failed requests",
    )

    # Configuration and results
    config_json: Mapped[Optional[str]] = mapped_column(
        SmartJSON(), nullable=True, doc="Batch configuration as JSON"
    )
    error_summary: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, doc="Summary of errors if any"
    )
    result_uri: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True, doc="URI where results can be downloaded"
    )

    # Relationships
    # Use default (select) loading to support selectinload eager loading in services
    items: Mapped[list["BatchItem"]] = relationship(
        "BatchItem",
        back_populates="batch_job",
        cascade="all, delete-orphan",
    )
    events: Mapped[list["BatchEvent"]] = relationship(
        "BatchEvent",
        back_populates="batch_job",
        cascade="all, delete-orphan",
    )

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_batch_jobs_provider", "provider"),
        Index("ix_batch_jobs_status", "status"),
        Index("ix_batch_jobs_provider_batch_id", "provider_batch_id"),
        Index("ix_batch_jobs_status_provider", "status", "provider"),
        Index(
            "ix_batch_jobs_provider_batch_composite", "provider", "provider_batch_id"
        ),
        Index("ix_batch_jobs_submitted_at", "submitted_at"),
        Index("ix_batch_jobs_mode_status", "mode", "status"),
    )

    @property
    def completion_rate(self) -> float:
        """Calculate completion rate as percentage"""
        if not self.request_count or self.request_count == 0:
            return 0.0
        success = self.success_count or 0
        error = self.error_count or 0
        return (success + error) / self.request_count * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage of completed items"""
        success = self.success_count or 0
        error = self.error_count or 0
        completed = success + error
        if completed == 0:
            return 0.0
        return success / completed * 100

    @property
    def is_complete(self) -> bool:
        """Check if batch is complete"""
        return self.status in [
            BatchJobStatus.COMPLETED,
            BatchJobStatus.FAILED,
            BatchJobStatus.CANCELLED,
        ]

    @property
    def pending_count(self) -> int:
        """Calculate number of pending requests"""
        request_count = self.request_count or 0
        success = self.success_count or 0
        error = self.error_count or 0
        return request_count - success - error

    def update_counts(self, success_delta: int = 0, error_delta: int = 0):
        """Update success and error counts"""
        self.success_count += success_delta
        self.error_count += error_delta

    def __repr__(self) -> str:
        return f"<BatchJob(id={self.id}, provider={self.provider}, status={self.status}, requests={self.request_count})>"


class BatchItem(FullAuditModel):
    """
    Individual item within a batch job

    Attributes:
        batch_job_id: Foreign key to parent batch job
        sequence_index: Order within the batch
        custom_external_id: Optional external identifier
        input_payload: Request data as JSON
        output_payload: Response data as JSON (when completed)
        status: Current item status
        error_detail: Error details if processing failed
    """

    __tablename__ = "batch_items"

    def __init__(self, **kwargs):
        # Set Python-level defaults
        kwargs.setdefault("status", BatchItemStatus.QUEUED)
        super().__init__(**kwargs)

    # Relationship to batch job
    batch_job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("batch_jobs.id", ondelete="CASCADE"),
        nullable=False,
        doc="Parent batch job ID",
    )

    # Item identification
    sequence_index: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="Order within the batch"
    )
    custom_external_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, doc="Optional external identifier"
    )

    # Item data
    input_payload: Mapped[Optional[str]] = mapped_column(
        SmartJSON(), nullable=True, doc="Request data as JSON"
    )
    output_payload: Mapped[Optional[str]] = mapped_column(
        SmartJSON(), nullable=True, doc="Response data as JSON"
    )

    # Status tracking
    status: Mapped[BatchItemStatus] = mapped_column(
        SAEnum(BatchItemStatus, native_enum=False, length=20),
        nullable=False,
        server_default=text("'queued'"),
        doc="Current item status (enum)",
    )
    error_detail: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, doc="Error details if processing failed"
    )

    # Relationship back to job
    batch_job: Mapped["BatchJob"] = relationship("BatchJob", back_populates="items")

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_batch_items_batch_job_id", "batch_job_id"),
        Index("ix_batch_items_status", "status"),
        Index("ix_batch_items_external_id", "custom_external_id"),
        Index("ix_batch_items_job_sequence", "batch_job_id", "sequence_index"),
        Index("ix_batch_items_job_status", "batch_job_id", "status"),
    )

    @property
    def is_completed(self) -> bool:
        """Check if item processing is complete"""
        return self.status in [BatchItemStatus.COMPLETED, BatchItemStatus.FAILED]

    @property
    def is_successful(self) -> bool:
        """Check if item completed successfully"""
        return self.status == BatchItemStatus.COMPLETED

    def __repr__(self) -> str:
        return f"<BatchItem(id={self.id}, batch_job_id={self.batch_job_id}, sequence={self.sequence_index}, status={self.status})>"


class BatchEvent(FullAuditModel):
    """
    Event tracking for batch processing

    Attributes:
        batch_job_id: Foreign key to parent batch job
        event_type: Type of event (status_change, error, etc.)
        old_status: Previous status (for status change events)
        new_status: New status (for status change events)
        event_timestamp: When the event occurred
        event_data: Additional event data as JSON
    """

    __tablename__ = "batch_events"

    # Relationship to batch job
    batch_job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("batch_jobs.id", ondelete="CASCADE"),
        nullable=False,
        doc="Parent batch job ID",
    )

    # Event details
    event_type: Mapped[BatchEventType] = mapped_column(
        SAEnum(BatchEventType, native_enum=False, length=30),
        nullable=False,
        doc="Type of event (enum)",
    )
    old_status: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True, doc="Previous status (for status change events)"
    )
    new_status: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True, doc="New status (for status change events)"
    )
    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc="When the event occurred",
    )
    event_data: Mapped[Optional[str]] = mapped_column(
        SmartJSON(), nullable=True, doc="Additional event data as JSON"
    )

    # Relationship back to job
    batch_job: Mapped["BatchJob"] = relationship("BatchJob", back_populates="events")

    # Indexes for common queries
    __table_args__ = (
        Index("ix_batch_events_batch_job_id", "batch_job_id"),
        Index("ix_batch_events_event_type", "event_type"),
        Index("ix_batch_events_timestamp", "event_timestamp"),
        Index("ix_batch_events_job_timestamp", "batch_job_id", "event_timestamp"),
        Index("ix_batch_events_type_timestamp", "event_type", "event_timestamp"),
    )

    def __repr__(self) -> str:
        return f"<BatchEvent(id={self.id}, batch_job_id={self.batch_job_id}, type={self.event_type}, timestamp={self.event_timestamp})>"

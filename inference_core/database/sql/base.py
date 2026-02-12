"""
Base Database Model

Base model class with common fields and functionality
for all database models.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Boolean, DateTime, String, Text
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import Mapped, declarative_mixin, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.types import TypeDecorator, Uuid

from .connection import Base


class SmartJSON(TypeDecorator):
    """
    Smart JSON type that adapts to different databases:
    - PostgreSQL: Uses native JSON type
    - MySQL 5.7+: Uses native JSON type
    - SQLite/older MySQL: Uses TEXT with JSON validation
    """

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSON())
        elif dialect.name == "mysql":
            # MySQL 5.7+ supports JSON
            return dialect.type_descriptor(JSON())
        else:
            # Fallback to TEXT for SQLite and older databases
            return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is not None:
            if dialect.name in ("postgresql", "mysql"):
                return value  # Native JSON support
            else:
                import json

                return json.dumps(value) if value is not None else None
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            if dialect.name in ("postgresql", "mysql"):
                return value  # Native JSON support
            else:
                import json

                try:
                    return json.loads(value) if value is not None else None
                except (json.JSONDecodeError, TypeError):
                    return None
        return None


@declarative_mixin
class TimestampMixin:
    """Mixin for timestamp fields"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="Record creation timestamp",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="Record last update timestamp",
    )


class BaseModel(Base, TimestampMixin):
    """
    Base model class with common fields and functionality

    Provides:
    - UUID primary key
    - Timestamp fields (created_at, updated_at)
    - Common utility methods
    """

    __abstract__ = True

    # Primary key using UUID
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),  # Database-agnostic UUID type as Python UUID object
        primary_key=True,
        default=uuid.uuid4,  # Function reference, not call
        doc="Unique identifier",
    )

    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Generate table name from class name"""
        return cls.__name__.lower()

    def to_dict(
        self, exclude: Optional[set] = None, include_relationships: bool = False
    ) -> Dict[str, Any]:
        """
        Convert model instance to dictionary

        Args:
            exclude: Set of fields to exclude
            include_relationships: Whether to include relationship data

        Returns:
            Dictionary representation
        """
        # Import here to avoid circular imports
        from inference_core.database.sql.serializers import ModelSerializer

        return ModelSerializer.to_dict(self, exclude, include_relationships)

    def update_from_dict(self, data: Dict[str, Any], exclude: Optional[set] = None):
        """
        Update model instance from dictionary

        Args:
            data: Dictionary with new values
            exclude: Set of fields to exclude from update
        """
        exclude = exclude or {"id", "created_at", "updated_at"}

        for key, value in data.items():
            if key not in exclude and hasattr(self, key):
                # Handle UUID conversion from string
                if value is not None and key in ["id", "created_by", "updated_by"]:
                    if isinstance(value, str):
                        try:
                            value = uuid.UUID(value)
                        except ValueError:
                            continue  # Skip invalid UUID strings
                setattr(self, key, value)

    @classmethod
    def create_from_dict(cls, data: Dict[str, Any], exclude: Optional[set] = None):
        """
        Create new instance from dictionary with automatic type conversion

        Args:
            data: Dictionary with values (UUIDs can be strings, dates can be ISO strings)
            exclude: Set of fields to exclude

        Returns:
            New model instance
        """
        # Import here to avoid circular imports
        from inference_core.database.sql.serializers import ModelSerializer

        return ModelSerializer.from_dict(cls, data, exclude)

    def __repr__(self) -> str:
        """String representation of the model"""
        return f"<{self.__class__.__name__}(id={self.id})>"

    def __str__(self) -> str:
        """Human-readable string representation"""
        return self.__repr__()


@declarative_mixin
class AuditMixin:
    """Mixin for audit fields"""

    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True), nullable=True, doc="ID of user who created the record"
    )
    updated_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True), nullable=True, doc="ID of user who last updated the record"
    )


@declarative_mixin
class MetadataMixin:
    """Mixin for metadata fields"""

    metadata_json: Mapped[Optional[str]] = mapped_column(
        SmartJSON(),  # Automatically adapts to database capabilities
        nullable=True,
        doc="Additional metadata as JSON",
    )

    tags: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True, doc="Comma-separated tags"
    )

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata as dictionary

        Returns:
            Metadata dictionary
        """
        if self.metadata_json:
            import json

            try:
                return json.loads(self.metadata_json)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    def set_metadata(self, metadata: Dict[str, Any]):
        """
        Set metadata from dictionary

        Args:
            metadata: Metadata dictionary
        """
        import json

        self.metadata_json = json.dumps(metadata)

    def get_tags_list(self) -> list:
        """
        Get tags as list

        Returns:
            List of tags
        """
        if self.tags:
            return [tag.strip() for tag in self.tags.split(",") if tag.strip()]
        return []

    def set_tags_list(self, tags: list):
        """
        Set tags from list

        Args:
            tags: List of tags
        """
        self.tags = ",".join(str(tag).strip() for tag in tags)


class FullAuditModel(BaseModel, AuditMixin, MetadataMixin):
    """
    Full audit model with all mixins

    Includes:
    - Base functionality (UUID, timestamps, soft delete)
    - Audit fields (created_by, updated_by)
    - Metadata fields (metadata_json, tags)
    """

    __abstract__ = True

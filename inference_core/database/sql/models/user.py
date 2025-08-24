"""
User Model

Database model for user management and authentication.
"""

from typing import Optional

from sqlalchemy import Boolean, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from ..base import FullAuditModel


class User(FullAuditModel):
    """
    User model for authentication and user management

    Attributes:
        email: User email (unique)
        username: Username (unique)
        hashed_password: Hashed password
        first_name: User first name
        last_name: User last name
        is_active: Whether user is active
        is_superuser: Whether user has admin privileges
        is_verified: Whether email is verified
    """

    __tablename__ = "users"

    # Authentication fields
    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True, doc="User email address"
    )
    username: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True, doc="Username"
    )
    hashed_password: Mapped[str] = mapped_column(
        String(128), nullable=False, doc="Hashed password"
    )

    # Profile fields
    first_name: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, doc="First name"
    )
    last_name: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, doc="Last name"
    )

    # Status fields
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False, index=True, doc="Whether user is active"
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Whether user has admin privileges",
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Whether email is verified",
    )

    # Add composite indexes for common queries
    __table_args__ = (
        Index("ix_users_email_active", "email", "is_active"),
        Index("ix_users_username_active", "username", "is_active"),
    )

    @property
    def full_name(self) -> str:
        """Get full name"""
        parts = []
        if self.first_name:
            parts.append(self.first_name)
        if self.last_name:
            parts.append(self.last_name)
        return " ".join(parts) or self.username

    def to_dict(
        self, exclude: Optional[set] = None, include_relationships: bool = False
    ) -> dict:
        """Convert to dict, excluding sensitive fields by default"""
        default_exclude = {"hashed_password"}
        if exclude:
            default_exclude.update(exclude)
        return super().to_dict(
            exclude=default_exclude, include_relationships=include_relationships
        )

    def is_admin(self) -> bool:
        """Check if user is admin"""
        return self.is_superuser and self.is_active

    def can_access(self) -> bool:
        """Check if user can access the system"""
        return self.is_active and not self.is_deleted

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"

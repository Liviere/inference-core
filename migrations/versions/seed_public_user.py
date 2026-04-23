"""Seed public (anonymous) user for LLM_API_ACCESS_MODE=public

Revision ID: seed_public_user
Revises: e70bd305a912
Create Date: 2026-04-23 00:00:00.000000

WHY:
    Agent-instance endpoints require a real users.id row due to the FK on
    UserAgentInstance.user_id. Public-access deployments map every anonymous
    caller to this single seeded account. The stored password hash is NOT a
    valid bcrypt string, so SecurityManager.verify_password rejects any login
    attempt against this account (bcrypt.checkpw raises ValueError, which the
    manager catches and returns False).
"""

import uuid
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "seed_public_user"
down_revision: Union[str, Sequence[str], None] = "e70bd305a912"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Mirrors inference_core.core.public_user constants — duplicated here on
# purpose so the migration stays self-contained and does not import app code.
_PUBLIC_USER_UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_PUBLIC_USER_USERNAME = "public"
_PUBLIC_USER_EMAIL = "public@inference-core.local"
_PUBLIC_USER_FIRST_NAME = "Public"
_PUBLIC_USER_LAST_NAME = "User"
_LOCKED_PASSWORD_HASH = "!locked!public-user-no-login!"


def _public_user_id_for_bind(bind) -> str:
    """Return the UUID literal matching the current dialect's storage format.

    SQLite stores SQLAlchemy ``Uuid(as_uuid=True)`` columns as 32-char hex
    strings, while PostgreSQL/MySQL accept the canonical hyphenated form.
    """
    if bind.dialect.name == "sqlite":
        return _PUBLIC_USER_UUID.hex
    return str(_PUBLIC_USER_UUID)


def upgrade() -> None:
    """Insert the public user row if it is not already present.

    Uses a plain existence check instead of dialect-specific ON CONFLICT so
    the migration stays portable across SQLite, PostgreSQL, and MySQL.
    """
    bind = op.get_bind()
    public_user_id = _public_user_id_for_bind(bind)

    existing = bind.execute(
        sa.text("SELECT id FROM users WHERE id = :id"),
        {"id": public_user_id},
    ).first()
    if existing:
        return

    # Defensive: avoid colliding with a human user that happens to already use
    # this username/email. Skip seeding in that case and let the operator
    # resolve the conflict manually.
    clash = bind.execute(
        sa.text("SELECT id FROM users WHERE username = :u OR email = :e"),
        {"u": _PUBLIC_USER_USERNAME, "e": _PUBLIC_USER_EMAIL},
    ).first()
    if clash:
        return

    bind.execute(
        sa.text(
            """
            INSERT INTO users (
                id, email, username, hashed_password,
                first_name, last_name,
                is_active, is_superuser, is_verified,
                created_at, updated_at
            )
            VALUES (
                :id, :email, :username, :hashed_password,
                :first_name, :last_name,
                :is_active, :is_superuser, :is_verified,
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
            )
            """
        ),
        {
            "id": public_user_id,
            "email": _PUBLIC_USER_EMAIL,
            "username": _PUBLIC_USER_USERNAME,
            "hashed_password": _LOCKED_PASSWORD_HASH,
            "first_name": _PUBLIC_USER_FIRST_NAME,
            "last_name": _PUBLIC_USER_LAST_NAME,
            "is_active": True,
            "is_superuser": False,
            "is_verified": True,
        },
    )


def downgrade() -> None:
    """Remove the seeded public user.

    ON DELETE CASCADE on user_agent_instances.user_id means rolling this back
    will drop every instance owned by the public user. That is acceptable —
    those rows are by definition shared anonymous state, not user data.
    """
    bind = op.get_bind()
    bind.execute(
        sa.text("DELETE FROM users WHERE id = :id"),
        {"id": _public_user_id_for_bind(bind)},
    )

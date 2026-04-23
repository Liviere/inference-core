"""Normalize public user UUID storage for SQLite.

Revision ID: fix_public_user_uuid_storage
Revises: seed_public_user
Create Date: 2026-04-23 17:05:00.000000

WHY:
    ``seed_public_user`` originally inserted the public user UUID as a
    hyphenated string. SQLite stores ``Uuid(as_uuid=True)`` as ``CHAR(32)``,
    so ORM lookups bind ``uuid.hex`` and miss the seeded row, breaking public
    mode auth with a misleading 503.
"""

import uuid
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fix_public_user_uuid_storage"
down_revision: Union[str, Sequence[str], None] = "seed_public_user"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_PUBLIC_USER_UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_LEGACY_PUBLIC_USER_ID = str(_PUBLIC_USER_UUID)
_SQLITE_PUBLIC_USER_ID = _PUBLIC_USER_UUID.hex


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _iter_user_fk_columns(bind):
    inspector = sa.inspect(bind)
    for table_name in inspector.get_table_names():
        for fk in inspector.get_foreign_keys(table_name):
            constrained_columns = fk.get("constrained_columns") or []
            referred_columns = fk.get("referred_columns") or []
            if (
                fk.get("referred_table") == "users"
                and referred_columns == ["id"]
                and len(constrained_columns) == 1
            ):
                yield table_name, constrained_columns[0]


def upgrade() -> None:
    """Rewrite the legacy hyphenated UUID to SQLite's 32-char storage format."""
    bind = op.get_bind()

    if bind.dialect.name != "sqlite":
        return

    has_correct = bind.execute(
        sa.text("SELECT 1 FROM users WHERE id = :id"),
        {"id": _SQLITE_PUBLIC_USER_ID},
    ).first()
    if has_correct:
        return

    has_legacy = bind.execute(
        sa.text("SELECT 1 FROM users WHERE id = :id"),
        {"id": _LEGACY_PUBLIC_USER_ID},
    ).first()
    if not has_legacy:
        return

    # Keep referential integrity if any SQLite rows were inserted manually
    # against the legacy public-user id before this fix landed.
    bind.execute(sa.text("PRAGMA defer_foreign_keys = ON"))

    for table_name, column_name in _iter_user_fk_columns(bind):
        bind.execute(
            sa.text(
                f"UPDATE {_quote(table_name)} "
                f"SET {_quote(column_name)} = :new_id "
                f"WHERE {_quote(column_name)} = :old_id"
            ),
            {"new_id": _SQLITE_PUBLIC_USER_ID, "old_id": _LEGACY_PUBLIC_USER_ID},
        )

    bind.execute(
        sa.text("UPDATE users SET id = :new_id WHERE id = :old_id"),
        {"new_id": _SQLITE_PUBLIC_USER_ID, "old_id": _LEGACY_PUBLIC_USER_ID},
    )


def downgrade() -> None:
    """No-op: keep the corrected UUID format instead of restoring bad data."""
    return None
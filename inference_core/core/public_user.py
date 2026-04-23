"""
Public (anonymous) user support for ``LLM_API_ACCESS_MODE=public``.

WHY this module exists:
    Agent-instance endpoints (/api/v1/agent-instances/*) are modelled around a
    per-user ownership scheme (UserAgentInstance.user_id → users.id FK). When
    the deployment is configured as ``LLM_API_ACCESS_MODE=public`` we still
    need a valid ``user_id`` to satisfy the FK and to drive the existing
    ownership filtering in services — all anonymous callers share a single
    seeded "public user" row.

Guarantees:
    * The public user UUID is a compile-time constant, identical across
      environments so migrations can seed it idempotently.
    * The seeded password hash is intentionally NOT a valid bcrypt hash, so
      ``SecurityManager.verify_password`` rejects every login attempt against
      this account (``bcrypt.checkpw`` returns False for non-bcrypt strings
      and the manager already swallows the raised ``ValueError``).
    * This fallback is consulted ONLY when ``llm_api_access_mode == "public"``
      — other modes keep the strict auth contract.
"""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from inference_core.database.sql.models.user import User

# Stable UUID for the shared anonymous account. Do not change — it is
# referenced by the Alembic seed migration and by runtime lookups.
PUBLIC_USER_ID: uuid.UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")

PUBLIC_USER_USERNAME: str = "public"
PUBLIC_USER_EMAIL: str = "public@inference-core.local"
PUBLIC_USER_FIRST_NAME: str = "Public"
PUBLIC_USER_LAST_NAME: str = "User"

# Sentinel that will never validate against bcrypt.checkpw. Kept short and
# obvious so log lines / DB inspections make the intent clear.
PUBLIC_USER_LOCKED_PASSWORD_HASH: str = "!locked!public-user-no-login!"


async def get_public_user_dict(db: AsyncSession) -> Optional[dict]:
    """Return the seeded public user payload, or None if it is missing.

    Missing row means the seed migration has not been applied yet — callers
    should treat that as an auth failure (401), not auto-create the row, to
    keep DB mutations out of the request path.
    """
    result = await db.execute(select(User).where(User.id == PUBLIC_USER_ID))
    user: Optional[User] = result.scalar_one_or_none()
    if not user:
        return None
    return {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "is_verified": user.is_verified,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
        "is_public_anonymous": True,
    }

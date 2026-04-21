"""LangGraph Agent Server custom authentication handler.

WHY: The Agent Server is exposed publicly (or to a frontend that talks to it
directly via ``@langchain/react`` ``useStream``).  We must validate the same
JWT access token that protects the FastAPI backend, so the user identity
flowing into ``ctx.user`` matches the identity the application uses for
authorization, cost tracking, memory namespace isolation, and resource
ownership.

This module is referenced by ``langgraph.json`` → ``auth.path`` and is
imported once by the LangGraph runtime at boot time.

Behaviour:
  * In production (``LANGGRAPH_AUTH_DISABLED`` unset) — JWT is required.
  * For local ``langgraph dev`` use without a token, set the env var
    ``LANGGRAPH_AUTH_DISABLED=true`` to fall back to a synthetic dev user.
    The Studio UI uses this when no API key is configured.

Authorization (``@auth.on``) is handled by a single global owner-scoping
handler — every thread/assistant/cron is tagged with ``owner=<user_id>`` so
users can only read/modify their own resources.  This mirrors the
single-owner pattern from the LangGraph auth docs.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langgraph_sdk import Auth

logger = logging.getLogger(__name__)

auth = Auth()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _auth_disabled() -> bool:
    """Whether to bypass JWT validation (local dev only)."""
    return os.environ.get("LANGGRAPH_AUTH_DISABLED", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _extract_bearer(headers: dict[bytes, bytes]) -> str | None:
    """Return the bearer token from request headers, or None.

    Accepts both ``authorization`` and ``Authorization``; the LangGraph
    runtime normalises header keys to lowercase bytes.
    """
    raw = headers.get(b"authorization") or headers.get(b"Authorization")
    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("latin-1", errors="ignore")
    parts = raw.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


@auth.authenticate
async def authenticate(headers: dict[bytes, bytes]) -> Auth.types.MinimalUserDict:
    """Validate the inference-core JWT and return user identity.

    Token validation is delegated to ``SecurityManager.verify_token`` so the
    Agent Server uses identical signing keys, algorithms, and rules as the
    FastAPI backend.  We accept access tokens only — refresh tokens are
    rejected at the verifier level.
    """
    if _auth_disabled():
        # WHY: Allow `langgraph dev` without a token for local development.
        # MUST stay off in production deployments.
        logger.warning(
            "LangGraph auth disabled — using synthetic dev user. "
            "Unset LANGGRAPH_AUTH_DISABLED in production."
        )
        return {
            "identity": "dev-user",
            "is_authenticated": True,
            "permissions": ["dev"],
        }

    token = _extract_bearer(headers)
    if not token:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization header (expected: 'Bearer <token>')",
        )

    # Lazy import — keeps module importable in stripped-down test envs and
    # avoids triggering settings/DB initialisation at module load time.
    from inference_core.core.security import security_manager

    token_data = security_manager.verify_token(token)
    if token_data is None:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Invalid or expired access token",
        )

    return {
        "identity": str(token_data.user_id),
        "is_authenticated": True,
        "permissions": [],
    }


# --------------------------------------------------------------------------
# Authorization — single-owner resource scoping
# --------------------------------------------------------------------------


@auth.on
async def owner_only(
    ctx: Auth.types.AuthContext,
    value: Any,
) -> dict[str, str]:
    """Tag every resource with ``owner=<user_id>`` and filter on the same.

    This is the single-owner pattern from the LangGraph docs — the simplest
    correct authorization for a multi-tenant app where each user owns their
    own threads, assistants, runs, and crons.
    """
    if not isinstance(value, dict):
        # Read/list payloads can be other shapes — just return the filter.
        return {"owner": ctx.user.identity}

    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity
    return {"owner": ctx.user.identity}

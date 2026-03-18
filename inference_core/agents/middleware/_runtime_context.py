"""
Per-request runtime context for middleware running on the Agent Server.

WHY: On the Agent Server, middleware instances are shared across concurrent
graph invocations (the compiled graph is a singleton).  Instance variables
like ``self.user_id`` would cause race conditions between requests.

This module provides ``contextvars``-based storage that is safe for concurrent
async tasks.  Node-style hooks (``before_agent``, ``after_model``) populate
the context from ``runtime.configurable``, and wrap-style hooks
(``wrap_model_call``) read from it — all within the same async task scope.

Local execution (AgentService) still sets instance attributes directly,
so the context vars act as a **fallback** when instance attrs are None.
"""

import contextvars
import uuid
from typing import Any, Optional

# Per-request identifiers (populated from runtime.configurable on Agent Server)
_user_id: contextvars.ContextVar[Optional[uuid.UUID]] = contextvars.ContextVar(
    "mw_user_id", default=None
)
_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "mw_session_id", default=None
)
_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "mw_request_id", default=None
)
_instance_id: contextvars.ContextVar[Optional[uuid.UUID]] = contextvars.ContextVar(
    "mw_instance_id", default=None
)
_instance_name: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "mw_instance_name", default=None
)


def populate_from_configurable(configurable: dict[str, Any]) -> None:
    """Extract middleware-relevant fields from ``runtime.configurable`` and
    store them in task-local context vars.

    Expected keys (set by ``agent_server_client``):
        user_id, session_id, request_id, instance_id, instance_name
    """
    raw_uid = configurable.get("user_id")
    if raw_uid is not None:
        _user_id.set(
            uuid.UUID(str(raw_uid)) if not isinstance(raw_uid, uuid.UUID) else raw_uid
        )

    if (sid := configurable.get("session_id")) is not None:
        _session_id.set(str(sid))

    if (rid := configurable.get("request_id")) is not None:
        _request_id.set(str(rid))

    raw_iid = configurable.get("instance_id")
    if raw_iid is not None:
        _instance_id.set(
            uuid.UUID(str(raw_iid)) if not isinstance(raw_iid, uuid.UUID) else raw_iid
        )

    if (iname := configurable.get("instance_name")) is not None:
        _instance_name.set(str(iname))


def get_user_id() -> Optional[uuid.UUID]:
    return _user_id.get()


def get_session_id() -> Optional[str]:
    return _session_id.get()


def get_request_id() -> Optional[str]:
    return _request_id.get()


def get_instance_id() -> Optional[uuid.UUID]:
    return _instance_id.get()


def get_instance_name() -> Optional[str]:
    return _instance_name.get()


def clear() -> None:
    """Reset all context vars (useful for testing)."""
    for var in (_user_id, _session_id, _request_id, _instance_id, _instance_name):
        var.set(None)

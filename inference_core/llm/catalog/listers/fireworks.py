"""Lister for Fireworks AI.

Models belong to accounts, and their ids say so:
``accounts/<account>/models/<model>``. The listing is that of the public
``fireworks`` account; a model of any other account is found by looking its id
up.
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from ..http import MAX_PAGES, CatalogResponseError, get_json, get_json_or_none
from ..types import DiscoveredModel, ModelKind, ModelLifecycle
from ._common import (
    bearer_headers,
    clean_model_id,
    clean_text,
    compact,
    dicts,
    from_iso,
    iso,
    optional_object,
    positive_int,
    require_list,
)

FIREWORKS_BASE_URL = "https://api.fireworks.ai/v1"
PUBLIC_ACCOUNT = "fireworks"
PAGE_SIZE = 200

_MODEL_ID = re.compile(r"accounts/[A-Za-z0-9._-]+/models/[A-Za-z0-9._-]+")


def _kind(item: Dict[str, Any]) -> ModelKind:
    kind = item.get("kind")
    if kind == "EMBEDDING_MODEL":
        return ModelKind.EMBEDDING
    if isinstance(kind, str) and kind.startswith("FLUMINA"):
        return ModelKind.IMAGE
    if isinstance(item.get("conversationConfig"), dict):
        return ModelKind.CHAT
    return ModelKind.UNKNOWN


def _date(value: Any) -> Optional[datetime]:
    """A ``{year, month, day}`` object as the start of that day, UTC."""
    if not isinstance(value, dict):
        return None
    try:
        return datetime(
            int(value["year"]),
            int(value["month"]),
            int(value["day"]),
            tzinfo=timezone.utc,
        )
    except KeyError, TypeError, ValueError:
        return None


def _flag(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _parse(item: Dict[str, Any]) -> Optional[DiscoveredModel]:
    model_id = clean_model_id(item.get("name"))
    if not model_id:
        return None
    retires_at = _date(item.get("deprecationDate"))
    return DiscoveredModel(
        provider="fireworks",
        model_id=model_id,
        kind=_kind(item),
        lifecycle=ModelLifecycle.DEPRECATED if retires_at else ModelLifecycle.ACTIVE,
        retires_at=retires_at,
        attributes=compact(
            {
                "display_name": clean_text(item.get("displayName")),
                "created_at": iso(from_iso(item.get("createTime"))),
                "context_window": positive_int(item.get("contextLength")),
                "capabilities": compact(
                    {
                        "vision": _flag(item.get("supportsImageInput")),
                        "tools": _flag(item.get("supportsTools")),
                    }
                ),
            }
        ),
    )


async def list_models(
    runtime: Dict[str, Any], client: httpx.AsyncClient
) -> List[DiscoveredModel]:
    models: List[DiscoveredModel] = []
    params: Dict[str, Any] = {"pageSize": PAGE_SIZE}
    for _ in range(MAX_PAGES):
        payload = await get_json(
            client,
            f"{FIREWORKS_BASE_URL}/accounts/{PUBLIC_ACCOUNT}/models",
            headers=bearer_headers(runtime),
            params=params,
        )
        items = require_list(payload, "models", may_be_absent=True)
        models.extend(model for model in map(_parse, dicts(items)) if model)
        token = payload.get("nextPageToken")
        if token is None or token == "":
            return models
        if not isinstance(token, str):
            raise CatalogResponseError("unreadable page token", status_code=200)
        params = {"pageSize": PAGE_SIZE, "pageToken": token}
    raise CatalogResponseError("listing does not end", status_code=200)


async def get_model(
    runtime: Dict[str, Any], client: httpx.AsyncClient, model_id: str
) -> Optional[DiscoveredModel]:
    if not _MODEL_ID.fullmatch(model_id):
        # Not a Fireworks model id, so there is no such model to ask about.
        return None
    payload = optional_object(
        await get_json_or_none(
            client, f"{FIREWORKS_BASE_URL}/{model_id}", headers=bearer_headers(runtime)
        )
    )
    return _parse(payload) if payload is not None else None

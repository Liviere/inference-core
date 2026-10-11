"""Lister for the Google Gemini API.

The listing has token limits and the methods a model supports, and nothing
about a model's life: no release date, no shutdown date. The API key goes in a
header, never in the query string, where it would end up in request logs.
"""

from typing import Any, Dict, List, Optional

import httpx

from ..http import MAX_PAGES, CatalogResponseError, get_json, get_json_or_none
from ..types import DiscoveredModel, ModelKind
from ._common import (
    clean_model_id,
    clean_text,
    compact,
    dicts,
    optional_object,
    path_segment,
    positive_int,
    require_list,
)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
PAGE_SIZE = 1000
_NAME_PREFIX = "models/"


def _headers(runtime: Dict[str, Any]) -> Dict[str, str]:
    return {"x-goog-api-key": str(runtime.get("api_key") or "")}


def _kind(methods: Any) -> ModelKind:
    if not isinstance(methods, list):
        return ModelKind.UNKNOWN
    if "generateContent" in methods:
        return ModelKind.CHAT
    if any(isinstance(m, str) and m.startswith("embed") for m in methods):
        return ModelKind.EMBEDDING
    return ModelKind.OTHER


def _parse(item: Dict[str, Any]) -> Optional[DiscoveredModel]:
    name = item.get("name")
    if isinstance(name, str) and name.startswith(_NAME_PREFIX):
        name = name[len(_NAME_PREFIX) :]
    model_id = clean_model_id(name)
    if not model_id:
        return None
    thinking = item.get("thinking")
    return DiscoveredModel(
        provider="gemini",
        model_id=model_id,
        kind=_kind(item.get("supportedGenerationMethods")),
        attributes=compact(
            {
                "display_name": clean_text(item.get("displayName")),
                "context_window": positive_int(item.get("inputTokenLimit")),
                "max_output_tokens": positive_int(item.get("outputTokenLimit")),
                "capabilities": compact(
                    {"reasoning": thinking if isinstance(thinking, bool) else None}
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
            f"{GEMINI_BASE_URL}/models",
            headers=_headers(runtime),
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
    payload = optional_object(
        await get_json_or_none(
            client,
            f"{GEMINI_BASE_URL}/models/{path_segment(model_id)}",
            headers=_headers(runtime),
        )
    )
    return _parse(payload) if payload is not None else None

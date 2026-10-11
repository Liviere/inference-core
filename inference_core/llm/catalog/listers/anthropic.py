"""Lister for Anthropic (provider ``claude``).

The listing says where each model is in its life and when it retires. It
leaves retired models out, and a configured alias is not in it either, so a
single model is looked up by id: that request resolves an alias and still
answers for a model that has been retired.
"""

from typing import Any, Dict, List, Optional

import httpx

from ..http import MAX_PAGES, CatalogResponseError, get_json, get_json_or_none
from ..types import DiscoveredModel, ModelKind, ModelLifecycle
from ._common import (
    clean_model_id,
    clean_text,
    compact,
    dicts,
    from_iso,
    iso,
    optional_object,
    path_segment,
    positive_int,
    require_list,
)

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
PAGE_SIZE = 1000

_LIFECYCLES = {
    "active": ModelLifecycle.ACTIVE,
    "deprecated": ModelLifecycle.DEPRECATED,
    "retired": ModelLifecycle.RETIRED,
}


def _headers(runtime: Dict[str, Any]) -> Dict[str, str]:
    return {
        "x-api-key": str(runtime.get("api_key") or ""),
        "anthropic-version": ANTHROPIC_VERSION,
    }


def _supported(capabilities: Dict[str, Any], key: str) -> Optional[bool]:
    entry = capabilities.get(key)
    supported = entry.get("supported") if isinstance(entry, dict) else None
    return supported if isinstance(supported, bool) else None


def _parse(item: Dict[str, Any]) -> Optional[DiscoveredModel]:
    model_id = clean_model_id(item.get("id"))
    if not model_id:
        return None
    raw_capabilities = item.get("capabilities")
    capabilities = raw_capabilities if isinstance(raw_capabilities, dict) else {}
    return DiscoveredModel(
        provider="claude",
        model_id=model_id,
        kind=ModelKind.CHAT,
        lifecycle=_LIFECYCLES.get(item.get("lifecycle"), ModelLifecycle.UNKNOWN),
        deprecated_at=from_iso(item.get("deprecated_at")),
        retires_at=from_iso(item.get("retires_at")),
        attributes=compact(
            {
                "display_name": clean_text(item.get("display_name")),
                "created_at": iso(from_iso(item.get("created_at"))),
                "context_window": positive_int(item.get("max_input_tokens")),
                "max_output_tokens": positive_int(item.get("max_tokens")),
                "capabilities": compact(
                    {
                        "vision": _supported(capabilities, "image_input"),
                        "reasoning": _supported(capabilities, "thinking"),
                    }
                ),
            }
        ),
    )


async def list_models(
    runtime: Dict[str, Any], client: httpx.AsyncClient
) -> List[DiscoveredModel]:
    models: List[DiscoveredModel] = []
    params: Dict[str, Any] = {"limit": PAGE_SIZE}
    for _ in range(MAX_PAGES):
        payload = await get_json(
            client,
            f"{ANTHROPIC_BASE_URL}/models",
            headers=_headers(runtime),
            params=params,
        )
        parsed = map(_parse, dicts(require_list(payload, "data")))
        models.extend(model for model in parsed if model)
        if not payload.get("has_more"):
            return models
        last_id = payload.get("last_id")
        if not isinstance(last_id, str) or not last_id:
            raise CatalogResponseError("more pages but no cursor", status_code=200)
        params = {"limit": PAGE_SIZE, "after_id": last_id}
    raise CatalogResponseError("listing does not end", status_code=200)


async def get_model(
    runtime: Dict[str, Any], client: httpx.AsyncClient, model_id: str
) -> Optional[DiscoveredModel]:
    payload = optional_object(
        await get_json_or_none(
            client,
            f"{ANTHROPIC_BASE_URL}/models/{path_segment(model_id)}",
            headers=_headers(runtime),
        )
    )
    return _parse(payload) if payload is not None else None

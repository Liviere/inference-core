"""Lister for DeepInfra.

DeepInfra's own listing is used, not its OpenAI-compatible one: only this one
says that a model is deprecated and what replaces it, and it keeps deprecated
models in the list, so a model does not look gone while it still answers.
"""

from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from ..http import get_json, get_json_or_none
from ..types import DiscoveredModel, ModelKind, ModelLifecycle
from ._common import (
    bearer_headers,
    clean_model_id,
    compact,
    dicts,
    from_iso,
    from_unix,
    iso,
    optional_object,
    positive_int,
    price,
    require_list,
)

DEEPINFRA_API_URL = "https://api.deepinfra.com"

# Prices are USD cents per token.
_CENTS_PER_TOKEN_TO_USD_PER_MILLION = 10_000

_KINDS = {
    "text-generation": ModelKind.CHAT,
    "embeddings": ModelKind.EMBEDDING,
    "text-to-image": ModelKind.IMAGE,
    "text-to-video": ModelKind.IMAGE,
    "text-to-speech": ModelKind.AUDIO,
    "automatic-speech-recognition": ModelKind.AUDIO,
}


def _pricing(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict) or raw.get("type") != "tokens":
        return {}
    return compact(
        {
            "input": price(
                raw.get("cents_per_input_token"), _CENTS_PER_TOKEN_TO_USD_PER_MILLION
            ),
            "output": price(
                raw.get("cents_per_output_token"), _CENTS_PER_TOKEN_TO_USD_PER_MILLION
            ),
        }
    )


def _parse(item: Dict[str, Any]) -> Optional[DiscoveredModel]:
    model_id = clean_model_id(item.get("model_name"))
    if not model_id:
        return None
    kind = item.get("reported_type") or item.get("type")
    # ``deprecated`` is the time the model was deprecated, or null.
    deprecated = item.get("deprecated")
    return DiscoveredModel(
        provider="deepinfra",
        model_id=model_id,
        kind=_KINDS.get(kind, ModelKind.OTHER) if kind else ModelKind.UNKNOWN,
        lifecycle=ModelLifecycle.DEPRECATED if deprecated else ModelLifecycle.ACTIVE,
        deprecated_at=from_unix(deprecated),
        replacement=clean_model_id(item.get("replaced_by")),
        attributes=compact(
            {
                "created_at": iso(from_iso(item.get("create_ts"))),
                "context_window": positive_int(item.get("max_tokens")),
                "max_output_tokens": positive_int(item.get("max_output_tokens")),
                "pricing": _pricing(item.get("pricing")),
            }
        ),
    )


async def list_models(
    runtime: Dict[str, Any], client: httpx.AsyncClient
) -> List[DiscoveredModel]:
    payload = await get_json(
        client, f"{DEEPINFRA_API_URL}/models/list", headers=bearer_headers(runtime)
    )
    models = map(_parse, dicts(require_list(payload, None)))
    return [model for model in models if model]


async def get_model(
    runtime: Dict[str, Any], client: httpx.AsyncClient, model_id: str
) -> Optional[DiscoveredModel]:
    # Model names are ``owner/name``; the slash is part of the path.
    payload = optional_object(
        await get_json_or_none(
            client,
            f"{DEEPINFRA_API_URL}/models/{quote(model_id, safe='/')}",
            headers=bearer_headers(runtime),
        )
    )
    return _parse(payload) if payload is not None else None

"""Listers for providers with an OpenAI-shaped ``GET /models``.

The shape is shared, what it carries is not: OpenAI adds a shutdown date,
Mistral aliases, capabilities and a deprecation date, xAI aliases, modalities
and prices. A server that is merely compatible promises an id and no more.
"""

from typing import Any, Callable, Dict, List, Optional

import httpx

from ..http import get_json, get_json_or_none
from ..types import DiscoveredModel, ModelKind, ModelLifecycle
from ._common import (
    base_url,
    bearer_headers,
    clean_aliases,
    clean_model_id,
    clean_text,
    compact,
    dicts,
    from_iso,
    from_unix,
    iso,
    optional_object,
    path_segment,
    positive_int,
    price,
    require_list,
)

OPENAI_BASE_URL = "https://api.openai.com/v1"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
XAI_BASE_URL = "https://api.x.ai/v1"

# xAI prices are USD cents per 100 million tokens.
_XAI_PRICE_TO_USD_PER_MILLION = 1 / 10_000

Parser = Callable[[Dict[str, Any]], Optional[DiscoveredModel]]


def _openai_kind(model_id: str) -> ModelKind:
    """OpenAI's listing has no kind, so the id is all there is to go by."""
    name = model_id.lower()
    if "embedding" in name:
        return ModelKind.EMBEDDING
    if name.startswith(("dall-e", "gpt-image", "sora")) or "image" in name:
        return ModelKind.IMAGE
    if any(
        part in name for part in ("whisper", "tts", "audio", "transcribe", "realtime")
    ):
        return ModelKind.AUDIO
    if "moderation" in name:
        return ModelKind.OTHER
    return ModelKind.CHAT


def _parse_openai(item: Dict[str, Any]) -> Optional[DiscoveredModel]:
    model_id = clean_model_id(item.get("id"))
    if not model_id:
        return None
    shutdown = from_iso(item.get("shutdown_date"))
    return DiscoveredModel(
        provider="openai",
        model_id=model_id,
        kind=_openai_kind(model_id),
        lifecycle=ModelLifecycle.DEPRECATED if shutdown else ModelLifecycle.ACTIVE,
        retires_at=shutdown,
        attributes=compact({"created_at": iso(from_unix(item.get("created")))}),
    )


def _parse_generic(item: Dict[str, Any]) -> Optional[DiscoveredModel]:
    model_id = clean_model_id(item.get("id"))
    if not model_id:
        return None
    return DiscoveredModel(provider="custom_openai_compatible", model_id=model_id)


def _mistral_kind(model_id: str, capabilities: Dict[str, Any]) -> ModelKind:
    if capabilities.get("completion_chat") is True:
        return ModelKind.CHAT
    if any(
        capabilities.get(key) is True
        for key in (
            "audio_transcription",
            "audio_transcription_realtime",
            "audio_speech",
        )
    ):
        return ModelKind.AUDIO
    if "embed" in model_id.lower():
        return ModelKind.EMBEDDING
    return ModelKind.OTHER if capabilities else ModelKind.UNKNOWN


def _parse_mistral(item: Dict[str, Any]) -> Optional[DiscoveredModel]:
    model_id = clean_model_id(item.get("id"))
    if not model_id:
        return None
    raw_capabilities = item.get("capabilities")
    capabilities = raw_capabilities if isinstance(raw_capabilities, dict) else {}
    deprecation = from_iso(item.get("deprecation"))
    return DiscoveredModel(
        provider="mistral",
        model_id=model_id,
        kind=_mistral_kind(model_id, capabilities),
        lifecycle=ModelLifecycle.DEPRECATED if deprecation else ModelLifecycle.ACTIVE,
        deprecated_at=deprecation,
        replacement=clean_model_id(item.get("deprecation_replacement_model")),
        aliases=clean_aliases(item.get("aliases")),
        attributes=compact(
            {
                "display_name": clean_text(item.get("name")),
                "created_at": iso(from_unix(item.get("created"))),
                "context_window": positive_int(item.get("max_context_length")),
                "capabilities": compact(
                    {
                        "vision": _flag(capabilities.get("vision")),
                        "tools": _flag(capabilities.get("function_calling")),
                        "reasoning": _flag(capabilities.get("reasoning")),
                    }
                ),
            }
        ),
    )


def _parse_xai(item: Dict[str, Any]) -> Optional[DiscoveredModel]:
    model_id = clean_model_id(item.get("id"))
    if not model_id:
        return None
    inputs = item.get("input_modalities")
    return DiscoveredModel(
        provider="xai",
        model_id=model_id,
        kind=ModelKind.CHAT,
        aliases=clean_aliases(item.get("aliases")),
        attributes=compact(
            {
                "created_at": iso(from_unix(item.get("created"))),
                "capabilities": compact(
                    {"vision": "image" in inputs if isinstance(inputs, list) else None}
                ),
                "pricing": compact(
                    {
                        "input": price(
                            item.get("prompt_text_token_price"),
                            _XAI_PRICE_TO_USD_PER_MILLION,
                        ),
                        "output": price(
                            item.get("completion_text_token_price"),
                            _XAI_PRICE_TO_USD_PER_MILLION,
                        ),
                        "cache_read": price(
                            item.get("cached_prompt_text_token_price"),
                            _XAI_PRICE_TO_USD_PER_MILLION,
                        ),
                    }
                ),
            }
        ),
    )


def _flag(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _lister(default_base_url: str, path: str, list_key: str, parse: Parser):
    """``list_models`` and ``get_model`` for one OpenAI-shaped endpoint."""

    async def list_models(
        runtime: Dict[str, Any], client: httpx.AsyncClient
    ) -> List[DiscoveredModel]:
        payload = await get_json(
            client,
            f"{base_url(runtime, default_base_url)}{path}",
            headers=bearer_headers(runtime),
        )
        models = map(parse, dicts(require_list(payload, list_key)))
        return [model for model in models if model]

    async def get_model(
        runtime: Dict[str, Any], client: httpx.AsyncClient, model_id: str
    ) -> Optional[DiscoveredModel]:
        payload = optional_object(
            await get_json_or_none(
                client,
                f"{base_url(runtime, default_base_url)}{path}/{path_segment(model_id)}",
                headers=bearer_headers(runtime),
            )
        )
        return parse(payload) if payload is not None else None

    return list_models, get_model


list_openai_models, get_openai_model = _lister(
    OPENAI_BASE_URL, "/models", "data", _parse_openai
)
list_mistral_models, get_mistral_model = _lister(
    MISTRAL_BASE_URL, "/models", "data", _parse_mistral
)
list_xai_models, get_xai_model = _lister(
    XAI_BASE_URL, "/language-models", "models", _parse_xai
)
# A compatible server has no default address: without ``base_url`` there is
# nothing to ask, and the catalog skips the provider before it gets here.
list_compatible_models, get_compatible_model = _lister(
    "", "/models", "data", _parse_generic
)

"""Model listers by provider name.

A lister knows how to ask one provider which models it offers. The built-in
providers have theirs registered when the catalog package is imported. An
application that registered a provider of its own with
``register_chat_model_provider`` can give it a lister here; a provider without
one is simply not part of the catalog.
"""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from .types import DiscoveredModel

ListModels = Callable[
    [Dict[str, Any], httpx.AsyncClient], Awaitable[List[DiscoveredModel]]
]
"""``list_models(runtime, client)``: every model the provider lists.

``runtime`` is ``LLMConfig.get_provider_runtime_config(provider)``: the
provider's YAML entry with ``api_key`` resolved. The whole listing or an
exception: a lister never returns part of a listing.
"""

GetModel = Callable[
    [Dict[str, Any], httpx.AsyncClient, str], Awaitable[Optional[DiscoveredModel]]
]
"""``get_model(runtime, client, model_id)``: one model, or ``None`` when the
provider says there is no such model. This is how a configured alias that the
listing does not show is told apart from a model that is gone."""


@dataclass(frozen=True)
class ModelLister:
    """How one provider's models are listed."""

    provider: str
    list_models: ListModels
    get_model: Optional[GetModel] = None


_REGISTRY: Dict[str, ModelLister] = {}


def register_model_lister(
    provider: str,
    list_models: ListModels,
    *,
    get_model: Optional[GetModel] = None,
) -> None:
    """Make ``list_models`` the way ``provider``'s models are listed.

    Registering the same provider again replaces the earlier lister, so a
    module that registers at import time stays safe to import twice.
    """
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("Provider name must be a non-empty string")
    if not callable(list_models):
        raise ValueError("list_models must be callable")
    if get_model is not None and not callable(get_model):
        raise ValueError("get_model must be callable")
    _REGISTRY[provider] = ModelLister(
        provider=provider, list_models=list_models, get_model=get_model
    )


def unregister_model_lister(provider: str) -> None:
    """Forget ``provider``'s lister (tests, mostly)."""
    _REGISTRY.pop(provider, None)


def get_model_lister(provider: str) -> Optional[ModelLister]:
    """Return ``provider``'s lister, or ``None`` when it has none."""
    return _REGISTRY.get(provider)


def listed_providers() -> List[str]:
    """Names of all providers that have a lister, sorted."""
    return sorted(_REGISTRY)

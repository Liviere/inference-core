"""Model catalog: what each provider offers, next to what is configured.

Providers publish the list of their models, and some say in it when a model
will be shut down. The catalog reads those lists, so that a new model or an
approaching retirement is seen here and not first in a failed request.

It is advisory. Which models can be used is decided by ``models:`` in the LLM
config, exactly as without it.
"""

from .http import CatalogError, CatalogHTTPError, CatalogResponseError
from .listers import anthropic, deepinfra, fireworks, gemini, openai_compatible
from .registry import (
    ModelLister,
    get_model_lister,
    listed_providers,
    register_model_lister,
    unregister_model_lister,
)
from .types import DiscoveredModel, ModelKind, ModelLifecycle


def _register_builtin_listers() -> None:
    register_model_lister(
        "openai",
        openai_compatible.list_openai_models,
        get_model=openai_compatible.get_openai_model,
    )
    register_model_lister(
        "custom_openai_compatible",
        openai_compatible.list_compatible_models,
        get_model=openai_compatible.get_compatible_model,
    )
    register_model_lister(
        "xai",
        openai_compatible.list_xai_models,
        get_model=openai_compatible.get_xai_model,
    )
    register_model_lister(
        "mistral",
        openai_compatible.list_mistral_models,
        get_model=openai_compatible.get_mistral_model,
    )
    register_model_lister(
        "claude", anthropic.list_models, get_model=anthropic.get_model
    )
    register_model_lister("gemini", gemini.list_models, get_model=gemini.get_model)
    register_model_lister(
        "deepinfra", deepinfra.list_models, get_model=deepinfra.get_model
    )
    register_model_lister(
        "fireworks", fireworks.list_models, get_model=fireworks.get_model
    )


_register_builtin_listers()

__all__ = [
    "CatalogError",
    "CatalogHTTPError",
    "CatalogResponseError",
    "DiscoveredModel",
    "ModelKind",
    "ModelLifecycle",
    "ModelLister",
    "get_model_lister",
    "listed_providers",
    "register_model_lister",
    "unregister_model_lister",
]

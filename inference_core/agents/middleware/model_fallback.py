"""Helpers for LangChain model fallback middleware.

Configured fallback models are stored as names. Runtime agents use these
helpers to build ``ModelFallbackMiddleware`` with models created through the
project's ``LLMModelFactory``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from langchain.agents.middleware import ModelFallbackMiddleware

logger = logging.getLogger(__name__)

_FALLBACK_KEYS = ("fallback", "fallback_models")
_MISSING = object()


def fallback_models_from_mapping(
    mapping: Mapping[str, Any] | None,
    *,
    default: Any = _MISSING,
) -> Any:
    """Return the configured fallback list from ``fallback`` or its alias."""
    if not mapping:
        return None if default is _MISSING else default

    for key in _FALLBACK_KEYS:
        if key in mapping:
            return mapping[key]

    return None if default is _MISSING else default


def canonicalize_fallback_overrides(
    overrides: Mapping[str, Any] | None,
    *,
    keep_alias: bool = True,
) -> dict[str, Any]:
    """Copy overrides and mirror fallback aliases to the canonical key."""
    result = dict(overrides or {})
    fallback_value = fallback_models_from_mapping(result, default=_MISSING)
    if fallback_value is _MISSING:
        return result

    result["fallback"] = fallback_value
    if keep_alias:
        result["fallback_models"] = fallback_value
    else:
        result.pop("fallback_models", None)
    return result


def normalize_fallback_model_names(
    fallback_models: Sequence[Any] | None,
    *,
    primary_model: str | None = None,
) -> list[str]:
    """Normalize fallback model names while preserving order."""
    if not fallback_models:
        return []

    raw_models = (
        [fallback_models] if isinstance(fallback_models, str) else fallback_models
    )
    seen: set[str] = set()
    names: list[str] = []
    primary = str(primary_model) if primary_model else None

    for raw_name in raw_models:
        if raw_name is None:
            continue
        name = str(raw_name).strip()
        if not name or name == primary or name in seen:
            continue
        seen.add(name)
        names.append(name)

    return names


def build_model_fallback_middleware(
    *,
    model_factory: Any | None,
    fallback_models: Sequence[Any] | None,
    primary_model: str | None = None,
    reasoning_output: bool = False,
    owner: str = "agent",
) -> ModelFallbackMiddleware | None:
    """Build ``ModelFallbackMiddleware`` from configured model names."""
    fallback_names = normalize_fallback_model_names(
        fallback_models,
        primary_model=primary_model,
    )
    if not fallback_names:
        return None

    fallback_model_objects: list[Any] = []
    configured_models = getattr(getattr(model_factory, "config", None), "models", None)

    for model_name in fallback_names:
        if configured_models is not None and model_name not in configured_models:
            logger.warning(
                "Skipping fallback model '%s' for %s: model is not configured",
                model_name,
                owner,
            )
            continue

        if model_factory is None:
            fallback_model_objects.append(model_name)
            continue

        try:
            model = model_factory.create_model(
                model_name,
                reasoning_output=reasoning_output,
            )
        except Exception:
            logger.exception(
                "Failed to create fallback model '%s' for %s",
                model_name,
                owner,
            )
            continue

        if model is None:
            logger.warning(
                "Skipping fallback model '%s' for %s: factory returned None",
                model_name,
                owner,
            )
            continue

        fallback_model_objects.append(model)

    if not fallback_model_objects:
        return None

    try:
        return ModelFallbackMiddleware(
            fallback_model_objects[0],
            *fallback_model_objects[1:],
        )
    except Exception:
        logger.exception("Failed to build model fallback middleware for %s", owner)
        return None

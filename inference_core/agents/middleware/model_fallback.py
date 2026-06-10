"""Helpers for LangChain model fallback middleware.

Configured fallback models are stored as names. Runtime agents use these
helpers to build ``ModelFallbackMiddleware`` with models created through the
project's ``LLMModelFactory``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from langchain.agents.middleware import ModelFallbackMiddleware
from langgraph.errors import GraphInterrupt

from inference_core.services._cancel import AgentCancelled

logger = logging.getLogger(__name__)

_FALLBACK_KEYS = ("fallback", "fallback_models")
_MISSING = object()


class CancelAwareModelFallbackMiddleware(ModelFallbackMiddleware):
    """``ModelFallbackMiddleware`` that never swallows cancellation/interrupts.

    WHY: The stock middleware catches bare ``Exception`` around every model
    attempt.  ``GraphInterrupt`` (raised by the stream cancel callback inside
    the LLM call, and by HITL flows) and ``AgentCancelled`` (raised by
    ``CostTrackingMiddleware``'s pre-model cancel check, which runs *inside*
    this wrapper because cost tracking is forced innermost) are both
    ``Exception`` subclasses — so a user cancellation was treated as a model
    failure and every configured fallback model was invoked in turn, each
    billing the full input prompt before being cancelled again.

    This subclass re-raises ``GraphInterrupt`` and ``AgentCancelled``
    immediately (primary and fallback attempts alike) and additionally
    checks ``cancel_check`` before starting each fallback attempt, so no
    new paid model call begins after cancellation was requested.

    ``cancel_check`` is populated by ``AgentService.set_cancel_check``,
    which assigns the callback to every middleware exposing the attribute.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cancel_check: Callable[[], bool] | None = None

    def _raise_if_cancelled(self) -> None:
        """Raise ``AgentCancelled`` when the cancel callback reports True.

        Best-effort: errors from the check itself are ignored so a broken
        callback can never break fallback handling.
        """
        if not self.cancel_check:
            return
        try:
            if self.cancel_check():
                raise AgentCancelled("Agent execution cancelled by user")
        except AgentCancelled:
            raise
        except Exception:
            pass

    def wrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """Try fallback models in sequence, propagating cancellation."""
        last_exception: Exception
        try:
            return handler(request)
        except (GraphInterrupt, AgentCancelled):
            raise
        except Exception as e:
            last_exception = e

        for fallback_model in self.models:
            self._raise_if_cancelled()
            try:
                return handler(request.override(model=fallback_model))
            except (GraphInterrupt, AgentCancelled):
                raise
            except Exception as e:
                last_exception = e
                continue

        raise last_exception

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """Async version of :meth:`wrap_model_call`."""
        last_exception: Exception
        try:
            return await handler(request)
        except (GraphInterrupt, AgentCancelled):
            raise
        except Exception as e:
            last_exception = e

        for fallback_model in self.models:
            self._raise_if_cancelled()
            try:
                return await handler(request.override(model=fallback_model))
            except (GraphInterrupt, AgentCancelled):
                raise
            except Exception as e:
                last_exception = e
                continue

        raise last_exception


def fallback_models_from_mapping(
    mapping: Mapping[str, Any] | None,
    *,
    default: Any = None,
) -> Any:
    """Return the configured fallback list from ``fallback`` or its alias."""
    if mapping is None:
        return default

    for key in _FALLBACK_KEYS:
        if key in mapping:
            return mapping[key]

    return default


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
        return CancelAwareModelFallbackMiddleware(
            fallback_model_objects[0],
            *fallback_model_objects[1:],
        )
    except Exception:
        logger.exception("Failed to build model fallback middleware for %s", owner)
        return None

"""Helpers for LangChain model fallback middleware.

Configured fallback models are stored as names. Runtime agents use these
helpers to build ``ModelFallbackMiddleware`` with models created through the
project's ``LLMModelFactory``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from langchain.agents.middleware import ModelFallbackMiddleware
from langgraph.errors import GraphInterrupt

from inference_core.llm.model_identity import get_model_identity
from inference_core.services._cancel import AgentCancelled

logger = logging.getLogger(__name__)

_FALLBACK_KEYS = ("fallback", "fallback_models")
_MISSING = object()


class ModelAnswerFailed(RuntimeError):
    """A model call that came back as an answer and is a failure.

    Raised in place of the provider's own error when every model answered
    that way, so the caller sees a failed call and not an empty reply.
    """


def failed_answer_reason(response: Any) -> str | None:
    """Why *response* is a failure delivered as an answer, or ``None``.

    A provider can end a call that went wrong without raising: the stream
    closes with ``finish_reason: "error"``. What has arrived by then is not
    an answer. It may be nothing at all, or the reasoning and the sentence
    that leads up to a tool call ("Everything is ready, here is the
    proposal:") with the call itself never sent, because a tool call's
    arguments come last and in one piece.

    Taken as an answer, such a message ends the agent's run where the model
    meant to act. An empty one also stays in the conversation, where a
    stricter provider refuses every later request because of it ("Assistant
    message must have either content or tool_calls").

    So a message whose finish reason is ``error`` is a failed call unless it
    carries a tool call the agent can run. Any other finish reason is the
    model's own: an empty message that ended with ``stop`` is its choice to
    say nothing, and one cut off at the output limit says ``length``.
    """
    inner = getattr(response, "model_response", response)
    result = getattr(inner, "result", None)
    messages = result if isinstance(result, list) else [inner]

    for message in reversed(messages):
        if getattr(message, "type", None) != "ai":
            continue
        metadata = getattr(message, "response_metadata", None)
        reason = metadata.get("finish_reason") if isinstance(metadata, dict) else None
        if str(reason or "").strip().lower() != "error":
            return None
        if getattr(message, "tool_calls", None):
            return None
        return "the model's answer ended with finish_reason 'error' before it was complete"
    return None


# A provider's error type is a word like ``invalid_request_error``; anything
# else may be text from the request and is left out.
_ERROR_TYPE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")


def _model_label(model: Any) -> str:
    """The name a model is known by, for a log line."""
    if model is None:
        return "the primary model"
    identity = get_model_identity(model)
    if identity is not None and identity.name:
        return identity.name
    for attr in ("model_name", "model", "model_id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


def _provider_error_type(exc: BaseException) -> str | None:
    """The provider's own name for the error (``invalid_request_error``), if any."""
    candidates: list[Any] = []
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        error = body.get("error", body)
        if isinstance(error, Mapping):
            candidates += [error.get("type"), error.get("code")]
    candidates += [getattr(exc, "type", None), getattr(exc, "code", None)]
    for value in candidates:
        if isinstance(value, str) and _ERROR_TYPE.fullmatch(value):
            return value
    return None


def describe_failure(exc: BaseException) -> str:
    """What a failed model call was, without what it said.

    The exception's class, the HTTP status and the provider's error type.
    Never the message: a provider's error can quote the request it refused.
    """
    parts = [type(exc).__name__]
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        parts.append(f"HTTP {status}")
    error_type = _provider_error_type(exc)
    if error_type:
        parts.append(error_type)
    return ", ".join(parts)


def _log_fallback(failed_model: Any, exc: BaseException, next_model: Any) -> None:
    """Say that a model failed and which one takes over.

    Without this a model that never works (a wrong key, a missing model)
    looks healthy in the logs: every call is answered by a fallback, and
    every call pays for the failed attempt first.
    """
    logger.warning(
        "Model %s failed (%s); falling back to %s",
        _model_label(failed_model),
        describe_failure(exc),
        _model_label(next_model),
    )


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

    It also takes a failure that a provider delivered as an answer for what
    it is (:func:`failed_answer_reason`) and tries the next model, where the
    stock middleware would hand the empty message on as the model's reply.

    Each switch to the next model is logged as a warning, with the failed
    call's class, HTTP status and the provider's error type, never its text
    (:func:`describe_failure`). Which model answered in the end is not
    changed.

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
            response = handler(request)
        except GraphInterrupt, AgentCancelled:
            raise
        except Exception as e:
            last_exception = e
        else:
            failure = failed_answer_reason(response)
            if failure is None:
                return response
            last_exception = ModelAnswerFailed(failure)

        failed_model = request.model
        for fallback_model in self.models:
            self._raise_if_cancelled()
            _log_fallback(failed_model, last_exception, fallback_model)
            failed_model = fallback_model
            try:
                response = handler(request.override(model=fallback_model))
            except GraphInterrupt, AgentCancelled:
                raise
            except Exception as e:
                last_exception = e
                continue
            failure = failed_answer_reason(response)
            if failure is None:
                return response
            last_exception = ModelAnswerFailed(failure)

        raise last_exception

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """Async version of :meth:`wrap_model_call`."""
        last_exception: Exception
        try:
            response = await handler(request)
        except GraphInterrupt, AgentCancelled:
            raise
        except Exception as e:
            last_exception = e
        else:
            failure = failed_answer_reason(response)
            if failure is None:
                return response
            last_exception = ModelAnswerFailed(failure)

        failed_model = request.model
        for fallback_model in self.models:
            self._raise_if_cancelled()
            _log_fallback(failed_model, last_exception, fallback_model)
            failed_model = fallback_model
            try:
                response = await handler(request.override(model=fallback_model))
            except GraphInterrupt, AgentCancelled:
                raise
            except Exception as e:
                last_exception = e
                continue
            failure = failed_answer_reason(response)
            if failure is None:
                return response
            last_exception = ModelAnswerFailed(failure)

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
    generation_params: Mapping[str, Any] | None = None,
    owner: str = "agent",
) -> ModelFallbackMiddleware | None:
    """Build ``ModelFallbackMiddleware`` from configured model names.

    ``generation_params`` carries per-agent generation-param overrides (from the
    agent's ``generation_params`` YAML field) so fallback models match the
    primary model's tuning. Keys a provider does not accept are dropped
    downstream by ``normalize_params``.
    """
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
                **(dict(generation_params) if generation_params else {}),
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

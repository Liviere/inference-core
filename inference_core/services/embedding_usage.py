"""What each embedding call used, for an application that wants to know.

The embedding service is called from many places (the memory store, vector
stores, the embeddings route, tools), most of which know nothing about who
the call is for. So it reports every call it makes to the hooks registered
here, and an application that keeps a usage log writes it from them, with
whatever it knows of the run the call belongs to.

Nothing is registered by default, and without a hook nothing is measured or
counted: the service behaves as it did before.

A report never carries the texts, only how many there were and what they
used. A hook that raises is logged (its exception's class, never its text)
and the embedding goes on.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReportedUsage:
    """What a provider's response said about one embedding call."""

    input_tokens: Optional[int] = None
    reported_model: Optional[str] = None
    # The provider's own figure for what the call cost, in its own unit (the
    # Cortecs gateway reports credits), when its response gives one.
    provider_cost: Optional[float] = None


@dataclass(frozen=True)
class EmbeddingUsage:
    """One call of the embedding service, without its texts."""

    backend: str
    """``local``, ``remote``, ``fake`` or a custom backend's name."""
    provider: Optional[str]
    """The remote provider (``cortecs``, ``deepinfra``, …); ``local`` or
    ``fake`` for the in-house backends."""
    model: Optional[str]
    texts: int
    input_tokens: Optional[int]
    estimated: bool
    """Whether ``input_tokens`` was counted here rather than reported by the
    provider."""
    latency_ms: int
    success: bool
    error_type: Optional[str] = None
    pricing: Any = None
    """The entry's ``PricingConfig`` (scaled already), or None."""
    reported_model: Optional[str] = None
    provider_cost: Optional[float] = None


EmbeddingUsageHook = Callable[[EmbeddingUsage], None]

_HOOKS: List[EmbeddingUsageHook] = []


def register_embedding_usage_hook(hook: EmbeddingUsageHook) -> None:
    """Call ``hook(usage)`` after every call of the embedding service.

    The hook runs in the caller's thread and context, synchronously, right
    after the backend returned (or failed). Registering the same hook again
    is a no-op, so a module that registers at start-up stays safe to import
    twice.
    """
    if not callable(hook):
        raise TypeError("An embedding usage hook must be callable")
    if hook not in _HOOKS:
        _HOOKS.append(hook)


def unregister_embedding_usage_hook(hook: EmbeddingUsageHook) -> None:
    """Forget a hook (a no-op for one never registered)."""
    try:
        _HOOKS.remove(hook)
    except ValueError:
        pass


def has_embedding_usage_hooks() -> bool:
    return bool(_HOOKS)


def report_embedding_usage(usage: EmbeddingUsage) -> None:
    """Hand ``usage`` to every hook; a failing hook never fails the call."""
    for hook in list(_HOOKS):
        try:
            hook(usage)
        except Exception as exc:
            logger.warning(
                "Embedding usage hook %s failed: %s",
                getattr(hook, "__qualname__", type(hook).__name__),
                type(exc).__name__,
            )


def estimate_tokens(texts: Sequence[Any]) -> Optional[int]:
    """Count the tokens of ``texts`` the way cost tracking estimates a
    chat call's: tiktoken's ``cl100k_base``, or about four characters a
    token without it. None when there is nothing to count."""
    strings = [text for text in texts if isinstance(text, str)]
    if not strings:
        return None
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = None
    if encoding is not None:
        try:
            return sum(
                len(encoding.encode(text, disallowed_special=())) for text in strings
            )
        except Exception:
            pass
    return sum(max(1, len(text) // 4) for text in strings if text)


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)

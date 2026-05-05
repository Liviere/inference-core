"""Best-effort resource cleanup helpers.

Third-party SDKs expose different lifecycle methods for network clients.  This
module centralizes the small amount of reflection needed to close those clients
without baking provider-specific assumptions into every caller.
"""

import asyncio
import concurrent.futures
import inspect
import logging
from collections.abc import Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

_CLOSE_METHODS = ("aclose", "close", "shutdown", "disconnect", "terminate")
_NESTED_RESOURCE_ATTRS = (
    "client",
    "async_client",
    "sync_client",
    "root_client",
    "connection_pool",
    "transport",
    "session",
    "imap_service",
    "_client",
    "_async_client",
    "_sync_client",
    "_transport",
    "_session",
)


async def close_resource(resource: Any, *, label: str | None = None) -> None:
    """Close a client-like object without knowing its exact SDK contract.

    WHY: OpenAI, Anthropic, Redis, MCP adapters, LangChain wrappers, and local
    service objects use different close method names.  A shared helper lets
    request/task shutdown paths close resources consistently while keeping
    cleanup best-effort and non-fatal.
    """
    await _close_resource(resource, label=label, seen=set())


def close_resource_sync(resource: Any, *, label: str | None = None) -> None:
    """Run ``close_resource`` from synchronous lifecycle boundaries.

    WHY: Some cleanup hooks are synchronous (context managers, Celery signals),
    while the resource to close may expose an async ``aclose`` method.  This
    wrapper guarantees cleanup still runs even when called from inside an
    already-running event loop.
    """
    coroutine = close_resource(resource, label=label)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coroutine)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(lambda: asyncio.run(coroutine)).result()


async def _close_resource(
    resource: Any,
    *,
    label: str | None,
    seen: set[int],
) -> None:
    if resource is None:
        return

    resource_id = id(resource)
    if resource_id in seen:
        return
    seen.add(resource_id)

    if isinstance(resource, Mapping):
        for key, value in list(resource.items()):
            await _close_resource(
                value, label=f"{label or 'mapping'}[{key!r}]", seen=seen
            )
        return

    if _is_plain_iterable(resource):
        for index, value in enumerate(list(resource)):
            await _close_resource(
                value, label=f"{label or 'iterable'}[{index}]", seen=seen
            )
        return

    close_called = await _call_first_close_method(resource, label=label)

    for attr_name in _NESTED_RESOURCE_ATTRS:
        nested = getattr(resource, attr_name, None)
        if nested is None:
            continue
        await _close_resource(
            nested,
            label=f"{label or type(resource).__name__}.{attr_name}",
            seen=seen,
        )

    if not close_called:
        await _call_context_exit(resource, label=label)


def _is_plain_iterable(resource: Any) -> bool:
    """Return True only for containers whose values should be closed."""
    if isinstance(resource, (str, bytes, bytearray)):
        return False
    return isinstance(resource, Iterable)


async def _call_first_close_method(resource: Any, *, label: str | None) -> bool:
    for method_name in _CLOSE_METHODS:
        method = getattr(resource, method_name, None)
        if not callable(method):
            continue
        try:
            result = method()
            if inspect.isawaitable(result):
                await result
            return True
        except TypeError:
            logger.debug(
                "Resource close method requires arguments: %s.%s",
                label or type(resource).__name__,
                method_name,
            )
            return False
        except Exception:
            logger.debug(
                "Resource close failed: %s.%s",
                label or type(resource).__name__,
                method_name,
                exc_info=True,
            )
            return True
    return False


async def _call_context_exit(resource: Any, *, label: str | None) -> None:
    async_exit = getattr(resource, "__aexit__", None)
    if callable(async_exit):
        try:
            result = async_exit(None, None, None)
            if inspect.isawaitable(result):
                await result
            return
        except Exception:
            logger.debug(
                "Async context cleanup failed: %s",
                label or type(resource).__name__,
                exc_info=True,
            )

    sync_exit = getattr(resource, "__exit__", None)
    if callable(sync_exit):
        try:
            sync_exit(None, None, None)
        except Exception:
            logger.debug(
                "Context cleanup failed: %s",
                label or type(resource).__name__,
                exc_info=True,
            )

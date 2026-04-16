"""Utilities for gracefully interrupting LangGraph agent streams.

WHY THIS EXISTS:

LangGraph's ``astream`` async generator reports *any* ``BaseException``
(including ``GeneratorExit``) to LangSmith via ``run_manager.on_chain_error``.
When a consumer calls ``aclose()`` on the stream (explicitly or via ``break``
in ``async for``), the resulting ``GeneratorExit`` makes LangSmith mark the
entire run as **Failed** — even though the interruption was intentional.

APPROACH — ``GraphInterrupt`` via callback:

``GraphInterrupt`` is the *only* exception that LangGraph's execution loop
suppresses cleanly (``_loop.__exit__`` returns ``True``).  When it is raised
inside a running node, LangGraph:

1. Saves the interrupt to the checkpointer (if any).
2. Finishes the current super-step — ``loop.tick()`` returns False.
3. Emits all remaining output events (updates, values, …).
4. Calls ``run_manager.on_chain_end()`` — LangSmith sees **Success** at the
   graph level.

We exploit this by registering a **callback** (``StreamCancelCallback``)
that raises ``GraphInterrupt`` from inside ``on_llm_new_token`` the moment
the consumer signals cancellation.  Because the exception originates inside
the LLM call:

* The HTTP connection to the provider is **closed immediately** — the
  provider stops generating tokens, saving cost.
* The LLM-level sub-trace in LangSmith shows as Error (``on_llm_error``
  fires before ``GraphInterrupt`` propagates), but the **parent graph
  trace** shows as Success — which is the primary metric users track.

After the interrupt, a small number of cheap internal LangGraph events
(``updates``, checkpoint writes) remain in the stream.  Those are consumed
by a quick background drain that adds no LLM cost.

USAGE::

    cancel_cb = StreamCancelCallback()
    config = {"callbacks": [cancel_cb], "configurable": {"thread_id": "…"}}

    raw_stream = agent.astream(input, config, stream_mode="messages")
    stream = InterruptibleStream(raw_stream, cancel_callback=cancel_cb)

    async for chunk in stream:
        if should_stop:
            await stream.stop()   # raises GraphInterrupt on next LLM token
            break
    await stream.close()          # waits for drain of remaining events
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langgraph.errors import GraphInterrupt

logger = logging.getLogger(__name__)

# Upper bound for draining internal LangGraph events after the LLM stops.
# After GraphInterrupt only bookkeeping events remain — should finish fast.
_DEFAULT_DRAIN_TIMEOUT: float = 10.0


class StreamCancelCallback(AsyncCallbackHandler):
    """Async LangChain callback that raises ``GraphInterrupt`` on cancellation.

    Use this callback **only** with async execution paths
    (``arun_agent_steps`` / ``astream``).  For sync execution paths use
    :class:`SyncStreamCancelCallback` instead — see its docstring for why.

    Requires ``raise_error = True`` so that the exception propagates from
    the callback through the chat model back to LangGraph's runner.
    ``run_inline = True`` ensures the check happens on the same task as the
    model call (no scheduling delay).
    """

    raise_error: bool = True
    run_inline: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._cancel: bool = False

    def cancel(self) -> None:
        """Signal that the next LLM token should trigger a GraphInterrupt."""
        self._cancel = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancel

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if self._cancel:
            raise GraphInterrupt(())

    # No-op overrides — prevent ``NotImplementedError`` noise from the
    # inherited ``AsyncCallbackHandler`` default implementations.
    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        pass

    async def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        pass


class SyncStreamCancelCallback(BaseCallbackHandler):
    """Sync LangChain callback that raises ``GraphInterrupt`` on cancellation.

    WHY: LangChain's sync ``handle_event`` dispatches ``AsyncCallbackHandler``
    methods via ``_run_coros`` which **catches and logs** all exceptions
    instead of propagating them.  This means ``GraphInterrupt`` raised inside
    an async callback is silently swallowed — producing repeated
    ``"Error in callback coroutine: GraphInterrupt(())"`` warnings and never
    actually stopping the LLM.

    By extending ``BaseCallbackHandler`` (sync), the ``on_llm_new_token``
    call runs synchronously on the same thread and its ``GraphInterrupt``
    propagates directly through LangGraph's execution loop, stopping the
    provider immediately.

    Use this callback with sync execution paths (``run_agent_steps`` /
    ``stream``).
    """

    raise_error: bool = True
    run_inline: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._cancel: bool = False

    def cancel(self) -> None:
        """Signal that the next LLM token should trigger a GraphInterrupt."""
        self._cancel = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancel

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if self._cancel:
            raise GraphInterrupt(())

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        pass

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        pass


class InterruptibleStream:
    """Async-iterator wrapper combining ``StreamCancelCallback`` with
    a background drain for clean stream shutdown.

    WHY: Calling ``aclose()`` on a LangGraph ``astream`` generator causes
    ``GeneratorExit`` → ``on_chain_error`` → LangSmith marks it as Failed.
    This wrapper uses ``GraphInterrupt`` instead to stop the LLM, then
    drains the few remaining internal events so the stream finishes
    naturally.

    ``cancel_callback`` is optional — when not provided, ``stop()``
    falls back to a passive drain (the LLM keeps generating tokens but
    they are discarded).  Always prefer providing the callback for real
    cost savings::

        cancel_cb = StreamCancelCallback()
        config = {"callbacks": [cancel_cb], ...}
        raw_stream = agent.astream(input, config, ...)
        stream = InterruptibleStream(raw_stream, cancel_callback=cancel_cb)

        async for chunk in stream:
            if should_stop:
                await stream.stop()
                break
        await stream.close()
    """

    def __init__(
        self,
        inner_stream: AsyncIterator[Any],
        cancel_callback: StreamCancelCallback | None = None,
        drain_timeout: float = _DEFAULT_DRAIN_TIMEOUT,
    ) -> None:
        self._inner: AsyncIterator[Any] = inner_stream.__aiter__()
        self._cancel_cb = cancel_callback
        self._stopped = False
        self._drain_task: asyncio.Task[None] | None = None
        self._drain_timeout = drain_timeout

    # -- async-iterator protocol -------------------------------------------

    def __aiter__(self) -> "InterruptibleStream":
        return self

    async def __anext__(self) -> Any:
        if self._stopped:
            raise StopAsyncIteration
        return await self._inner.__anext__()

    async def aclose(self) -> None:
        """Drain remaining events instead of propagating GeneratorExit."""
        await self.stop()
        await self.close()

    # -- public API --------------------------------------------------------

    async def stop(self) -> None:
        """Signal the LLM to stop generating and begin draining.

        If a ``StreamCancelCallback`` was provided, sets its cancel flag so
        that the very next ``on_llm_new_token`` raises ``GraphInterrupt``.
        The LLM HTTP connection is closed immediately by the exception
        propagation — the provider stops billing for further tokens.

        In all cases, marks this stream as exhausted (``__anext__`` will
        raise ``StopAsyncIteration``) and starts a background drain for the
        remaining cheap internal LangGraph events.
        """
        if self._stopped:
            return
        self._stopped = True
        if self._cancel_cb is not None:
            self._cancel_cb.cancel()
        self._ensure_drain()

    async def close(self) -> None:
        """Wait for the background drain to complete.

        Safe to call multiple times or when no drain is pending.
        """
        if self._drain_task is not None:
            await self._drain_task

    # -- internals ---------------------------------------------------------

    def _ensure_drain(self) -> None:
        """Schedule a background task to consume remaining events."""
        if self._drain_task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
            self._drain_task = loop.create_task(
                self._drain_with_timeout(),
                name="interruptible-stream-drain",
            )
        except RuntimeError:
            logger.debug("InterruptibleStream: no event loop, skipping drain")

    async def _drain_with_timeout(self) -> None:
        """Drain remaining events, falling back to hard close on timeout."""
        try:
            await asyncio.wait_for(self._drain_inner(), timeout=self._drain_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "InterruptibleStream: drain timed out after %.1fs, "
                "falling back to hard aclose (trace may show as failed)",
                self._drain_timeout,
            )
            if hasattr(self._inner, "aclose"):
                try:
                    await self._inner.aclose()
                except Exception:
                    pass

    async def _drain_inner(self) -> None:
        """Silently consume remaining events so the inner generator finishes.

        After GraphInterrupt, only cheap bookkeeping events remain —
        no more LLM tokens are generated.
        """
        try:
            while True:
                await self._inner.__anext__()
        except StopAsyncIteration:
            pass
        except Exception:
            logger.debug(
                "InterruptibleStream: exception during drain (suppressed)",
                exc_info=True,
            )


class SyncInterruptibleStream:
    """Sync-iterator wrapper combining ``StreamCancelCallback`` with
    a blocking drain for clean stream shutdown.

    WHY: The sync ``agent.stream()`` generator suffers from the same
    ``GeneratorExit`` → LangSmith-failed problem as the async path.
    When ``stop()`` is called, the cancel callback signals
    ``GraphInterrupt`` on the next ``on_llm_new_token``.  ``drain()``
    then consumes the remaining cheap internal LangGraph events so the
    generator finishes naturally — no ``GeneratorExit`` is ever sent.

    Usage::

        cancel_cb = SyncStreamCancelCallback()
        config = {"callbacks": [cancel_cb], "configurable": {"thread_id": "…"}}

        raw = agent.stream(input, config, stream_mode="messages")
        stream = SyncInterruptibleStream(raw, cancel_callback=cancel_cb)

        for chunk in stream:
            if should_stop:
                stream.stop()
                break
        stream.drain()   # consume remaining events (no LLM cost)
    """

    def __init__(
        self,
        inner_stream,
        cancel_callback: StreamCancelCallback | SyncStreamCancelCallback | None = None,
        drain_timeout: float = _DEFAULT_DRAIN_TIMEOUT,
    ) -> None:
        self._inner = iter(inner_stream)
        self._cancel_cb = cancel_callback
        self._stopped = False
        self._drain_timeout = drain_timeout

    # -- iterator protocol -------------------------------------------------

    def __iter__(self) -> "SyncInterruptibleStream":
        return self

    def __next__(self):
        if self._stopped:
            raise StopIteration
        return next(self._inner)

    # -- public API --------------------------------------------------------

    def stop(self) -> None:
        """Signal the LLM to stop and mark stream as exhausted."""
        if self._stopped:
            return
        self._stopped = True
        if self._cancel_cb is not None:
            self._cancel_cb.cancel()

    def drain(self) -> None:
        """Consume remaining events so the generator finishes naturally.

        After ``GraphInterrupt``, only a handful of cheap bookkeeping
        events remain.  Falls back to a hard ``.close()`` on timeout.
        """
        import time

        deadline = time.monotonic() + self._drain_timeout
        try:
            while True:
                if time.monotonic() > deadline:
                    logger.warning(
                        "SyncInterruptibleStream: drain timed out after %.1fs, "
                        "falling back to hard close (trace may show as failed)",
                        self._drain_timeout,
                    )
                    if hasattr(self._inner, "close"):
                        try:
                            self._inner.close()
                        except Exception:
                            pass
                    return
                next(self._inner)
        except StopIteration:
            pass
        except Exception:
            logger.debug(
                "SyncInterruptibleStream: exception during drain (suppressed)",
                exc_info=True,
            )

    def close(self) -> None:
        """Convenience: stop + drain."""
        self.stop()
        self.drain()

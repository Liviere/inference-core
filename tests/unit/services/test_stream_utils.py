"""Unit tests for InterruptibleStream — graceful LangGraph stream interruption."""

import asyncio

import pytest
from langgraph.errors import GraphInterrupt

from inference_core.services.stream_utils import (
    InterruptibleStream,
    StreamCancelCallback,
    SyncInterruptibleStream,
    SyncStreamCancelCallback,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_range(n: int):
    """Async generator that yields 0..n-1."""
    for i in range(n):
        yield i


async def _slow_async_range(n: int, delay: float = 0.05):
    """Async generator with a delay between yields."""
    for i in range(n):
        await asyncio.sleep(delay)
        yield i


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInterruptibleStreamBasic:
    """Verify that InterruptibleStream correctly wraps an async iterator."""

    async def test_passes_all_chunks_when_not_stopped(self):
        stream = InterruptibleStream(_async_range(5))
        collected = [chunk async for chunk in stream]
        assert collected == [0, 1, 2, 3, 4]

    async def test_stop_causes_early_exit(self):
        stream = InterruptibleStream(_async_range(10))
        collected = []
        async for chunk in stream:
            collected.append(chunk)
            if chunk == 2:
                await stream.stop()
                break
        assert collected == [0, 1, 2]

    async def test_anext_raises_stop_after_stop_called(self):
        stream = InterruptibleStream(_async_range(5))
        await stream.stop()
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()


class TestInterruptibleStreamDrain:
    """Verify that remaining chunks are drained on close / stop."""

    async def test_drain_consumes_remaining_chunks(self):
        """After stop(), remaining chunks should be consumed by drain."""
        consumed = []

        async def tracked_gen():
            for i in range(10):
                consumed.append(i)
                yield i

        stream = InterruptibleStream(tracked_gen())
        async for chunk in stream:
            if chunk == 3:
                await stream.stop()
                break

        # Wait for drain to finish
        await stream.close()
        assert consumed == list(range(10))

    async def test_aclose_drains_and_awaits(self):
        """Explicit aclose() should drain and then return."""
        consumed = []

        async def tracked_gen():
            for i in range(5):
                consumed.append(i)
                yield i

        stream = InterruptibleStream(tracked_gen())
        # Read only one chunk
        chunk = await stream.__anext__()
        assert chunk == 0

        # Close — should drain the rest
        await stream.aclose()
        assert consumed == [0, 1, 2, 3, 4]

    async def test_async_for_break_with_stop_drains(self):
        """Breaking out of async-for after stop() triggers background drain."""
        consumed = []

        async def tracked_gen():
            for i in range(8):
                consumed.append(i)
                yield i

        stream = InterruptibleStream(tracked_gen())
        async for chunk in stream:
            if chunk == 1:
                # Must call stop() before break — async-for does NOT
                # automatically call aclose() on custom async iterators.
                await stream.stop()
                break

        # stop() schedules background drain; give it time to finish
        await stream.close()
        assert consumed == list(range(8))


class TestInterruptibleStreamTimeout:
    """Verify drain timeout handling."""

    async def test_drain_timeout_falls_back_to_hard_close(self):
        """When drain exceeds timeout, stream falls back to aclose."""

        async def infinite_gen():
            i = 0
            while True:
                yield i
                i += 1
                await asyncio.sleep(0.01)

        stream = InterruptibleStream(infinite_gen(), drain_timeout=0.1)
        async for chunk in stream:
            if chunk == 2:
                await stream.stop()
                break

        # close() should complete (not hang) thanks to the timeout
        await stream.close()


class TestInterruptibleStreamSafe:
    """Edge cases and safety checks."""

    async def test_double_stop_is_safe(self):
        stream = InterruptibleStream(_async_range(5))
        await stream.stop()
        await stream.stop()  # should not error
        await stream.close()

    async def test_await_drain_without_stop_is_safe(self):
        stream = InterruptibleStream(_async_range(3))
        collected = [chunk async for chunk in stream]
        assert collected == [0, 1, 2]
        # No stop was called — close should be a no-op
        await stream.close()

    async def test_generator_error_during_drain_is_suppressed(self):
        async def failing_gen():
            yield 0
            yield 1
            raise ValueError("boom")

        stream = InterruptibleStream(failing_gen())
        chunk = await stream.__anext__()
        assert chunk == 0

        await stream.aclose()
        # Should complete without raising — error is suppressed


# ---------------------------------------------------------------------------
# SyncStreamCancelCallback tests
# ---------------------------------------------------------------------------


class TestSyncStreamCancelCallback:
    """Verify the sync cancel callback raises GraphInterrupt synchronously."""

    def test_raises_graph_interrupt_when_cancelled(self):
        cb = SyncStreamCancelCallback()
        cb.cancel()
        with pytest.raises(GraphInterrupt):
            cb.on_llm_new_token("token")

    def test_no_raise_when_not_cancelled(self):
        cb = SyncStreamCancelCallback()
        cb.on_llm_new_token("token")  # should not raise

    def test_is_cancelled_flag(self):
        cb = SyncStreamCancelCallback()
        assert not cb.is_cancelled
        cb.cancel()
        assert cb.is_cancelled

    def test_on_chat_model_start_is_noop(self):
        cb = SyncStreamCancelCallback()
        cb.on_chat_model_start({}, [[]])  # should not raise

    def test_on_llm_start_is_noop(self):
        cb = SyncStreamCancelCallback()
        cb.on_llm_start({}, [])  # should not raise


class TestStreamCancelCallbackNoops:
    """Verify the async cancel callback has no-op startup hooks."""

    async def test_on_chat_model_start_is_noop(self):
        cb = StreamCancelCallback()
        await cb.on_chat_model_start({}, [[]])  # should not raise

    async def test_on_llm_start_is_noop(self):
        cb = StreamCancelCallback()
        await cb.on_llm_start({}, [])  # should not raise


# ---------------------------------------------------------------------------
# SyncInterruptibleStream tests
# ---------------------------------------------------------------------------


class TestSyncInterruptibleStreamBasic:
    """Verify SyncInterruptibleStream with the sync cancel callback."""

    def test_passes_all_chunks_when_not_stopped(self):
        stream = SyncInterruptibleStream(range(5))
        collected = list(stream)
        assert collected == [0, 1, 2, 3, 4]

    def test_stop_causes_early_exit(self):
        cb = SyncStreamCancelCallback()
        stream = SyncInterruptibleStream(range(10), cancel_callback=cb)
        collected = []
        for chunk in stream:
            collected.append(chunk)
            if chunk == 2:
                stream.stop()
                break
        assert collected == [0, 1, 2]
        assert cb.is_cancelled

    def test_drain_after_stop(self):
        consumed = []

        def tracked_gen():
            for i in range(8):
                consumed.append(i)
                yield i

        cb = SyncStreamCancelCallback()
        stream = SyncInterruptibleStream(tracked_gen(), cancel_callback=cb)
        for chunk in stream:
            if chunk == 2:
                stream.stop()
                break
        stream.drain()
        assert consumed == list(range(8))

    def test_close_combines_stop_and_drain(self):
        consumed = []

        def tracked_gen():
            for i in range(5):
                consumed.append(i)
                yield i

        cb = SyncStreamCancelCallback()
        stream = SyncInterruptibleStream(tracked_gen(), cancel_callback=cb)
        _ = next(iter(stream))
        stream.close()
        assert consumed == [0, 1, 2, 3, 4]

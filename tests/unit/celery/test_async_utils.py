"""Tests for Celery async_utils module.

Covers run_async_safely, run_in_worker_loop, get_current_loop,
get_worker_loop_safe, and _run_in_new_thread_loop.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from inference_core.celery.async_utils import (
    _current_loop_ctx,
    _run_in_new_thread_loop,
    get_current_loop,
    get_worker_loop_safe,
    is_in_worker_context,
    run_async_safely,
    run_in_worker_loop,
)

# ---------------------------------------------------------------------------
# get_worker_loop_safe
# ---------------------------------------------------------------------------


class TestGetWorkerLoopSafe:
    """Verify safe retrieval of the Celery worker event loop."""

    def test_returns_loop_when_available(self):
        """Returns the worker loop when get_worker_loop() succeeds."""
        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False

        with patch(
            "inference_core.celery.async_utils.get_worker_loop",
            return_value=mock_loop,
            create=True,
        ):
            with patch(
                "inference_core.celery.celery_main.get_worker_loop",
                return_value=mock_loop,
                create=True,
            ):
                result = get_worker_loop_safe()

        assert result is mock_loop

    def test_returns_none_when_loop_closed(self):
        """Returns None if worker loop exists but is closed."""
        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = True

        with patch(
            "inference_core.celery.celery_main.get_worker_loop",
            return_value=mock_loop,
            create=True,
        ):
            result = get_worker_loop_safe()

        assert result is None

    def test_returns_none_when_import_fails(self):
        """Returns None if celery_main import raises."""
        with patch(
            "inference_core.celery.celery_main.get_worker_loop",
            side_effect=ImportError("no celery"),
            create=True,
        ):
            result = get_worker_loop_safe()

        assert result is None

    def test_returns_none_when_no_worker(self):
        """Returns None when get_worker_loop returns None."""
        with patch(
            "inference_core.celery.celery_main.get_worker_loop",
            return_value=None,
            create=True,
        ):
            result = get_worker_loop_safe()

        assert result is None


# ---------------------------------------------------------------------------
# get_current_loop
# ---------------------------------------------------------------------------


class TestGetCurrentLoop:
    """Verify event loop resolution from context variable or worker."""

    def test_prefers_context_variable(self):
        """Returns loop from _current_loop_ctx when set."""
        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False

        token = _current_loop_ctx.set(mock_loop)
        try:
            result = get_current_loop()
            assert result is mock_loop
        finally:
            _current_loop_ctx.reset(token)

    def test_falls_back_to_worker_loop(self):
        """When context var is None, falls back to worker loop."""
        token = _current_loop_ctx.set(None)
        try:
            mock_loop = MagicMock()
            mock_loop.is_closed.return_value = False

            with patch(
                "inference_core.celery.async_utils.get_worker_loop_safe",
                return_value=mock_loop,
            ):
                result = get_current_loop()
            assert result is mock_loop
        finally:
            _current_loop_ctx.reset(token)

    def test_returns_none_when_nothing_available(self):
        """Returns None when no loop is available."""
        token = _current_loop_ctx.set(None)
        try:
            with patch(
                "inference_core.celery.async_utils.get_worker_loop_safe",
                return_value=None,
            ):
                result = get_current_loop()
            assert result is None
        finally:
            _current_loop_ctx.reset(token)

    def test_ignores_closed_context_loop(self):
        """Skips context variable if loop is closed, tries worker loop."""
        closed_loop = MagicMock()
        closed_loop.is_closed.return_value = True

        worker_loop = MagicMock()
        worker_loop.is_closed.return_value = False

        token = _current_loop_ctx.set(closed_loop)
        try:
            with patch(
                "inference_core.celery.async_utils.get_worker_loop_safe",
                return_value=worker_loop,
            ):
                result = get_current_loop()
            assert result is worker_loop
        finally:
            _current_loop_ctx.reset(token)


# ---------------------------------------------------------------------------
# is_in_worker_context
# ---------------------------------------------------------------------------


class TestIsInWorkerContext:
    """Verify worker context detection."""

    def test_true_when_worker_loop_available(self):
        """Returns True when get_worker_loop_safe returns a loop."""
        with patch(
            "inference_core.celery.async_utils.get_worker_loop_safe",
            return_value=MagicMock(),
        ):
            assert is_in_worker_context() is True

    def test_false_when_no_worker_loop(self):
        """Returns False when get_worker_loop_safe returns None."""
        with patch(
            "inference_core.celery.async_utils.get_worker_loop_safe",
            return_value=None,
        ):
            assert is_in_worker_context() is False


# ---------------------------------------------------------------------------
# run_in_worker_loop
# ---------------------------------------------------------------------------


class TestRunInWorkerLoop:
    """Verify coroutine execution on worker loop with context propagation."""

    def test_runs_on_worker_loop(self):
        """Coroutine executes on the worker loop and returns result."""
        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False
        mock_loop.run_until_complete.return_value = 42

        with patch(
            "inference_core.celery.celery_main.get_worker_loop",
            return_value=mock_loop,
            create=True,
        ):
            async def sample_coro():
                return 42

            result = run_in_worker_loop(sample_coro())

        assert result == 42
        mock_loop.run_until_complete.assert_called_once()

    def test_sets_and_resets_context_variable(self):
        """Context variable is set during execution and reset after."""
        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False
        mock_loop.run_until_complete.return_value = "ok"

        original_ctx = _current_loop_ctx.get()

        with patch(
            "inference_core.celery.celery_main.get_worker_loop",
            return_value=mock_loop,
            create=True,
        ):
            async def check_coro():
                return "ok"

            run_in_worker_loop(check_coro())

        # After execution, context should be reset
        assert _current_loop_ctx.get() == original_ctx

    def test_fallback_to_temporary_loop(self):
        """When worker loop is None, creates temporary loop."""
        with patch(
            "inference_core.celery.celery_main.get_worker_loop",
            return_value=None,
            create=True,
        ):
            async def add():
                return 1 + 1

            result = run_in_worker_loop(add())
            assert result == 2


# ---------------------------------------------------------------------------
# run_async_safely
# ---------------------------------------------------------------------------


class TestRunAsyncSafely:
    """Verify the unified sync-to-async wrapper."""

    def test_reuses_context_loop_when_not_running(self):
        """When context loop is set and not running, uses it directly."""
        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False
        mock_loop.is_running.return_value = False
        mock_loop.run_until_complete.return_value = "result"

        token = _current_loop_ctx.set(mock_loop)
        try:
            with patch(
                "inference_core.celery.async_utils.get_worker_loop_safe",
                return_value=None,
            ):
                async def my_coro():
                    return "result"

                result = run_async_safely(my_coro())
        finally:
            _current_loop_ctx.reset(token)

        assert result == "result"
        mock_loop.run_until_complete.assert_called_once()

    def test_falls_back_to_thread_when_loop_running(self):
        """When loop is running, falls back to thread pool executor."""
        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False
        mock_loop.is_running.return_value = True  # Cannot use run_until_complete

        token = _current_loop_ctx.set(mock_loop)
        try:
            with patch(
                "inference_core.celery.async_utils.get_worker_loop_safe",
                return_value=None,
            ):
                async def compute():
                    return 99

                result = run_async_safely(compute(), timeout=5.0)
        finally:
            _current_loop_ctx.reset(token)

        assert result == 99

    def test_works_with_no_loop_at_all(self):
        """When no loop is available, uses thread pool fallback."""
        token = _current_loop_ctx.set(None)
        try:
            with patch(
                "inference_core.celery.async_utils.get_worker_loop_safe",
                return_value=None,
            ):
                async def greet():
                    return "hello"

                result = run_async_safely(greet(), timeout=5.0)
        finally:
            _current_loop_ctx.reset(token)

        assert result == "hello"

    def test_propagates_coroutine_exception(self):
        """Exceptions from the coroutine are propagated to caller."""
        token = _current_loop_ctx.set(None)
        try:
            with patch(
                "inference_core.celery.async_utils.get_worker_loop_safe",
                return_value=None,
            ):
                async def blow_up():
                    raise ValueError("kaboom")

                with pytest.raises(ValueError, match="kaboom"):
                    run_async_safely(blow_up(), timeout=5.0)
        finally:
            _current_loop_ctx.reset(token)

    def test_no_timeout_skips_wait_for(self):
        """When timeout=None, coroutine runs without asyncio.wait_for wrapper."""
        token = _current_loop_ctx.set(None)
        try:
            with patch(
                "inference_core.celery.async_utils.get_worker_loop_safe",
                return_value=None,
            ):
                async def quick():
                    return "no timeout"

                result = run_async_safely(quick(), timeout=None)
        finally:
            _current_loop_ctx.reset(token)

        assert result == "no timeout"


# ---------------------------------------------------------------------------
# _run_in_new_thread_loop
# ---------------------------------------------------------------------------


class TestRunInNewThreadLoop:
    """Verify the thread-pool fallback for isolated loop execution."""

    def test_runs_coroutine_in_thread(self):
        """Coroutine runs in a new loop in a thread pool worker."""

        async def compute():
            return 7 * 6

        result = _run_in_new_thread_loop(compute(), timeout=5.0)
        assert result == 42

    def test_propagates_errors(self):
        """Errors are propagated from thread to caller."""

        async def fail():
            raise RuntimeError("thread error")

        with pytest.raises(RuntimeError, match="thread error"):
            _run_in_new_thread_loop(fail(), timeout=5.0)

    def test_works_without_timeout(self):
        """Coroutine runs without timeout wrapper when timeout=None."""

        async def simple():
            return "done"

        result = _run_in_new_thread_loop(simple(), timeout=None)
        assert result == "done"

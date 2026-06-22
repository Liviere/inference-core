"""
Tests for the global vector-store service singleton getter.

Regression coverage for ASSISTANTS-3: under a threaded Celery worker pool many
threads race on first access to ``get_vector_store_service()``. The getter must
never publish a half-initialized service (one whose ``provider`` is still None),
and must recover if an earlier build left the provider unset due to a transient
factory error.
"""

import threading
import time
from unittest.mock import patch

import pytest

import inference_core.services.vector_store_service as vss_mod
from inference_core.services.vector_store_service import get_vector_store_service


@pytest.fixture(autouse=True)
def _reset_service_singleton():
    """Each test starts and ends with a fresh process-wide service singleton."""
    vss_mod._vector_store_service = None
    yield
    vss_mod._vector_store_service = None


def test_getter_publishes_only_fully_initialized_service():
    provider = object()
    with patch.object(vss_mod, "get_vector_store_provider", return_value=provider):
        service = get_vector_store_service()

    assert service.provider is provider
    assert service.is_available is True
    # The module global is the same fully-initialized instance.
    assert vss_mod._vector_store_service is service


def test_getter_returns_same_singleton_instance():
    provider = object()
    with patch.object(vss_mod, "get_vector_store_provider", return_value=provider):
        first = get_vector_store_service()
        second = get_vector_store_service()

    assert first is second


def test_getter_recovers_from_transient_none_provider():
    """A first build with a None provider is retried on the next call (same instance)."""
    provider = object()
    sequence = [None, provider]

    def fake_get_provider():
        return sequence.pop(0)

    with patch.object(
        vss_mod, "get_vector_store_provider", side_effect=fake_get_provider
    ):
        first = get_vector_store_service()  # provider resolves to None
        assert first.provider is None
        assert first.is_available is False

        second = get_vector_store_service()  # retry -> provider now set

    assert second is first  # same singleton object, healed in place
    assert second.provider is provider
    assert second.is_available is True


def test_concurrent_first_access_never_observes_uninitialized_service():
    """Build-before-publish + locking: no concurrent caller sees provider=None.

    This fails on the previous implementation, which published the service to the
    module global BEFORE initializing its provider — letting a second thread
    return the half-built instance (is_available False) during the init window.
    """
    provider = object()

    def slow_get_provider():
        # Widen the init window and release the GIL so other threads can run.
        time.sleep(0.02)
        return provider

    results: list[bool] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(20)

    def worker():
        try:
            barrier.wait()  # maximize contention on the first build
            service = get_vector_store_service()
            results.append(service.is_available)
        except Exception as exc:  # pragma: no cover - surfaced via assertion
            errors.append(exc)

    with patch.object(
        vss_mod, "get_vector_store_provider", side_effect=slow_get_provider
    ):
        threads = [threading.Thread(target=worker) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert not errors
    assert len(results) == 20
    # Every caller observed a fully-initialized, available service.
    assert all(results)

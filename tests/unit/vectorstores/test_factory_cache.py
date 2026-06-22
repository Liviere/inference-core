"""
Tests for the vector-store provider factory cache semantics.

Regression coverage for ASSISTANTS-3: a *transient* provider construction
failure must NOT be cached permanently (which previously pinned a Celery worker
into "Vector store service is not available" for its whole lifetime). A disabled
backend and a successful provider ARE cached.
"""

from unittest.mock import MagicMock, patch

import pytest

import inference_core.vectorstores.factory as factory_mod
from inference_core.vectorstores.factory import (
    clear_provider_cache,
    get_vector_store_provider,
)


@pytest.fixture(autouse=True)
def _reset_provider_cache():
    """Each test starts and ends with a clean process-wide provider cache."""
    clear_provider_cache()
    yield
    clear_provider_cache()


def _enabled_settings(backend: str = "memory") -> MagicMock:
    settings = MagicMock()
    settings.is_vector_store_enabled = True
    settings.vector_backend = backend
    return settings


def test_transient_failure_is_not_cached_and_next_call_recovers():
    """A construction error returns None WITHOUT caching, so the next call retries."""
    sentinel = object()
    calls = {"n": 0}

    def flaky(backend, config):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient boom")
        return sentinel

    with (
        patch.object(factory_mod, "get_settings", return_value=_enabled_settings()),
        patch.object(factory_mod, "_get_config_from_settings", return_value={}),
        patch.object(factory_mod, "create_vector_store_provider", side_effect=flaky),
    ):
        # First call fails to construct -> None, and is NOT memoized.
        assert get_vector_store_provider() is None
        # Second call retries and succeeds -> real provider, now cached.
        assert get_vector_store_provider() is sentinel
        # Third call is served from cache (no further construction attempt).
        assert get_vector_store_provider() is sentinel

    assert calls["n"] == 2  # fail + success; third served from cache


def test_successful_provider_is_cached_as_singleton():
    sentinel = object()
    with (
        patch.object(factory_mod, "get_settings", return_value=_enabled_settings()),
        patch.object(factory_mod, "_get_config_from_settings", return_value={}),
        patch.object(
            factory_mod, "create_vector_store_provider", return_value=sentinel
        ) as make,
    ):
        first = get_vector_store_provider()
        second = get_vector_store_provider()

    assert first is second is sentinel
    make.assert_called_once()


def test_disabled_backend_is_cached_and_short_circuits():
    settings = MagicMock()
    settings.is_vector_store_enabled = False
    with (
        patch.object(factory_mod, "get_settings", return_value=settings) as get_set,
        patch.object(factory_mod, "create_vector_store_provider") as make,
    ):
        assert get_vector_store_provider() is None
        assert get_vector_store_provider() is None

    make.assert_not_called()
    # The "disabled" decision is memoized: settings consulted only once.
    assert get_set.call_count == 1


def test_clear_provider_cache_resets_both_success_and_disabled_state():
    sentinel = object()
    with (
        patch.object(factory_mod, "get_settings", return_value=_enabled_settings()),
        patch.object(factory_mod, "_get_config_from_settings", return_value={}),
        patch.object(
            factory_mod, "create_vector_store_provider", return_value=sentinel
        ) as make,
    ):
        assert get_vector_store_provider() is sentinel
        clear_provider_cache()
        # After clearing, the provider is rebuilt (construction called again).
        assert get_vector_store_provider() is sentinel

    assert make.call_count == 2

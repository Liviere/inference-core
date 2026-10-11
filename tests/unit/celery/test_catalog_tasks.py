"""
Tests for the Celery task that keeps the model catalog current.

Covers:
- the task does nothing while the catalog is switched off
- one run at a time
- a run reads the providers that are due and writes the gauges
- the gauges are written even when the reading did not finish
"""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_core.celery.tasks import catalog_tasks
from inference_core.llm.catalog.service import ProviderRefresh
from inference_core.llm.catalog.summary import CatalogSummary

MODULE = "inference_core.celery.tasks.catalog_tasks"


def _settings(enabled: bool) -> MagicMock:
    settings = MagicMock()
    settings.llm_model_catalog_enabled = enabled
    settings.llm_model_catalog_refresh_interval_seconds = 7200
    settings.llm_model_catalog_http_timeout_seconds = 7
    return settings


class TestCatalogRefreshTask:
    def test_does_nothing_while_switched_off(self):
        with (
            patch(f"{MODULE}.get_settings", return_value=_settings(False)),
            patch(f"{MODULE}.get_sync_redis") as get_redis,
            patch(f"{MODULE}.run_in_worker_loop") as run,
        ):
            assert catalog_tasks.catalog_refresh.run() == {"status": "disabled"}

        get_redis.assert_not_called()
        run.assert_not_called()

    def test_one_run_at_a_time(self):
        redis = MagicMock()
        redis.set.return_value = None
        with (
            patch(f"{MODULE}.get_settings", return_value=_settings(True)),
            patch(f"{MODULE}.get_sync_redis", return_value=redis),
            patch(f"{MODULE}.run_in_worker_loop") as run,
        ):
            result = catalog_tasks.catalog_refresh.run()

        assert result == {"status": "skipped", "reason": "lock"}
        run.assert_not_called()
        redis.delete.assert_not_called()

    def test_runs_with_the_configured_interval_and_frees_the_lock(self):
        redis = MagicMock()
        redis.set.return_value = True
        seen = {}

        def fake_run(coro):
            coro.close()
            return {"status": "ok", "providers": {}}

        async def fake_refresh_and_publish(interval, read_timeout):
            seen.update(interval=interval, read_timeout=read_timeout)

        with (
            patch(f"{MODULE}.get_settings", return_value=_settings(True)),
            patch(f"{MODULE}.get_sync_redis", return_value=redis),
            patch(f"{MODULE}.run_in_worker_loop", side_effect=fake_run),
            patch(
                f"{MODULE}._refresh_and_publish",
                side_effect=lambda interval, read_timeout: fake_refresh_and_publish(
                    interval, read_timeout
                ),
            ) as refresh_and_publish,
        ):
            result = catalog_tasks.catalog_refresh.run()

        assert result == {"status": "ok", "providers": {}}
        refresh_and_publish.assert_called_once_with(timedelta(seconds=7200), 7.0)
        redis.set.assert_called_once_with(
            catalog_tasks.CATALOG_REFRESH_LOCK_KEY, "1", nx=True, ex=600
        )
        redis.delete.assert_called_once_with(catalog_tasks.CATALOG_REFRESH_LOCK_KEY)

    def test_lock_is_freed_when_the_run_fails(self):
        redis = MagicMock()
        redis.set.return_value = True

        def failing_run(coro):
            coro.close()
            raise RuntimeError("boom")

        with (
            patch(f"{MODULE}.get_settings", return_value=_settings(True)),
            patch(f"{MODULE}.get_sync_redis", return_value=redis),
            patch(f"{MODULE}.run_in_worker_loop", side_effect=failing_run),
        ):
            with pytest.raises(RuntimeError):
                catalog_tasks.catalog_refresh.run()

        redis.delete.assert_called_once_with(catalog_tasks.CATALOG_REFRESH_LOCK_KEY)


class TestRefreshAndPublish:
    async def test_reads_due_providers_and_writes_the_gauges(self):
        summary = CatalogSummary(
            providers=["openai"], configured={("openai", "missing"): 1}
        )
        refresh = AsyncMock(
            return_value=[ProviderRefresh("openai", "refreshed", model_count=3)]
        )
        with (
            patch(f"{MODULE}.get_llm_config", return_value="config"),
            patch(f"{MODULE}.refresh", refresh),
            patch(f"{MODULE}.summarize", AsyncMock(return_value=summary)) as summarize,
            patch(f"{MODULE}.set_model_catalog_gauges") as set_gauges,
        ):
            result = await catalog_tasks._refresh_and_publish(timedelta(hours=2), 7.0)

        assert result == {"status": "ok", "providers": {"openai": "refreshed"}}
        (config,), options = refresh.call_args
        assert config == "config"
        assert options["only_due"] is True
        assert options["interval"] == timedelta(hours=2)
        assert options["read_timeout"] == 7.0
        assert summarize.call_args.kwargs["interval"] == timedelta(hours=2)
        gauges = set_gauges.call_args.kwargs
        assert gauges["providers"] == ["openai"]
        assert gauges["configured"] == {("openai", "missing"): 1}
        assert gauges["event_types"] == (
            "added",
            "removed",
            "reappeared",
            "lifecycle_changed",
        )
        assert gauges["drift_states"] == ("missing", "deprecated", "retiring")
        assert gauges["now"] == refresh.call_args.kwargs["now"].timestamp()

    async def test_gauges_are_written_when_the_reading_did_not_finish(self):
        with (
            patch(f"{MODULE}.get_llm_config", return_value="config"),
            patch(f"{MODULE}.refresh", AsyncMock(side_effect=TimeoutError())),
            patch(f"{MODULE}.summarize", AsyncMock(return_value=CatalogSummary())),
            patch(f"{MODULE}.set_model_catalog_gauges") as set_gauges,
        ):
            result = await catalog_tasks._refresh_and_publish(timedelta(days=1), 20.0)

        assert result == {"status": "ok", "providers": {}}
        set_gauges.assert_called_once()


class TestGaugesWhileSwitchedOff:
    def _reset(self, monkeypatch, *, enabled: bool, multiproc: bool):
        if multiproc:
            monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", "/tmp/does-not-matter")
        else:
            monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
        with (
            patch(f"{MODULE}.get_settings", return_value=_settings(enabled)),
            patch(f"{MODULE}.set_model_catalog_gauges") as set_gauges,
        ):
            catalog_tasks.reset_gauges_while_switched_off(sender=None)
        return set_gauges

    def test_a_worker_starting_with_the_catalog_off_zeroes_them(self, monkeypatch):
        set_gauges = self._reset(monkeypatch, enabled=False, multiproc=True)

        gauges = set_gauges.call_args.kwargs
        assert "openai" in gauges["providers"]
        assert gauges["now"] == 0.0
        for name in ("last_success", "stale", "model_counts", "configured"):
            assert gauges[name] == {}

    def test_left_alone_while_the_catalog_is_on(self, monkeypatch):
        self._reset(monkeypatch, enabled=True, multiproc=True).assert_not_called()

    def test_left_alone_where_gauges_die_with_the_process(self, monkeypatch):
        self._reset(monkeypatch, enabled=False, multiproc=False).assert_not_called()

    def test_connected_to_the_worker_start(self):
        from celery.signals import worker_ready

        receivers = [receiver() for _, receiver in worker_ready.receivers]
        assert catalog_tasks.reset_gauges_while_switched_off in receivers

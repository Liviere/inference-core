"""Tests for the Celery factory helpers."""

from typing import Callable, List

from celery import Celery

from inference_core.celery.celery_main import (
    BaseTaskMixin,
    attach_base_task_class,
    create_celery_app,
)


def test_create_celery_app_extends_include_list() -> None:
    """Custom modules provided to the factory should be included."""
    module_name = "tests.fixtures.celery_ext.tasks"
    app = create_celery_app(include_modules=[module_name])
    assert module_name in app.conf.include


def test_create_celery_app_supports_autodiscover_callable(monkeypatch) -> None:
    """Additional autodiscover entries should be forwarded to Celery."""
    recorded: List[object] = []

    def fake_autodiscover(
        self: Celery,
        packages: object | None = None,
        *args,
        **kwargs,
    ) -> list[str]:
        recorded.append(packages)
        return []

    monkeypatch.setattr(Celery, "autodiscover_tasks", fake_autodiscover, raising=False)

    resolver: Callable[[], list[str]] = lambda: ["tests.fixtures.celery_ext"]
    create_celery_app(autodiscover=[resolver])

    assert any(
        isinstance(entry, list) and "inference_core.celery.tasks" in entry
        for entry in recorded
    )
    assert any(callable(entry) for entry in recorded)


def test_create_celery_app_merges_task_routes() -> None:
    """Extra task routes should be merged with defaults."""
    extra = {"tests.fixtures.celery_ext.ping": {"queue": "external"}}
    app = create_celery_app(extra_task_routes=extra)
    for key, value in extra.items():
        assert app.conf.task_routes[key] == value


def test_create_celery_app_merges_beat_schedule() -> None:
    """Periodic tasks provided externally should augment the schedule."""
    schedule = {
        "tests.custom-beat": {
            "task": "tests.fixtures.celery_ext.ping",
            "schedule": 60.0,
        }
    }
    app = create_celery_app(beat_schedule_overrides=schedule)
    for key, value in schedule.items():
        assert app.conf.beat_schedule[key] == value


def test_attach_base_task_class_returns_expected_type() -> None:
    """The helper should attach a subclass exposing mixin behaviour."""
    app = create_celery_app()
    base_cls = attach_base_task_class(app)

    assert issubclass(base_cls, BaseTaskMixin)
    assert base_cls.abstract is True
    assert app.Task is base_cls

    class _SampleTask(base_cls):
        name = "tests.sample-task"

    assert issubclass(_SampleTask, base_cls)


def test_model_catalog_task_is_included() -> None:
    """The catalog task module is part of every app the factory builds."""
    app = create_celery_app()
    assert "inference_core.celery.tasks.catalog_tasks" in app.conf.include


def test_model_catalog_is_scheduled_only_when_enabled() -> None:
    """Beat asks the catalog for due providers only with the flag on."""
    from inference_core.core.config import get_settings

    base = get_settings()
    off = create_celery_app(
        custom_settings=base.model_copy(update={"llm_model_catalog_enabled": False})
    )
    on = create_celery_app(
        custom_settings=base.model_copy(update={"llm_model_catalog_enabled": True})
    )

    assert "llm-catalog-refresh" not in off.conf.beat_schedule
    entry = on.conf.beat_schedule["llm-catalog-refresh"]
    assert entry["task"] == "llm.catalog_refresh"
    assert entry["schedule"] == 900.0
    assert entry["options"] == {"queue": "default", "expires": 900}
    # The schedule shared by every app is left as it was.
    assert "batch-poll" in on.conf.beat_schedule
    assert (
        "llm-catalog-refresh"
        not in create_celery_app(
            custom_settings=base.model_copy(update={"llm_model_catalog_enabled": False})
        ).conf.beat_schedule
    )


def test_model_catalog_schedule_can_be_overridden() -> None:
    """An application's own entry under the same key wins."""
    from inference_core.core.config import get_settings

    settings = get_settings().model_copy(update={"llm_model_catalog_enabled": True})
    override = {
        "llm-catalog-refresh": {"task": "llm.catalog_refresh", "schedule": 60.0}
    }

    app = create_celery_app(custom_settings=settings, beat_schedule_overrides=override)

    assert app.conf.beat_schedule["llm-catalog-refresh"]["schedule"] == 60.0

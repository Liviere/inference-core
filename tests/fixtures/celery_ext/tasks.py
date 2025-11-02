"""Celery tasks used to verify external module registration in tests."""

from celery import shared_task


@shared_task(name="tests.fixtures.celery_ext.ping")
def ping() -> str:
    """Return a simple response used in tests."""
    return "pong"

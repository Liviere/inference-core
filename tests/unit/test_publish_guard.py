"""No test publishes a Celery task to a real broker.

The guard in ``tests/conftest.py`` stops a publish to any transport but the
in-memory one, and names the test, the task and the queue. These tests
publish on apps of their own, with the broker given explicitly.
"""

import pytest
from celery import Celery

from tests.conftest import RealBrokerPublish


def _app(monkeypatch, broker: str) -> Celery:
    # Celery prefers the environment's broker to the one it is given.
    monkeypatch.delenv("CELERY_BROKER_URL", raising=False)
    monkeypatch.delenv("CELERY_RESULT_BACKEND", raising=False)
    return Celery("publish_guard", broker=broker, set_as_current=False)


def test_a_publish_to_redis_fails_with_the_tests_name(monkeypatch):
    # Nothing listens there: the guard refuses before kombu connects.
    app = _app(monkeypatch, "redis://127.0.0.1:1/0")

    with pytest.raises(RealBrokerPublish) as refused:
        app.send_task("email.send", kwargs={}, queue="mail")

    message = str(refused.value)
    assert "test_a_publish_to_redis_fails_with_the_tests_name" in message
    assert "'email.send'" in message
    assert "'mail'" in message


def test_the_in_memory_broker_is_allowed(monkeypatch):
    app = _app(monkeypatch, "memory://")

    result = app.send_task("email.send", kwargs={}, queue="mail")

    assert result.id


@pytest.mark.real_broker
def test_a_marked_test_reaches_the_broker(monkeypatch):
    # The guard lets it through; the made-up broker then refuses to connect.
    app = _app(monkeypatch, "redis://127.0.0.1:1/0")
    app.conf.broker_connection_retry = False
    app.conf.broker_connection_max_retries = 0

    with pytest.raises(Exception) as failed:
        app.send_task("email.send", kwargs={}, queue="mail", retry=False)

    assert not isinstance(failed.value, RealBrokerPublish)

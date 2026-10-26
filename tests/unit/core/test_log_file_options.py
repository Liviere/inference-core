"""Options of the JSON log files: who rotates them, and one file per worker.

Both are off by default, and then nothing changes: the process rotates its
own file at midnight and every worker writes ``logs/celery.log``.
"""

import logging
import logging.handlers
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from inference_core.celery import celery_main
from inference_core.core import logging_config


def _settings(**overrides):
    values = {
        "log_file_rotation": "time",
        "celery_log_file_per_node": False,
        "debug": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_the_settings_default_to_today_s_behaviour():
    from inference_core.core.config import Settings

    fields = Settings.model_fields
    assert fields["log_file_rotation"].default == "time"
    assert fields["celery_log_file_per_node"].default is False


def test_the_process_rotates_its_file_by_default(tmp_path):
    with patch.object(logging_config, "get_settings", return_value=_settings()):
        handler = logging_config._build_file_handler(str(tmp_path / "a.log"), "INFO")
    try:
        assert isinstance(handler, logging.handlers.TimedRotatingFileHandler)
        assert handler.when == "MIDNIGHT"
        assert handler.backupCount == 30
    finally:
        handler.close()


def test_external_rotation_reopens_a_moved_file(tmp_path):
    """logrotate moves the file; the next record goes to a new one."""
    path = tmp_path / "a.log"
    settings = _settings(log_file_rotation="external")
    with patch.object(logging_config, "get_settings", return_value=settings):
        handler = logging_config._build_file_handler(str(path), "INFO")
    log = logging.getLogger("test_external_rotation")
    log.propagate = False
    log.addHandler(handler)
    try:
        assert isinstance(handler, logging.handlers.WatchedFileHandler)
        assert not isinstance(handler, logging.handlers.TimedRotatingFileHandler)
        log.warning("before")
        path.rename(tmp_path / "a.log.1")
        log.warning("after")
        assert "before" in (tmp_path / "a.log.1").read_text()
        assert "after" in path.read_text()
        assert "before" not in path.read_text()
    finally:
        log.removeHandler(handler)
        handler.close()


@pytest.mark.parametrize(
    "per_node, node, expected",
    [
        (False, "mail-worker", "celery.log"),
        (True, "mail-worker", "celery-mail-worker.log"),
        (True, None, "celery.log"),
        (True, "odd/name x", "celery-odd_name_x.log"),
    ],
)
def test_celery_log_file_name(monkeypatch, per_node, node, expected):
    monkeypatch.setattr(celery_main, "_node_name", None)
    settings = _settings(celery_log_file_per_node=per_node)
    with patch.object(celery_main, "get_settings", return_value=settings):
        assert celery_main.celery_log_file_name(node) == expected


def test_the_worker_s_node_name_is_taken_at_worker_init(monkeypatch):
    """Pool processes forked later inherit it; the name is before the '@'."""
    monkeypatch.setattr(celery_main, "_node_name", None)
    sender = SimpleNamespace(hostname="mail-worker@3f2a9c", pool_cls="prefork")

    celery_main._on_worker_init(sender=sender)

    settings = _settings(celery_log_file_per_node=True)
    with patch.object(celery_main, "get_settings", return_value=settings):
        assert celery_main.celery_log_file_name() == "celery-mail-worker.log"


def test_setup_writes_to_the_node_s_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(celery_main, "_node_name", "threads-worker")
    settings = _settings(celery_log_file_per_node=True, log_file_rotation="external")
    before = list(logging.root.handlers)
    with (
        patch.object(celery_main, "get_settings", return_value=settings),
        patch.object(logging_config, "get_settings", return_value=settings),
    ):
        celery_main.setup_celery_file_logging()
    added = [h for h in logging.root.handlers if h not in before]
    try:
        assert len(added) == 1
        assert added[0].baseFilename == str(tmp_path / "logs" / "celery-threads-worker.log")
    finally:
        for handler in added:
            logging.root.removeHandler(handler)
            handler.close()

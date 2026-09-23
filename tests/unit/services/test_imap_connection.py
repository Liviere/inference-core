"""Which client ImapConnection opens for a host configuration."""

import pytest

from inference_core.core.email_config import (
    EmailHostConfig,
    ImapHostConfig,
    SmtpHostConfig,
)
from inference_core.services import imap_service
from inference_core.services.imap_service import ImapConnection


def _connection(**imap_overrides) -> ImapConnection:
    imap = ImapHostConfig(
        host="imap.example.com",
        username="test@example.com",
        password_env="TEST_PASSWORD",
        **imap_overrides,
    )
    smtp = SmtpHostConfig(
        host="smtp.example.com",
        port=465,
        use_ssl=True,
        username="test@example.com",
        from_email="test@example.com",
    )
    return ImapConnection(
        host_alias="primary",
        imap_config=imap,
        host_config=EmailHostConfig(smtp=smtp, imap=imap),
    )


@pytest.fixture
def opened(monkeypatch):
    seen = {}

    def recorder(name):
        def factory(host, port, **kwargs):
            seen.update(name=name, host=host, port=port, kwargs=kwargs)
            return object()

        return factory

    monkeypatch.setattr(imap_service, "PinnedIMAP4SSL", recorder("PinnedIMAP4SSL"))
    monkeypatch.setattr(imap_service, "PinnedIMAP4", recorder("PinnedIMAP4"))
    monkeypatch.setattr(imap_service.imaplib, "IMAP4_SSL", recorder("IMAP4_SSL"))
    monkeypatch.setattr(imap_service.imaplib, "IMAP4", recorder("IMAP4"))
    return seen


@pytest.mark.parametrize(
    ("use_ssl", "address", "expected"),
    [
        (True, "203.0.113.7", "PinnedIMAP4SSL"),
        (False, "203.0.113.7", "PinnedIMAP4"),
        (True, None, "IMAP4_SSL"),
        (False, None, "IMAP4"),
    ],
)
def test_opens_the_client_for_the_configuration(opened, use_ssl, address, expected):
    _connection(use_ssl=use_ssl, port=993, connect_address=address)._open()

    assert opened["name"] == expected
    assert (opened["host"], opened["port"]) == ("imap.example.com", 993)
    assert opened["kwargs"]["timeout"] == 30
    if address:
        assert opened["kwargs"]["pinned_address"] == address

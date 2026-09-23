"""SMTP/IMAP clients that dial a given address but keep TLS on the host name."""

import imaplib
import socket

import pytest

from inference_core.services.mail_transport import (
    PinnedIMAP4,
    PinnedIMAP4SSL,
    PinnedSMTP,
    PinnedSMTPSSL,
)


@pytest.fixture
def dialled(monkeypatch):
    """Capture the address a connection would have been opened to."""
    calls = []

    def fake_create_connection(address, *args, **kwargs):
        calls.append(address)
        return object()

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    return calls


class _FakeContext:
    """Stand-in for an ssl.SSLContext that records the SNI name."""

    def __init__(self):
        self.server_hostname = None

    def wrap_socket(self, sock, server_hostname=None):
        self.server_hostname = server_hostname
        return sock


def _bare(cls, **attrs):
    """Build an instance without running __init__ (which would connect)."""
    obj = cls.__new__(cls)
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


class TestPinnedSMTP:
    def test_dials_the_pinned_address(self, dialled):
        smtp = _bare(
            PinnedSMTP,
            _pinned_address="93.184.216.34",
            debuglevel=0,
            source_address=None,
        )
        smtp._get_socket("smtp.example.com", 587, 10)
        assert dialled == [("93.184.216.34", 587)]

    def test_ssl_dials_pinned_but_verifies_the_name(self, dialled):
        context = _FakeContext()
        smtp = _bare(
            PinnedSMTPSSL,
            _pinned_address="93.184.216.34",
            _host="smtp.example.com",
            context=context,
            debuglevel=0,
            source_address=None,
        )
        smtp._get_socket("smtp.example.com", 465, 10)
        assert dialled == [("93.184.216.34", 465)]
        assert context.server_hostname == "smtp.example.com"


class TestPinnedIMAP:
    def test_dials_the_pinned_address(self, dialled):
        imap = _bare(PinnedIMAP4, _pinned_address="93.184.216.34", port=143)
        imap._create_socket(10)
        assert dialled == [("93.184.216.34", 143)]

    def test_ssl_dials_pinned_but_verifies_the_name(self, dialled):
        context = _FakeContext()
        imap = _bare(
            PinnedIMAP4SSL,
            _pinned_address="93.184.216.34",
            host="imap.example.com",
            port=993,
            ssl_context=context,
        )
        imap._create_socket(10)
        assert dialled == [("93.184.216.34", 993)]
        assert context.server_hostname == "imap.example.com"

    def test_ssl_override_is_not_bypassed(self):
        """IMAP4_SSL calls IMAP4._create_socket directly, so a mixin would miss."""
        assert PinnedIMAP4SSL._create_socket is not imaplib.IMAP4_SSL._create_socket

    def test_zero_timeout_is_rejected(self, dialled):
        imap = _bare(PinnedIMAP4, _pinned_address="93.184.216.34", port=143)
        with pytest.raises(ValueError, match="Non-blocking socket"):
            imap._create_socket(0)
        assert dialled == []

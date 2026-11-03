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


class TestMailConnectionHook:
    """A hook is told of each connection core's mail services open."""

    @pytest.fixture
    def reported(self):
        from inference_core.services import mail_transport

        calls = []

        def hook(protocol, host, port):
            calls.append((protocol, host, port))

        mail_transport.register_mail_connection_hook(hook)
        yield calls
        mail_transport.unregister_mail_connection_hook(hook)

    def test_none_is_registered_by_default(self):
        from inference_core.services import mail_transport

        assert mail_transport._CONNECTION_HOOKS == []

    def test_an_imap_connection_is_reported_before_it_is_opened(
        self, reported, monkeypatch
    ):
        from inference_core.core.email_config import (
            EmailHostConfig,
            ImapHostConfig,
            SmtpHostConfig,
        )
        from inference_core.services import imap_service
        from inference_core.services.imap_service import ImapConnection

        def refused(*args, **kwargs):
            raise ConnectionRefusedError()

        monkeypatch.setattr(imap_service, "PinnedIMAP4SSL", refused)
        imap = ImapHostConfig(
            host="imap.example.com",
            port=993,
            username="test@example.com",
            connect_address="203.0.113.7",
        )
        smtp = SmtpHostConfig(
            host="smtp.example.com",
            port=465,
            username="test@example.com",
            from_email="test@example.com",
        )
        connection = ImapConnection(
            host_alias="primary",
            imap_config=imap,
            host_config=EmailHostConfig(smtp=smtp, imap=imap),
        )

        with pytest.raises(ConnectionRefusedError):
            connection._open()

        # The host name, not the address dialled; a failed one too.
        assert reported == [("imap", "imap.example.com", 993)]

    def test_an_smtp_connection_is_reported(self, reported, monkeypatch):
        from unittest.mock import MagicMock

        from inference_core.core.email_config import (
            EmailConfig,
            EmailHostConfig,
            EmailSettings,
            FullEmailConfig,
            SmtpHostConfig,
        )
        from inference_core.services import email_service
        from inference_core.services.email_service import EmailService

        smtp = SmtpHostConfig(
            host="smtp.example.com",
            port=587,
            use_ssl=False,
            username="test@example.com",
            password="secret",
            from_email="test@example.com",
            connect_address="203.0.113.7",
        )
        client = MagicMock()
        monkeypatch.setattr(email_service, "PinnedSMTP", client)
        config = FullEmailConfig(
            email=EmailConfig(
                default_host="primary",
                hosts={"primary": EmailHostConfig(smtp=smtp)},
            ),
            settings=EmailSettings(),
        )

        EmailService(config).send_email(
            to="recipient@example.com", subject="Hook", text="Body"
        )

        assert reported == [("smtp", "smtp.example.com", 587)]

    def test_a_failing_hook_does_not_fail_the_connection(self, caplog):
        from inference_core.services import mail_transport

        def broken(protocol, host, port):
            raise RuntimeError("secret detail")

        mail_transport.register_mail_connection_hook(broken)
        try:
            mail_transport.report_mail_connection("imap", "imap.example.com", 993)
        finally:
            mail_transport.unregister_mail_connection_hook(broken)

        assert "RuntimeError" in caplog.text
        assert "secret detail" not in caplog.text

    def test_registering_twice_calls_once_and_unknown_ones_are_refused(self):
        from inference_core.services import mail_transport

        calls = []

        def hook(protocol, host, port):
            calls.append(protocol)

        mail_transport.register_mail_connection_hook(hook)
        mail_transport.register_mail_connection_hook(hook)
        try:
            mail_transport.report_mail_connection("smtp", "smtp.example.com", 465)
        finally:
            mail_transport.unregister_mail_connection_hook(hook)
        mail_transport.unregister_mail_connection_hook(hook)

        assert calls == ["smtp"]
        with pytest.raises(TypeError):
            mail_transport.register_mail_connection_hook("not callable")

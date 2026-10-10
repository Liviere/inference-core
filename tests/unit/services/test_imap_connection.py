"""Which client ImapConnection opens for a host configuration."""

import ssl

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


@pytest.mark.parametrize("address", ["203.0.113.7", None])
def test_tls_checks_the_certificate_and_the_name(opened, address):
    """imaplib's own default context verifies nothing."""
    _connection(use_ssl=True, port=993, connect_address=address)._open()

    context = opened["kwargs"]["ssl_context"]
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_verification_can_be_turned_off_per_host(opened):
    _connection(use_ssl=True, port=993, verify_hostname=False)._open()

    context = opened["kwargs"]["ssl_context"]
    assert context.verify_mode == ssl.CERT_NONE
    assert context.check_hostname is False


# ── Signing in ────────────────────────────────────────────────────────


class _Client:
    """An IMAP client whose sign-in ends the way a test says."""

    def __init__(self, failure=None):
        self.failure = failure
        self.sign_ins = 0
        self.closed = False

    def login(self, _user, _password):
        self.sign_ins += 1
        if self.failure is not None:
            raise self.failure

    def authenticate(self, _mechanism, _answer):
        self.login(None, None)

    def noop(self):
        return "OK", [b""]

    def shutdown(self):
        self.closed = True


def _signing_in(monkeypatch, *clients, **imap_overrides) -> ImapConnection:
    connection = _connection(password="secret", **imap_overrides)
    queue = list(clients)
    monkeypatch.setattr(ImapConnection, "_open", lambda self: queue.pop(0))
    return connection


def _refused(text):
    return imap_service.imaplib.IMAP4.error(text)


@pytest.mark.parametrize(
    ("answer", "code", "rejects"),
    [
        (b"[AUTHENTICATIONFAILED] Authentication failed.", "AUTHENTICATIONFAILED", True),
        (b"[AUTHORIZATIONFAILED] Not allowed.", "AUTHORIZATIONFAILED", True),
        (b"[EXPIRED] Password expired.", "EXPIRED", True),
        (b"LOGIN failed.", None, True),
        # What a server says about itself is not about the credentials.
        (b"[UNAVAILABLE] Temporary authentication failure.", "UNAVAILABLE", False),
        (b"[LIMIT] Too many login attempts.", "LIMIT", False),
        (b"[INUSE] Mailbox is busy.", "INUSE", False),
        (b"[ALERT] Please log in via your web browser.", "ALERT", False),
        (b"Too many simultaneous connections.", None, False),
        (b"Server busy, try again later.", None, False),
    ],
)
def test_a_refused_sign_in_says_what_was_refused(monkeypatch, answer, code, rejects):
    client = _Client(_refused(answer))
    connection = _signing_in(monkeypatch, client)

    with pytest.raises(imap_service.ImapAuthenticationError) as raised:
        connection.connect()

    assert raised.value.response_code == code
    assert raised.value.rejects_credentials is rejects
    # The wording other code matches on is kept.
    assert raised.value.message.startswith("Authentication failed:")


def test_a_refused_xoauth2_sign_in_is_a_refusal_too(monkeypatch):
    client = _Client(_refused("[AUTHENTICATIONFAILED] Invalid credentials (Failure)"))
    connection = _signing_in(
        monkeypatch, client, auth_type="oauth", access_token="token"
    )

    with pytest.raises(imap_service.ImapAuthenticationError) as raised:
        connection.connect()

    assert raised.value.rejects_credentials is True


@pytest.mark.parametrize(
    "failure",
    [
        imap_service.imaplib.IMAP4.abort("socket error: EOF"),
        imap_service.imaplib.IMAP4.error("LOGIN command error: BAD [b'Syntax']"),
        imap_service.imaplib.IMAP4.error(
            "command LOGIN illegal in state LOGOUT, only allowed in states NONAUTH"
        ),
        TimeoutError("timed out"),
    ],
)
def test_a_sign_in_that_broke_is_not_a_refusal(monkeypatch, failure):
    connection = _signing_in(monkeypatch, _Client(failure))

    with pytest.raises(imap_service.ImapConnectionError) as raised:
        connection.connect()

    assert not isinstance(raised.value, imap_service.ImapAuthenticationError)
    assert raised.value.message.startswith("Connection error:")
    assert raised.value.original_error is failure


def test_a_refused_greeting_is_not_a_refusal_of_the_sign_in(monkeypatch):
    connection = _connection(password="secret")
    greeting = _refused(b"* BYE Too many connections from your IP")

    def refuse(self):
        raise greeting

    monkeypatch.setattr(ImapConnection, "_open", refuse)

    with pytest.raises(imap_service.ImapConnectionError) as raised:
        connection.connect()

    assert not isinstance(raised.value, imap_service.ImapAuthenticationError)
    assert connection._connection is None


def test_a_connection_that_did_not_sign_in_is_not_kept(monkeypatch):
    """It passes NOOP, so a kept one was taken for healthy and never signed in."""
    refused = _Client(_refused(b"[AUTHENTICATIONFAILED] Authentication failed."))
    accepted = _Client()
    connection = _signing_in(monkeypatch, refused, accepted)

    with pytest.raises(imap_service.ImapAuthenticationError):
        connection.connect()

    assert refused.closed is True
    assert connection._connection is None

    connection.connect()

    assert accepted.sign_ins == 1
    assert connection._connection is accepted

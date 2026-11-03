"""SMTP/IMAP clients that connect to a given address instead of resolving the host.

When a mail host comes from user input, a caller may resolve it, check the
addresses (e.g. refuse private ones) and then need the connection to land on
exactly one of those addresses. Opening the connection by name resolves it a
second time, which is a DNS-rebinding window. These classes dial the address
they are given and keep the host name for everything else.

TLS stays on the real name in both directions:

* ``SMTP_SSL._get_socket`` wraps whatever its parent returns and passes
  ``self._host`` as ``server_hostname``, so rewriting the address in the parent
  call leaves SNI and certificate verification on the host name; STARTTLS does
  the same.
* ``IMAP4_SSL._create_socket`` calls ``IMAP4._create_socket`` explicitly rather
  than through the MRO, so a mixin would be skipped — it is replaced outright
  below, keeping ``server_hostname=self.host``.

A caller may also want to know which mail servers a process reached, for
instance to keep a record of its outbound connections. A hook registered with
:func:`register_mail_connection_hook` is told the protocol, host and port of
each connection core's mail services open. None is registered by default.
"""

import imaplib
import logging
import smtplib
import socket
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

#: ``hook(protocol, host, port)``; ``protocol`` is ``"smtp"`` or ``"imap"``.
MailConnectionHook = Callable[[str, str, int], None]

_CONNECTION_HOOKS: List[MailConnectionHook] = []


def register_mail_connection_hook(hook: MailConnectionHook) -> None:
    """Call ``hook(protocol, host, port)`` before every SMTP or IMAP
    connection opened by ``EmailService`` and ``ImapConnection``.

    ``host`` is the configured host name, not the address dialled; nothing of
    the credentials or the mail is passed. The hook runs in the caller's
    thread and context, synchronously, just before the socket is opened, so
    a connection that then fails is reported too. Registering the same hook
    again is a no-op.
    """
    if not callable(hook):
        raise TypeError("A mail connection hook must be callable")
    if hook not in _CONNECTION_HOOKS:
        _CONNECTION_HOOKS.append(hook)


def unregister_mail_connection_hook(hook: MailConnectionHook) -> None:
    """Forget a hook (a no-op for one never registered)."""
    try:
        _CONNECTION_HOOKS.remove(hook)
    except ValueError:
        pass


def report_mail_connection(protocol: str, host: str, port: int) -> None:
    """Tell every hook of a connection about to be opened.

    A failing hook never fails the connection; its error is logged by type.
    """
    for hook in list(_CONNECTION_HOOKS):
        try:
            hook(protocol, host, port)
        except Exception as exc:
            logger.warning("Mail connection hook failed: %s", type(exc).__name__)


class _PinnedSMTPMixin:
    """Dial ``_pinned_address`` instead of resolving the hostname again."""

    _pinned_address: str

    def _get_socket(self, host: str, port: int, timeout: float):  # type: ignore[override]
        return super()._get_socket(self._pinned_address, port, timeout)


class PinnedSMTP(_PinnedSMTPMixin, smtplib.SMTP):
    def __init__(self, host: str, port: int, *, pinned_address: str, **kwargs: Any):
        self._pinned_address = pinned_address
        super().__init__(host, port, **kwargs)


class PinnedSMTPSSL(_PinnedSMTPMixin, smtplib.SMTP_SSL):
    def __init__(self, host: str, port: int, *, pinned_address: str, **kwargs: Any):
        self._pinned_address = pinned_address
        super().__init__(host, port, **kwargs)


def _connect_to_pinned(address: str, port: int, timeout: Optional[float]):
    if timeout is not None and not timeout:
        raise ValueError("Non-blocking socket (timeout=0) is not supported")
    if timeout is not None:
        return socket.create_connection((address, port), timeout)
    return socket.create_connection((address, port))


class PinnedIMAP4(imaplib.IMAP4):
    def __init__(
        self,
        host: str,
        port: int,
        *,
        pinned_address: str,
        timeout: Optional[float] = None,
    ):
        self._pinned_address = pinned_address
        super().__init__(host, port, timeout)

    def _create_socket(self, timeout):  # type: ignore[override]
        return _connect_to_pinned(self._pinned_address, self.port, timeout)


class PinnedIMAP4SSL(imaplib.IMAP4_SSL):
    def __init__(
        self,
        host: str,
        port: int,
        *,
        pinned_address: str,
        ssl_context=None,
        timeout: Optional[float] = None,
    ):
        self._pinned_address = pinned_address
        super().__init__(host, port, ssl_context=ssl_context, timeout=timeout)

    def _create_socket(self, timeout):  # type: ignore[override]
        sock = _connect_to_pinned(self._pinned_address, self.port, timeout)
        return self.ssl_context.wrap_socket(sock, server_hostname=self.host)

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
"""

import imaplib
import smtplib
import socket
from typing import Any, Optional


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

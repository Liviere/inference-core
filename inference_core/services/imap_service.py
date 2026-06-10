"""
IMAP Service

Production-grade IMAP email reading service with support for:
- Secure SSL/TLS connections
- Multiple mailbox folder navigation
- Message parsing (multipart, encodings)
- Comprehensive logging and error handling

This service complements EmailService (SMTP) for bidirectional email operations.
"""

import base64
import email
import imaplib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from email.header import decode_header, make_header
from email.message import Message
from email.utils import parseaddr, parsedate_to_datetime
from enum import Enum as PyEnum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel

from inference_core.core.email_config import (
    EmailHostConfig,
    FullEmailConfig,
    ImapHostConfig,
    get_email_config,
)

logger = logging.getLogger(__name__)


class ImapConnectionError(Exception):
    """Exception raised when IMAP connection fails."""

    def __init__(
        self, message: str, host_alias: str, original_error: Optional[Exception] = None
    ):
        self.message = message
        self.host_alias = host_alias
        self.original_error = original_error
        super().__init__(f"IMAP connection failed for {host_alias}: {message}")


class ImapReadError(Exception):
    """Exception raised when reading emails fails."""

    def __init__(
        self,
        message: str,
        host_alias: str,
        folder: str,
        original_error: Optional[Exception] = None,
    ):
        self.message = message
        self.host_alias = host_alias
        self.folder = folder
        self.original_error = original_error
        super().__init__(f"IMAP read failed for {host_alias}/{folder}: {message}")


class EmailMessage(BaseModel):
    """Parsed email message data.

    Contains structured representation of an email for agent consumption.
    All fields are extracted and decoded from raw IMAP message.
    """

    uid: str
    message_id: str
    subject: str
    from_address: str
    from_name: Optional[str] = None
    to_addresses: List[str] = []
    cc_addresses: List[str] = []
    date: Optional[datetime] = None
    body_text: str = ""
    body_html: Optional[str] = None
    has_attachments: bool = False
    attachment_names: List[str] = []
    folder: str = "INBOX"
    is_read: bool = False


class SpecialFolder(str, PyEnum):
    """Logical (provider-agnostic) mailbox roles.

    Used to resolve the concrete folder name on a given server via
    RFC 6154 ``SPECIAL-USE`` attributes with name-based fallbacks.
    """

    TRASH = "trash"
    JUNK = "junk"
    ARCHIVE = "archive"
    SENT = "sent"
    DRAFTS = "drafts"
    ALL = "all"


class FolderInfo(BaseModel):
    """A mailbox folder with its (lower-cased, ``\\``-stripped) IMAP attributes."""

    name: str
    attributes: List[str] = []


# RFC 6154 SPECIAL-USE attribute (without the leading backslash) per role.
_SPECIAL_USE_ATTR: Dict[SpecialFolder, str] = {
    SpecialFolder.TRASH: "trash",
    SpecialFolder.JUNK: "junk",
    SpecialFolder.ARCHIVE: "archive",
    SpecialFolder.SENT: "sent",
    SpecialFolder.DRAFTS: "drafts",
    SpecialFolder.ALL: "all",
}

# Case-insensitive name fallbacks (ordered by likelihood) covering Gmail,
# Outlook/Office365, Dovecot/cPanel (``INBOX.``-prefixed) and generic IMAP.
_SPECIAL_NAME_FALLBACKS: Dict[SpecialFolder, List[str]] = {
    SpecialFolder.TRASH: [
        "Trash",
        "Deleted",
        "Deleted Items",
        "Deleted Messages",
        "[Gmail]/Trash",
        "[Google Mail]/Trash",
        "INBOX.Trash",
    ],
    SpecialFolder.JUNK: [
        "Junk",
        "Spam",
        "Junk E-mail",
        "Junk Email",
        "Bulk Mail",
        "[Gmail]/Spam",
        "[Google Mail]/Spam",
        "INBOX.Junk",
        "INBOX.Spam",
    ],
    SpecialFolder.ARCHIVE: [
        "Archive",
        "Archives",
        "[Gmail]/All Mail",
        "[Google Mail]/All Mail",
        "INBOX.Archive",
    ],
    SpecialFolder.SENT: [
        "Sent",
        "Sent Items",
        "Sent Mail",
        "[Gmail]/Sent Mail",
        "[Google Mail]/Sent Mail",
        "INBOX.Sent",
    ],
    SpecialFolder.DRAFTS: [
        "Drafts",
        "[Gmail]/Drafts",
        "[Google Mail]/Drafts",
        "INBOX.Drafts",
    ],
    SpecialFolder.ALL: [
        "[Gmail]/All Mail",
        "[Google Mail]/All Mail",
        "All Mail",
        "Archive",
    ],
}


def _modified_b64encode(text: str) -> str:
    """Encode a unicode run as IMAP modified BASE64 (UTF-16BE, ``/``→``,``)."""
    raw = base64.b64encode(text.encode("utf-16-be")).decode("ascii")
    return raw.rstrip("=").replace("/", ",")


def _modified_b64decode(chunk: str) -> str:
    """Decode an IMAP modified-BASE64 run back to unicode."""
    data = chunk.replace(",", "/")
    data += "=" * ((-len(data)) % 4)
    return base64.b64decode(data).decode("utf-16-be")


def imap_utf7_encode(name: str) -> str:
    """Encode a folder name to IMAP modified UTF-7 (RFC 3501 §5.1.3)."""
    out: List[str] = []
    buf: List[str] = []

    def _flush() -> None:
        if buf:
            out.append("&" + _modified_b64encode("".join(buf)) + "-")
            buf.clear()

    for ch in name:
        if 0x20 <= ord(ch) <= 0x7E:
            _flush()
            out.append("&-" if ch == "&" else ch)
        else:
            buf.append(ch)
    _flush()
    return "".join(out)


def imap_utf7_decode(name: str) -> str:
    """Decode an IMAP modified-UTF-7 folder name to unicode."""
    out: List[str] = []
    i, n = 0, len(name)
    while i < n:
        ch = name[i]
        if ch == "&":
            end = name.find("-", i)
            if end == -1:
                out.append(name[i:])
                break
            chunk = name[i + 1 : end]
            out.append("&" if chunk == "" else _modified_b64decode(chunk))
            i = end + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _quote_mailbox(name: str) -> str:
    """Modified-UTF-7 encode and double-quote a mailbox name for IMAP commands."""
    encoded = imap_utf7_encode(name)
    escaped = encoded.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + escaped + '"'


# Parses a single IMAP LIST response line: ``(\attrs) "sep" name``.
_LIST_LINE_RE = re.compile(
    rb'^\((?P<attrs>[^)]*)\)\s+(?P<sep>"[^"]*"|NIL)\s+(?P<name>.*)$'
)


@dataclass
class ImapConnection:
    """Manages IMAP connection state for a single host.

    Handles connection lifecycle, reconnection, and folder selection.
    Designed to be used as context manager or explicitly closed.
    """

    host_alias: str
    imap_config: ImapHostConfig
    host_config: EmailHostConfig
    _connection: Optional[imaplib.IMAP4_SSL] = field(default=None, repr=False)
    _current_folder: Optional[str] = field(default=None, repr=False)
    _handled_uids: Set[bytes] = field(default_factory=set, repr=False)

    def connect(self) -> None:
        """Establish IMAP connection with authentication."""
        if self._connection:
            try:
                self._connection.noop()
                return  # Already connected and healthy
            except imaplib.IMAP4.error:
                logger.warning(
                    "[%s] Lost IMAP connection, reconnecting...", self.host_alias
                )
                self._connection = None

        password = self.imap_config.get_password()
        if self.imap_config.auth_type != "oauth" and not password:
            raise ImapConnectionError(
                f"Password not found in env var: {self.imap_config.password_env}",
                self.host_alias,
            )

        try:
            logger.info(
                "[%s] Connecting to IMAP %s:%d",
                self.host_alias,
                self.imap_config.host,
                self.imap_config.port,
            )

            if self.imap_config.use_ssl:
                self._connection = imaplib.IMAP4_SSL(
                    self.imap_config.host,
                    self.imap_config.port,
                    timeout=self.imap_config.timeout,
                )
            else:
                self._connection = imaplib.IMAP4(
                    self.imap_config.host,
                    self.imap_config.port,
                    timeout=self.imap_config.timeout,
                )

            if self.imap_config.auth_type == "oauth" and self.imap_config.access_token:
                auth_string = f"user={self.imap_config.username}\x01auth=Bearer {self.imap_config.access_token}\x01\x01"
                self._connection.authenticate("XOAUTH2", lambda x: auth_string)
            else:
                self._connection.login(self.imap_config.username, password)
            logger.info("[%s] IMAP login successful", self.host_alias)

        except imaplib.IMAP4.error as e:
            raise ImapConnectionError(f"Authentication failed: {e}", self.host_alias, e)
        except Exception as e:
            raise ImapConnectionError(f"Connection error: {e}", self.host_alias, e)

    def disconnect(self) -> None:
        """Close IMAP connection gracefully."""
        if self._connection:
            try:
                self._connection.logout()
            except imaplib.IMAP4.error:
                pass
            finally:
                self._connection = None
                self._current_folder = None
            logger.debug("[%s] IMAP disconnected", self.host_alias)

    def select_folder(self, folder: Optional[str] = None) -> int:
        """Select mailbox folder and return message count.

        Args:
            folder: Folder name (uses default_folder if None)

        Returns:
            Number of messages in folder
        """
        self.connect()  # Ensure connected
        folder = folder or self.imap_config.default_folder

        if self._current_folder == folder:
            # Refresh folder state
            self._connection.noop()
        else:
            typ, data = self._connection.select(folder)
            if typ != "OK":
                raise ImapReadError(
                    f"Failed to select folder: {data}",
                    self.host_alias,
                    folder,
                )
            self._current_folder = folder

        # Get message count
        typ, data = self._connection.search(None, "ALL")
        if typ == "OK" and data[0]:
            return len(data[0].split())
        return 0

    def __enter__(self) -> "ImapConnection":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()


class ImapService:
    """IMAP service for reading emails.

    Provides high-level operations for fetching and parsing emails
    across multiple configured IMAP hosts.
    """

    def __init__(self, config: Optional[FullEmailConfig] = None):
        """Initialize IMAP service.

        Args:
            config: Email configuration (loads from file if not provided)
        """
        self.config = config or get_email_config()
        self._connections: Dict[str, ImapConnection] = {}

    def _get_connection(self, host_alias: Optional[str] = None) -> ImapConnection:
        """Get or create IMAP connection for host.

        Args:
            host_alias: Host alias (uses default_host if None)

        Returns:
            ImapConnection instance

        Raises:
            ValueError: If host not found or IMAP not configured
        """
        alias = host_alias or self.config.email.default_host
        host_config = self.config.email.get_host_config(alias)

        if not host_config.has_imap():
            raise ValueError(f"IMAP not configured for host '{alias}'")

        if alias not in self._connections:
            self._connections[alias] = ImapConnection(
                host_alias=alias,
                imap_config=host_config.imap,
                host_config=host_config,
            )

        return self._connections[alias]

    def fetch_unseen_emails(
        self,
        host_alias: Optional[str] = None,
        folder: Optional[str] = None,
        limit: int = 10,
        mark_as_read: bool = False,
    ) -> List[EmailMessage]:
        """Fetch unread emails from mailbox.

        Args:
            host_alias: Host alias (uses default if None)
            folder: Folder to read from (uses default_folder if None)
            limit: Maximum number of emails to fetch
            mark_as_read: Whether to mark fetched emails as read

        Returns:
            List of parsed EmailMessage objects
        """
        conn = self._get_connection(host_alias)
        alias = host_alias or self.config.email.default_host

        try:
            conn.select_folder(folder)
            folder_name = folder or conn.imap_config.default_folder

            # Search for unseen messages
            typ, data = conn._connection.uid("SEARCH", None, "UNSEEN")
            if typ != "OK":
                raise ImapReadError("Search failed", alias, folder_name)

            uids = data[0].split() if data[0] else []
            uids = uids[-limit:]  # Take most recent

            messages = []
            for uid in uids:
                try:
                    msg = self._fetch_message(conn, uid, folder_name)
                    if msg:
                        msg.is_read = False
                        messages.append(msg)

                        if mark_as_read:
                            conn._connection.uid("STORE", uid, "+FLAGS", "\\Seen")

                except Exception as e:
                    logger.error(
                        "[%s] Failed to fetch UID %s: %s", alias, uid.decode(), e
                    )
                    continue

            logger.info(
                "[%s] Fetched %d unseen emails from %s",
                alias,
                len(messages),
                folder_name,
            )
            return messages

        except ImapReadError:
            raise
        except Exception as e:
            raise ImapReadError(str(e), alias, folder or "INBOX", e)

    def fetch_emails(
        self,
        host_alias: Optional[str] = None,
        folder: Optional[str] = None,
        limit: int = 10,
        search_criteria: str = "ALL",
    ) -> List[EmailMessage]:
        """Fetch emails matching search criteria.

        Args:
            host_alias: Host alias (uses default if None)
            folder: Folder to read from
            limit: Maximum number of emails to fetch
            search_criteria: IMAP search criteria (e.g., "ALL", "UNSEEN", "FROM sender@example.com")

        Returns:
            List of parsed EmailMessage objects
        """
        conn = self._get_connection(host_alias)
        alias = host_alias or self.config.email.default_host

        try:
            conn.select_folder(folder)
            folder_name = folder or conn.imap_config.default_folder

            typ, data = conn._connection.uid("SEARCH", None, search_criteria)
            if typ != "OK":
                raise ImapReadError(
                    f"Search failed: {search_criteria}", alias, folder_name
                )

            uids = data[0].split() if data[0] else []
            uids = uids[-limit:]  # Take most recent

            messages = []
            for uid in uids:
                try:
                    msg = self._fetch_message(conn, uid, folder_name)
                    if msg:
                        messages.append(msg)
                except Exception as e:
                    logger.error(
                        "[%s] Failed to fetch UID %s: %s", alias, uid.decode(), e
                    )

            logger.info(
                "[%s] Fetched %d emails from %s (criteria: %s)",
                alias,
                len(messages),
                folder_name,
                search_criteria[:50],
            )
            return messages

        except ImapReadError:
            raise
        except Exception as e:
            raise ImapReadError(str(e), alias, folder or "INBOX", e)

    def get_email_by_uid(
        self,
        host_alias: Optional[str] = None,
        folder: Optional[str] = None,
        uid: str = "",
    ) -> Optional[EmailMessage]:
        """Fetch a single email by UID.

        Args:
            host_alias: Host alias (uses default if None)
            folder: Folder to read from
            uid: UID of the email to fetch
        Returns:
            Parsed EmailMessage object or None if not found
        """
        if not uid:
            return None

        conn = self._get_connection(host_alias)
        alias = host_alias or self.config.email.default_host

        try:
            conn.select_folder(folder)
            folder_name = folder or conn.imap_config.default_folder
            return self._fetch_message(conn, uid.encode(), folder_name)

        except ImapReadError:
            raise
        except Exception as e:
            raise ImapReadError(str(e), alias, folder or "INBOX", e)

    def get_folder_uid_status(
        self,
        host_alias: Optional[str] = None,
        folder: Optional[str] = None,
    ) -> tuple[int, int]:
        """Return ``(uidnext, uidvalidity)`` for a folder without fetching bodies.

        ``UIDNEXT`` is the UID that will be assigned to the next message, so it
        serves as a high-watermark: any existing message has ``uid < uidnext``.
        ``UIDVALIDITY`` lets callers detect server-side UID resets.

        Args:
            host_alias: Host alias (uses default if None)
            folder: Folder to inspect (uses default_folder if None)

        Returns:
            Tuple of (uidnext, uidvalidity).
        """
        conn = self._get_connection(host_alias)
        alias = host_alias or self.config.email.default_host
        folder_name = folder or conn.imap_config.default_folder

        try:
            # SELECT exposes UIDNEXT / UIDVALIDITY as untagged responses.
            conn.select_folder(folder)
            uidnext = self._read_int_response(conn, "UIDNEXT")
            uidvalidity = self._read_int_response(conn, "UIDVALIDITY")

            # Fallback to STATUS if SELECT did not surface the values.
            if uidnext is None or uidvalidity is None:
                typ, data = conn._connection.status(
                    folder_name, "(UIDNEXT UIDVALIDITY)"
                )
                if typ == "OK" and data and data[0]:
                    text = data[0].decode() if isinstance(data[0], bytes) else data[0]
                    next_match = re.search(r"UIDNEXT\s+(\d+)", text)
                    valid_match = re.search(r"UIDVALIDITY\s+(\d+)", text)
                    if next_match:
                        uidnext = int(next_match.group(1))
                    if valid_match:
                        uidvalidity = int(valid_match.group(1))

            if uidnext is None or uidvalidity is None:
                raise ImapReadError(
                    "UIDNEXT/UIDVALIDITY unavailable", alias, folder_name
                )

            return uidnext, uidvalidity

        except ImapReadError:
            raise
        except Exception as e:
            raise ImapReadError(str(e), alias, folder_name, e)

    @staticmethod
    def _read_int_response(conn: ImapConnection, key: str) -> Optional[int]:
        """Read an integer untagged IMAP response value (e.g. UIDNEXT)."""
        try:
            typ, data = conn._connection.response(key)
        except Exception:
            return None
        if typ != key or not data or data[0] is None:
            return None
        value = data[0]
        if isinstance(value, bytes):
            value = value.decode()
        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else None

    def _fetch_message(
        self, conn: ImapConnection, uid: bytes, folder: str
    ) -> Optional[EmailMessage]:
        """Fetch and parse a single message by UID."""
        typ, data = conn._connection.uid("FETCH", uid, "(BODY.PEEK[])")
        if typ != "OK" or not data or not data[0]:
            return None

        # Data format is slightly different for UID FETCH
        # data[0] is tuple (b'SEQ (UID 123 RFC822 {size}', b'content')
        # or b'SEQ (UID 123 RFC822 {size})' if content is empty?
        # Usually it's a list containing a tuple.

        raw_msg = None
        for item in data:
            if isinstance(item, tuple):
                raw_msg = item[1]
                break

        if raw_msg is None:
            return None

        msg = email.message_from_bytes(raw_msg)

        return self._parse_message(msg, uid.decode(), folder)

    def _parse_message(self, msg: Message, uid: str, folder: str) -> EmailMessage:
        """Parse raw email message into EmailMessage."""
        # Parse headers
        subject = self._decode_header(msg.get("Subject", "(no subject)"))
        from_raw = msg.get("From", "")
        from_name, from_addr = parseaddr(from_raw)
        from_name = self._decode_header(from_name) if from_name else None

        # Parse recipients
        to_raw = msg.get("To", "")
        to_addresses = [addr.strip() for addr in to_raw.split(",") if addr.strip()]

        cc_raw = msg.get("Cc", "")
        cc_addresses = [addr.strip() for addr in cc_raw.split(",") if addr.strip()]

        # Parse date
        date_str = msg.get("Date")
        date = None
        if date_str:
            try:
                date = parsedate_to_datetime(date_str)
            except ValueError, TypeError:
                pass

        # Parse body
        body_text, body_html = self._extract_body(msg)

        # Check attachments
        attachment_names = []
        has_attachments = False
        if msg.is_multipart():
            for part in msg.walk():
                disposition = part.get("Content-Disposition", "")
                if "attachment" in disposition:
                    has_attachments = True
                    filename = part.get_filename()
                    if filename:
                        attachment_names.append(self._decode_header(filename))

        return EmailMessage(
            uid=uid,
            message_id=msg.get("Message-ID", uid),
            subject=subject,
            from_address=from_addr,
            from_name=from_name,
            to_addresses=to_addresses,
            cc_addresses=cc_addresses,
            date=date,
            body_text=body_text,
            body_html=body_html,
            has_attachments=has_attachments,
            attachment_names=attachment_names,
            folder=folder,
        )

    def _extract_body(self, msg: Message) -> tuple[str, Optional[str]]:
        """Extract text and HTML body from message."""
        body_text = ""
        body_html = None

        if msg.is_multipart():
            for part in msg.walk():
                # skip multipart container parts
                if part.is_multipart():
                    continue

                content_type = part.get_content_type()
                disposition = str(part.get_content_disposition() or "").lower()

                # Some email clients put the body in 'inline' disposition
                if disposition == "attachment":
                    continue

                if content_type == "text/plain":
                    text = self._decode_payload(part)
                    if text and not body_text:
                        body_text = text
                elif content_type == "text/html":
                    html = self._decode_payload(part)
                    if html and not body_html:
                        body_html = html
        else:
            # Single part message
            content_type = msg.get_content_type()
            payload = self._decode_payload(msg)
            if content_type == "text/html":
                body_html = payload
            else:
                body_text = payload

        # Final check 1: if is_multipart was True but no sub-parts were processed
        # (e.g. broken boundary parsing), try to decode the whole message as a fallback
        if msg.is_multipart() and not body_text and not body_html:
            fallback_payload = self._decode_payload(msg)
            if fallback_payload:
                if (
                    "<html>" in fallback_payload.lower()
                    or "<body>" in fallback_payload.lower()
                ):
                    body_html = fallback_payload
                else:
                    body_text = fallback_payload

        # Final check 2: if we have HTML but no text, try to generate text from HTML
        if not body_text.strip() and body_html:
            try:
                # Very simple HTML to text conversion for LLM consumption
                text = re.sub(
                    r"<(script|style)[^>]*>.*?</\1>",
                    "",
                    body_html,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
                text = re.sub(
                    r"</(p|div|h1|h2|h3|h4|h5|h6)>", "\n", text, flags=re.IGNORECASE
                )
                text = re.sub(r"<[^>]+>", "", text)
                # Decode HTML entities (basic)
                from html import unescape

                body_text = unescape(text).strip()
            except Exception as e:
                logger.warning("Failed to convert HTML to text: %s", e)

        return body_text, body_html

    def _decode_payload(self, part: Message) -> str:
        """Decode message payload with charset handling."""
        try:
            # get_payload(decode=True) handles Content-Transfer-Encoding (base64, etc.)
            payload = part.get_payload(decode=True)

            if payload is None:
                # This can happen if it's a multipart part or parsing failed
                raw_payload = part.get_payload()
                if isinstance(raw_payload, list):
                    return ""  # It's a multipart container
                return str(raw_payload) if raw_payload else ""

            if isinstance(payload, str):
                return payload

            # At this point payload is bytes, need to decode using charset
            charset = part.get_content_charset() or "utf-8"
            try:
                return payload.decode(charset, errors="replace")
            except LookupError, UnicodeDecodeError:
                # Fallback to utf-8 if specified charset fails
                return payload.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(
                "Failed to decode payload for part %s: %s", part.get_content_type(), e
            )
            return ""

    def _decode_header(self, value: str) -> str:
        """Decode MIME-encoded header value."""
        if not value:
            return ""
        try:
            return str(make_header(decode_header(value)))
        except Exception:
            return value

    def list_folders(self, host_alias: Optional[str] = None) -> List[str]:
        """List available mailbox folder names (modified-UTF-7 decoded)."""
        return [f.name for f in self.list_folders_with_attributes(host_alias)]

    def list_folders_with_attributes(
        self, host_alias: Optional[str] = None
    ) -> List[FolderInfo]:
        """List folders with their IMAP attributes (incl. RFC 6154 special-use).

        Attributes are returned lower-cased and without the leading backslash
        (e.g. ``\\Trash`` -> ``trash``). Folder names are decoded from IMAP
        modified UTF-7.
        """
        conn = self._get_connection(host_alias)
        conn.connect()

        typ, data = conn._connection.list()
        if typ != "OK":
            return []

        folders: List[FolderInfo] = []
        for item in data:
            if not isinstance(item, (bytes, bytearray)):
                # Some servers return tuples for literal folder names.
                if isinstance(item, tuple) and item and isinstance(item[0], bytes):
                    item = item[0]
                else:
                    continue
            match = _LIST_LINE_RE.match(bytes(item))
            if not match:
                continue
            try:
                attrs_raw = match.group("attrs").decode("ascii", "ignore")
                attributes = [
                    a.lstrip("\\").lower() for a in attrs_raw.split() if a.strip()
                ]
                name_raw = match.group("name").decode("ascii", "ignore").strip()
                if name_raw.startswith('"') and name_raw.endswith('"'):
                    name_raw = name_raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
                name = imap_utf7_decode(name_raw)
                folders.append(FolderInfo(name=name, attributes=attributes))
            except Exception:
                continue

        return folders

    def resolve_special_folder(
        self,
        special: SpecialFolder,
        host_alias: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve a logical folder role to a concrete server folder name.

        Resolution order:
          1. RFC 6154 ``SPECIAL-USE`` attribute advertised by the server.
          2. Case-insensitive match against well-known provider/IMAP names.

        Returns ``None`` if no suitable folder can be found.
        """
        folders = self.list_folders_with_attributes(host_alias)
        if not folders:
            return None

        # 1. Special-use attribute (most reliable, provider-independent).
        wanted_attr = _SPECIAL_USE_ATTR.get(special)
        if wanted_attr:
            for f in folders:
                if wanted_attr in f.attributes:
                    return f.name

        # 2. Name-based fallback against the actual folder list.
        by_lower = {f.name.lower(): f.name for f in folders}
        for candidate in _SPECIAL_NAME_FALLBACKS.get(special, []):
            actual = by_lower.get(candidate.lower())
            if actual:
                return actual

        return None

    @staticmethod
    def _supports_move(conn: "ImapConnection") -> bool:
        """Whether the server advertises the RFC 6851 ``MOVE`` capability."""
        try:
            caps = conn._connection.capabilities  # tuple of str, upper-cased
            return "MOVE" in caps
        except Exception:
            return False

    def _select_folder(self, conn: "ImapConnection", folder: str) -> None:
        """Select a folder with proper quoting/encoding (supports spaces/UTF-7)."""
        conn.connect()
        typ, data = conn._connection.select(_quote_mailbox(folder))
        if typ != "OK":
            raise ImapReadError(
                f"Failed to select folder: {data}", conn.host_alias, folder
            )
        conn._current_folder = folder

    def set_flags(
        self,
        uid: str,
        folder: str,
        flags: List[str],
        add: bool = True,
        host_alias: Optional[str] = None,
        expunge: bool = False,
    ) -> bool:
        """Add or remove IMAP flags on a message (UID STORE)."""
        conn = self._get_connection(host_alias)
        self._select_folder(conn, folder)
        op = "+FLAGS" if add else "-FLAGS"
        conn._connection.uid("STORE", uid.encode(), op, "(" + " ".join(flags) + ")")
        if expunge:
            conn._connection.expunge()
        return True

    def move_email(
        self,
        uid: str,
        source_folder: str,
        dest_folder: str,
        host_alias: Optional[str] = None,
    ) -> bool:
        """Move a message between folders.

        Uses RFC 6851 ``UID MOVE`` when supported, otherwise falls back to
        ``COPY`` + ``\\Deleted`` + ``EXPUNGE``.
        """
        conn = self._get_connection(host_alias)
        self._select_folder(conn, source_folder)
        quoted_dest = _quote_mailbox(dest_folder)

        if self._supports_move(conn):
            typ, data = conn._connection.uid("MOVE", uid.encode(), quoted_dest)
            if typ != "OK":
                raise ImapReadError(
                    f"MOVE failed: {data}", conn.host_alias, dest_folder
                )
            return True

        # Fallback: COPY then flag-delete + expunge in the source folder.
        typ, data = conn._connection.uid("COPY", uid.encode(), quoted_dest)
        if typ != "OK":
            raise ImapReadError(f"COPY failed: {data}", conn.host_alias, dest_folder)
        conn._connection.uid("STORE", uid.encode(), "+FLAGS", "(\\Deleted)")
        conn._connection.expunge()
        return True

    def delete_email(
        self,
        uid: str,
        folder: str,
        permanent: bool = False,
        host_alias: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> bool:
        """Delete a message.

        ``permanent=True`` flags ``\\Deleted`` and expunges (unrecoverable).
        Otherwise the message is moved to the server's Trash folder.
        """
        if permanent:
            return self.set_flags(
                uid,
                folder,
                ["\\Deleted"],
                add=True,
                host_alias=host_alias,
                expunge=True,
            )

        trash = self.resolve_special_folder(
            SpecialFolder.TRASH, host_alias, provider=provider
        )
        if not trash:
            raise ImapReadError(
                "Could not locate a Trash folder on this account",
                host_alias or "",
                folder,
            )
        if trash.lower() == folder.lower():
            return True  # already in Trash
        return self.move_email(uid, folder, trash, host_alias=host_alias)

    def archive_email(
        self,
        uid: str,
        folder: str,
        host_alias: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> bool:
        """Archive a message.

        Gmail has no ``\\Archive`` folder; archiving means moving out of the
        Inbox into *All Mail* (``\\All``). Other providers use ``\\Archive``.
        """
        is_gmail = (provider or "").lower() in {"google", "gmail"}
        target = None
        if is_gmail:
            target = self.resolve_special_folder(
                SpecialFolder.ALL, host_alias, provider=provider
            )
        if not target:
            target = self.resolve_special_folder(
                SpecialFolder.ARCHIVE, host_alias, provider=provider
            )
        if not target:
            raise ImapReadError(
                "Could not locate an Archive/All Mail folder on this account",
                host_alias or "",
                folder,
            )
        if target.lower() == folder.lower():
            return True
        return self.move_email(uid, folder, target, host_alias=host_alias)

    def mark_as_spam(
        self,
        uid: str,
        folder: str,
        host_alias: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> bool:
        """Move a message to the server's Junk/Spam folder."""
        junk = self.resolve_special_folder(
            SpecialFolder.JUNK, host_alias, provider=provider
        )
        if not junk:
            raise ImapReadError(
                "Could not locate a Junk/Spam folder on this account",
                host_alias or "",
                folder,
            )
        if junk.lower() == folder.lower():
            return True
        return self.move_email(uid, folder, junk, host_alias=host_alias)

    def get_folder_stats(
        self, host_alias: Optional[str] = None, folder: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get folder statistics.

        Args:
            host_alias: Host alias
            folder: Folder name

        Returns:
            Dict with total, unseen counts
        """
        conn = self._get_connection(host_alias)
        folder_name = folder or conn.imap_config.default_folder
        total = conn.select_folder(folder_name)

        # Count unseen
        typ, data = conn._connection.search(None, "UNSEEN")
        unseen = len(data[0].split()) if typ == "OK" and data[0] else 0

        return {
            "folder": folder_name,
            "total": total,
            "unseen": unseen,
        }

    def close_all(self) -> None:
        """Close all IMAP connections."""
        for conn in self._connections.values():
            conn.disconnect()
        self._connections.clear()

    def list_configured_hosts(self) -> List[str]:
        """Get list of hosts with IMAP configured."""
        return self.config.email.list_imap_enabled_hosts()


def get_imap_service() -> Optional[ImapService]:
    """Get IMAP service instance if configured.

    Returns:
        ImapService instance or None if no IMAP hosts configured
    """
    try:
        config = get_email_config()
        imap_hosts = config.email.list_imap_enabled_hosts()

        if not imap_hosts:
            logger.debug("No IMAP hosts configured - IMAP service unavailable")
            return None

        return ImapService(config)

    except FileNotFoundError:
        logger.warning("Email config not found - IMAP service unavailable")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize IMAP service: {e}")
        return None


def is_imap_configured() -> bool:
    """Check if any IMAP host is configured."""
    service = get_imap_service()
    return service is not None and len(service.list_configured_hosts()) > 0

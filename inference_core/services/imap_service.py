"""
IMAP Service

Production-grade IMAP email reading service with support for:
- Secure SSL/TLS connections
- Multiple mailbox folder navigation
- Message parsing (multipart, encodings)
- Comprehensive logging and error handling

This service complements EmailService (SMTP) for bidirectional email operations.
"""

import email
import imaplib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from email.header import decode_header, make_header
from email.message import Message
from email.utils import parseaddr, parsedate_to_datetime
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
        if not password:
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

    def _fetch_message(
        self, conn: ImapConnection, uid: bytes, folder: str
    ) -> Optional[EmailMessage]:
        """Fetch and parse a single message by UID."""
        typ, data = conn._connection.uid("FETCH", uid, "(RFC822)")
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
            except (ValueError, TypeError):
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
                content_type = part.get_content_type()
                disposition = part.get_content_disposition()

                if disposition == "attachment":
                    continue

                if content_type == "text/plain" and not body_text:
                    body_text = self._decode_payload(part)
                elif content_type == "text/html" and not body_html:
                    body_html = self._decode_payload(part)
        else:
            content_type = msg.get_content_type()
            payload = self._decode_payload(msg)
            if content_type == "text/html":
                body_html = payload
            else:
                body_text = payload

        return body_text, body_html

    def _decode_payload(self, part: Message) -> str:
        """Decode message payload with charset handling."""
        payload = part.get_payload(decode=True)
        if not payload:
            return ""

        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            return payload.decode("utf-8", errors="replace")

    def _decode_header(self, value: str) -> str:
        """Decode MIME-encoded header value."""
        if not value:
            return ""
        try:
            return str(make_header(decode_header(value)))
        except Exception:
            return value

    def list_folders(self, host_alias: Optional[str] = None) -> List[str]:
        """List available mailbox folders.

        Args:
            host_alias: Host alias (uses default if None)

        Returns:
            List of folder names
        """
        conn = self._get_connection(host_alias)
        conn.connect()

        typ, data = conn._connection.list()
        if typ != "OK":
            return []

        folders = []
        for item in data:
            if isinstance(item, bytes):
                # Parse folder name from IMAP LIST response
                # Format: '(\\HasNoChildren) "/" "FolderName"'
                try:
                    decoded = item.decode("utf-8")
                    parts = decoded.rsplit('"', 2)
                    if len(parts) >= 2:
                        folders.append(parts[-2])
                except Exception:
                    continue

        return folders

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

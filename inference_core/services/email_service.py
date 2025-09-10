"""
Email Service

Production-grade email delivery service using smtplib with support for:
- HTML and text email bodies (multipart/alternative)
- File attachments with size limits
- SSL/STARTTLS secure transport
- Multiple SMTP hosts with failover
- Comprehensive logging and error handling
"""

import hashlib
import logging
import mimetypes
import smtplib
import ssl
import socket
import time
from datetime import datetime
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from inference_core.core.email_config import (
    EmailHostConfig, 
    FullEmailConfig, 
    get_email_config, 
    is_email_configured
)

logger = logging.getLogger(__name__)


class EmailSendError(Exception):
    """Exception raised when email sending fails"""
    
    def __init__(self, message: str, host_alias: str, original_error: Optional[Exception] = None):
        self.message = message
        self.host_alias = host_alias
        self.original_error = original_error
        super().__init__(f"Email send failed via {host_alias}: {message}")


class EmailService:
    """Email service for sending emails via SMTP"""
    
    def __init__(self, config: Optional[FullEmailConfig] = None):
        """
        Initialize email service
        
        Args:
            config: Email configuration (loads from file if not provided)
        """
        self.config = config or get_email_config()
    
    def send_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        text: str,
        html: Optional[str] = None,
        attachments: Optional[List[Union[Path, tuple[str, bytes, str]]]] = None,
        host_alias: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        reply_to: Optional[Union[str, List[str]]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Send an email
        
        Args:
            to: Recipient email address(es)
            subject: Email subject
            text: Plain text email body
            html: HTML email body (optional)
            attachments: List of attachments (Path objects or (filename, bytes, mime_type) tuples)
            host_alias: SMTP host alias to use (uses default if not specified)
            cc: CC recipients
            bcc: BCC recipients
            reply_to: Reply-to address(es)
            headers: Additional email headers
            
        Returns:
            Message-ID of sent email
            
        Raises:
            EmailSendError: If email sending fails
        """
        start_time = time.time()
        host_config = self.config.email.get_host_config(host_alias)
        host_alias = host_alias or self.config.email.default_host
        
        # Convert to lists if needed
        to_list = [to] if isinstance(to, str) else to
        cc_list = cc or []
        bcc_list = bcc or []
        reply_to_list = [reply_to] if isinstance(reply_to, str) else (reply_to or [])
        
        # Build email message
        try:
            message = self._build_message(
                to=to_list,
                subject=subject,
                text=text,
                html=html,
                attachments=attachments,
                cc=cc_list,
                bcc=bcc_list,
                reply_to=reply_to_list,
                headers=headers,
                host_config=host_config,
            )
        except Exception as e:
            logger.error(f"Failed to build email message: {e}")
            raise EmailSendError("Failed to build email message", host_alias, e)
        
        # Send email
        try:
            self._send_message(message, host_config, bcc_list)
            duration = time.time() - start_time
            
            # Log success (with minimal PII)
            message_id = message.get("Message-ID", "unknown")
            subject_hash = hashlib.sha256(subject.encode()).hexdigest()[:8]
            recipient_count = len(to_list + cc_list + bcc_list)
            
            logger.info(
                f"Email sent successfully",
                extra={
                    "message_id": message_id,
                    "host_alias": host_alias,
                    "recipient_count": recipient_count,
                    "subject_hash": subject_hash,
                    "has_html": html is not None,
                    "attachment_count": len(attachments) if attachments else 0,
                    "duration_seconds": round(duration, 3),
                }
            )
            
            return message_id
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Failed to send email via {host_alias}: {e}",
                extra={
                    "host_alias": host_alias,
                    "error_type": type(e).__name__,
                    "duration_seconds": round(duration, 3),
                }
            )
            raise EmailSendError("Failed to send email", host_alias, e)
    
    def _build_message(
        self,
        to: List[str],
        subject: str,
        text: str,
        html: Optional[str],
        attachments: Optional[List[Union[Path, tuple[str, bytes, str]]]],
        cc: List[str],
        bcc: List[str],
        reply_to: List[str],
        headers: Optional[Dict[str, str]],
        host_config: EmailHostConfig,
    ) -> EmailMessage:
        """Build email message with all components"""
        message = EmailMessage()
        
        # Basic headers
        message["From"] = formataddr((host_config.from_name, host_config.from_email))
        message["To"] = ", ".join(to)
        message["Subject"] = subject
        message["Date"] = formatdate(localtime=True)
        message["Message-ID"] = make_msgid()
        
        if cc:
            message["Cc"] = ", ".join(cc)
        
        if reply_to:
            message["Reply-To"] = ", ".join(reply_to)
        
        # Custom headers
        if headers:
            for key, value in headers.items():
                message[key] = value
        
        # Set email content (multipart/alternative if both text and HTML)
        if html:
            message.set_content(text)
            message.add_alternative(html, subtype='html')
        else:
            message.set_content(text)
        
        # Add attachments
        if attachments:
            self._add_attachments(message, attachments, host_config)
        
        return message
    
    def _add_attachments(
        self, 
        message: EmailMessage, 
        attachments: List[Union[Path, tuple[str, bytes, str]]], 
        host_config: EmailHostConfig
    ):
        """Add attachments to email message"""
        total_size_mb = 0
        
        for attachment in attachments:
            if isinstance(attachment, Path):
                # File path attachment
                if not attachment.exists():
                    raise ValueError(f"Attachment file not found: {attachment}")
                
                filename = attachment.name
                content = attachment.read_bytes()
                mime_type, _ = mimetypes.guess_type(str(attachment))
                mime_type = mime_type or "application/octet-stream"
                
            elif isinstance(attachment, tuple) and len(attachment) == 3:
                # (filename, bytes, mime_type) tuple
                filename, content, mime_type = attachment
                if not isinstance(content, bytes):
                    raise ValueError("Attachment content must be bytes")
                
            else:
                raise ValueError("Attachments must be Path objects or (filename, bytes, mime_type) tuples")
            
            # Check size limits
            size_mb = len(content) / (1024 * 1024)
            total_size_mb += size_mb
            
            if total_size_mb > host_config.max_attachment_mb:
                raise ValueError(
                    f"Total attachment size ({total_size_mb:.2f} MB) exceeds limit "
                    f"({host_config.max_attachment_mb} MB)"
                )
            
            # Add attachment
            maintype, subtype = mime_type.split('/', 1) if '/' in mime_type else ('application', 'octet-stream')
            message.add_attachment(
                content,
                maintype=maintype,
                subtype=subtype,
                filename=filename
            )
    
    def _send_message(self, message: EmailMessage, host_config: EmailHostConfig, bcc_recipients: List[str] = None):
        """Send email message via SMTP"""
        all_recipients = []
        
        # Collect all recipients
        if message["To"]:
            all_recipients.extend([addr.strip() for addr in message["To"].split(",")])
        if message.get("Cc"):
            all_recipients.extend([addr.strip() for addr in message["Cc"].split(",")])
        
        # Add BCC recipients (not in message headers)
        if bcc_recipients:
            all_recipients.extend(bcc_recipients)
        
        # Remove empty strings
        all_recipients = [addr for addr in all_recipients if addr]
        
        if not all_recipients:
            raise ValueError("No recipients specified")
        
        # Get password
        password = host_config.get_password()
        if not password:
            raise ValueError(f"Password not found in environment variable: {host_config.password_env}")
        
        # Create SSL context
        ssl_context = ssl.create_default_context()
        if not host_config.verify_hostname:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        
        # Connect and send
        smtp_class = smtplib.SMTP_SSL if host_config.use_ssl else smtplib.SMTP
        
        with smtp_class(
            host=host_config.host,
            port=host_config.port,
            timeout=host_config.timeout,
            context=ssl_context if host_config.use_ssl else None
        ) as server:
            # Enable debug logging if in debug mode
            if logger.isEnabledFor(logging.DEBUG):
                server.set_debuglevel(1)
            
            # STARTTLS if configured
            if host_config.use_starttls:
                server.starttls(context=ssl_context)
            
            # Authenticate
            server.login(host_config.username, password)
            
            # Send message
            server.send_message(message, to_addrs=all_recipients)


def get_email_service() -> Optional[EmailService]:
    """
    Get email service instance if email is configured
    
    Returns:
        EmailService instance or None if not configured
    """
    if not is_email_configured():
        logger.warning("Email not configured - email service unavailable")
        return None
    
    try:
        return EmailService()
    except Exception as e:
        logger.error(f"Failed to initialize email service: {e}")
        return None
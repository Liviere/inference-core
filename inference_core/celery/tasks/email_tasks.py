"""
Email Celery Tasks

Celery tasks for asynchronous email sending with retry/backoff handling.
"""

import base64
import logging
import smtplib
import socket
import ssl
import time
from typing import Any, Dict, List, Optional, Union

from celery import current_task

from inference_core.celery.celery_main import celery_app
from inference_core.services.email_service import EmailSendError, get_email_service

logger = logging.getLogger(__name__)


class EmailTaskError(Exception):
    """Exception for email task errors"""

    pass


@celery_app.task(
    bind=True,
    name="email.send",
    queue="mail",
    autoretry_for=(
        smtplib.SMTPException,
        ssl.SSLError,
        socket.timeout,
        ConnectionError,
        EmailSendError,
    ),
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
    max_retries=5,
    acks_late=True,
    time_limit=600,
)
def send_email_task(
    self,
    to: Union[str, List[str]],
    subject: str,
    text: str,
    html: Optional[str] = None,
    attachments: Optional[List[Dict[str, str]]] = None,
    host_alias: Optional[str] = None,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
    reply_to: Optional[Union[str, List[str]]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Send email via Celery task

    Args:
        to: Recipient email address(es)
        subject: Email subject
        text: Plain text email body
        html: HTML email body (optional)
        attachments: List of attachment dicts with 'filename', 'content_b64', 'mime' keys
        host_alias: SMTP host alias to use
        cc: CC recipients
        bcc: BCC recipients
        reply_to: Reply-to address(es)
        headers: Additional email headers

    Returns:
        Dict with task result information

    Raises:
        EmailTaskError: If task fails permanently
        Retry: If task should be retried
    """
    start_time = time.time()
    task_id = current_task.request.id if current_task else "unknown"

    try:
        # Get email service
        email_service = get_email_service()
        if not email_service:
            raise EmailTaskError("Email service not available - check configuration")

        # Process attachments
        processed_attachments = None
        if attachments:
            processed_attachments = []
            for att in attachments:
                if not all(key in att for key in ["filename", "content_b64", "mime"]):
                    raise EmailTaskError(
                        "Invalid attachment format - missing required keys"
                    )

                try:
                    content = base64.b64decode(att["content_b64"])
                    processed_attachments.append(
                        (att["filename"], content, att["mime"])
                    )
                except Exception as e:
                    raise EmailTaskError(
                        f"Failed to decode attachment {att['filename']}: {e}"
                    )

        # Log task start
        logger.info(
            f"Starting email send task",
            extra={
                "task_id": task_id,
                "host_alias": host_alias,
                "recipient_count": len(to) if isinstance(to, list) else 1,
                "has_html": html is not None,
                "attachment_count": len(attachments) if attachments else 0,
            },
        )

        # Send email
        message_id = email_service.send_email(
            to=to,
            subject=subject,
            text=text,
            html=html,
            attachments=processed_attachments,
            host_alias=host_alias,
            cc=cc,
            bcc=bcc,
            reply_to=reply_to,
            headers=headers,
        )

        duration = time.time() - start_time

        # Log success
        logger.info(
            f"Email task completed successfully",
            extra={
                "task_id": task_id,
                "message_id": message_id,
                "duration_seconds": round(duration, 3),
            },
        )

        return {
            "status": "success",
            "message_id": message_id,
            "task_id": task_id,
            "duration": round(duration, 3),
        }

    except EmailSendError as e:
        duration = time.time() - start_time

        # Check if this is a retryable error
        if _is_retryable_error(e.original_error):
            logger.warning(
                f"Email task failed with retryable error, retrying",
                extra={
                    "task_id": task_id,
                    "error": str(e),
                    "retry_count": self.request.retries,
                    "max_retries": self.max_retries,
                    "duration_seconds": round(duration, 3),
                },
            )
            raise self.retry(exc=e)
        else:
            # Non-retryable error, fail permanently
            logger.error(
                f"Email task failed with non-retryable error",
                extra={
                    "task_id": task_id,
                    "error": str(e),
                    "duration_seconds": round(duration, 3),
                },
            )
            raise EmailTaskError(f"Email send failed: {e}")

    except Exception as e:
        duration = time.time() - start_time

        # Check if this is a retryable error
        if _is_retryable_error(e):
            logger.warning(
                f"Email task failed with retryable error, retrying",
                extra={
                    "task_id": task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "retry_count": self.request.retries,
                    "max_retries": self.max_retries,
                    "duration_seconds": round(duration, 3),
                },
            )
            raise self.retry(exc=e)
        else:
            # Non-retryable error, fail permanently
            logger.error(
                f"Email task failed with non-retryable error",
                extra={
                    "task_id": task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_seconds": round(duration, 3),
                },
            )
            raise EmailTaskError(f"Email task failed: {e}")


def _is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable

    Args:
        error: Exception to check

    Returns:
        True if error should trigger a retry
    """
    if isinstance(
        error,
        (
            smtplib.SMTPConnectError,
            smtplib.SMTPServerDisconnected,
            smtplib.SMTPResponseException,
            socket.timeout,
            socket.gaierror,
            ConnectionError,
            ssl.SSLError,
        ),
    ):
        return True

    # Check for specific SMTP response codes that are retryable
    if isinstance(error, smtplib.SMTPException):
        # Temporary failures (4xx codes) are retryable
        error_str = str(error)
        if any(code in error_str for code in ["421", "450", "451", "452"]):
            return True

    return False


def send_email_async(
    to: Union[str, List[str]],
    subject: str,
    text: str,
    html: Optional[str] = None,
    attachments: Optional[List[Dict[str, str]]] = None,
    host_alias: Optional[str] = None,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
    reply_to: Optional[Union[str, List[str]]] = None,
    headers: Optional[Dict[str, str]] = None,
    countdown: Optional[int] = None,
    eta: Optional[Any] = None,
) -> Any:
    """
    Send email asynchronously via Celery

    Args:
        to: Recipient email address(es)
        subject: Email subject
        text: Plain text email body
        html: HTML email body (optional)
        attachments: List of attachment dicts with 'filename', 'content_b64', 'mime' keys
        host_alias: SMTP host alias to use
        cc: CC recipients
        bcc: BCC recipients
        reply_to: Reply-to address(es)
        headers: Additional email headers
        countdown: Delay in seconds before execution
        eta: Specific datetime for execution

    Returns:
        Celery AsyncResult
    """
    return send_email_task.apply_async(
        args=[to, subject, text],
        kwargs={
            "html": html,
            "attachments": attachments,
            "host_alias": host_alias,
            "cc": cc,
            "bcc": bcc,
            "reply_to": reply_to,
            "headers": headers,
        },
        countdown=countdown,
        eta=eta,
    )


def encode_attachment(filename: str, content: bytes, mime_type: str) -> Dict[str, str]:
    """
    Encode attachment for Celery serialization

    Args:
        filename: Attachment filename
        content: File content as bytes
        mime_type: MIME type

    Returns:
        Dict suitable for Celery serialization
    """
    return {
        "filename": filename,
        "content_b64": base64.b64encode(content).decode("ascii"),
        "mime": mime_type,
    }

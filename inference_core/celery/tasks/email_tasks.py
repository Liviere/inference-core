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


# =============================================================================
# IMAP Polling Tasks
# =============================================================================


class ImapPollError(Exception):
    """Exception for IMAP polling task errors."""

    pass


@celery_app.task(
    bind=True,
    name="email.poll_imap",
    queue="mail",
    autoretry_for=(
        ConnectionError,
        socket.timeout,
        ssl.SSLError,
    ),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=3,
    acks_late=True,
    time_limit=120,
)
def poll_imap_task(
    self,
    host_alias: Optional[str] = None,
    folder: str = "INBOX",
    limit: int = 50,
    callback_task: Optional[str] = None,
    callback_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Poll IMAP mailbox for new emails.

    Fetches unseen emails and optionally dispatches them to a callback task
    for processing (e.g., agent-based email handling).

    Args:
        host_alias: IMAP host alias (uses default if None)
        folder: Mailbox folder to poll
        limit: Maximum emails to fetch per poll
        callback_task: Optional Celery task name to invoke for each email.
                       Task receives: (email_data: dict, host_alias: str)
        callback_kwargs: Additional kwargs to pass to callback task

    Returns:
        Dict with poll results (count, message_ids)
    """
    start_time = time.time()
    task_id = current_task.request.id if current_task else "unknown"

    try:
        from inference_core.services.imap_service import get_imap_service

        imap_service = get_imap_service()
        if not imap_service:
            raise ImapPollError("IMAP service not available - check configuration")

        # Determine which host to poll
        alias = host_alias or imap_service.config.email.default_host

        logger.info(
            "Starting IMAP poll task",
            extra={
                "task_id": task_id,
                "host_alias": alias,
                "folder": folder,
                "limit": limit,
            },
        )

        # Fetch unseen emails
        messages = imap_service.fetch_unseen_emails(
            host_alias=alias,
            folder=folder,
            limit=limit,
            mark_as_read=False,  # Let callback decide
        )

        message_ids = []
        callback_results = []

        for msg in messages:
            message_ids.append(msg.uid)

            # Dispatch to callback if configured
            if callback_task:
                email_data = {
                    "uid": msg.uid,
                    "message_id": msg.message_id,
                    "subject": msg.subject,
                    "from_address": msg.from_address,
                    "from_name": msg.from_name,
                    "to_addresses": msg.to_addresses,
                    "date": msg.date.isoformat() if msg.date else None,
                    "body_text": msg.body_text,
                    "body_html": msg.body_html,
                    "has_attachments": msg.has_attachments,
                    "attachment_names": msg.attachment_names,
                    "folder": msg.folder,
                }

                try:
                    result = celery_app.send_task(
                        callback_task,
                        args=[email_data, alias],
                        kwargs=callback_kwargs or {},
                    )
                    callback_results.append({"uid": msg.uid, "task_id": str(result.id)})
                except Exception as e:
                    logger.error(
                        "Failed to dispatch callback for email %s: %s",
                        msg.uid,
                        e,
                    )

        duration = time.time() - start_time

        logger.info(
            "IMAP poll completed",
            extra={
                "task_id": task_id,
                "host_alias": alias,
                "emails_found": len(messages),
                "callbacks_dispatched": len(callback_results),
                "duration_seconds": round(duration, 3),
            },
        )

        return {
            "status": "success",
            "host_alias": alias,
            "folder": folder,
            "emails_found": len(messages),
            "message_ids": message_ids,
            "callback_results": callback_results,
            "duration": round(duration, 3),
        }

    except ImapPollError:
        raise
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "IMAP poll failed",
            extra={
                "task_id": task_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_seconds": round(duration, 3),
            },
        )

        if _is_imap_retryable_error(e):
            raise self.retry(exc=e)

        raise ImapPollError(f"IMAP poll failed: {e}")


@celery_app.task(
    bind=True,
    name="email.poll_all_imap",
    queue="mail",
    time_limit=300,
)
def poll_all_imap_accounts_task(
    self,
    folder: str = "INBOX",
    limit: int = 50,
    callback_task: Optional[str] = None,
    callback_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Poll all configured IMAP accounts for new emails.

    Dispatches individual poll_imap_task for each configured IMAP host.

    Args:
        folder: Mailbox folder to poll (same for all accounts)
        limit: Maximum emails to fetch per account
        callback_task: Celery task name for processing emails
        callback_kwargs: Additional kwargs for callback

    Returns:
        Dict with dispatched task info per account
    """
    start_time = time.time()
    task_id = current_task.request.id if current_task else "unknown"

    try:
        from inference_core.services.imap_service import get_imap_service

        imap_service = get_imap_service()
        if not imap_service:
            return {
                "status": "skipped",
                "reason": "No IMAP service configured",
            }

        imap_hosts = imap_service.list_configured_hosts()
        if not imap_hosts:
            return {
                "status": "skipped",
                "reason": "No IMAP hosts configured",
            }

        logger.info(
            "Starting poll for all IMAP accounts",
            extra={
                "task_id": task_id,
                "account_count": len(imap_hosts),
                "accounts": imap_hosts,
            },
        )

        dispatched = []
        for host_alias in imap_hosts:
            try:
                result = poll_imap_task.delay(
                    host_alias=host_alias,
                    folder=folder,
                    limit=limit,
                    callback_task=callback_task,
                    callback_kwargs=callback_kwargs,
                )
                dispatched.append(
                    {
                        "host_alias": host_alias,
                        "task_id": str(result.id),
                    }
                )
            except Exception as e:
                logger.error(
                    "Failed to dispatch poll task for %s: %s",
                    host_alias,
                    e,
                )
                dispatched.append(
                    {
                        "host_alias": host_alias,
                        "error": str(e),
                    }
                )

        duration = time.time() - start_time

        return {
            "status": "success",
            "accounts_polled": len(dispatched),
            "dispatched_tasks": dispatched,
            "duration": round(duration, 3),
        }

    except Exception as e:
        logger.error("Failed to poll all IMAP accounts: %s", e)
        raise


def _is_imap_retryable_error(error: Exception) -> bool:
    """Determine if IMAP error is retryable."""
    import imaplib

    if isinstance(
        error,
        (
            socket.timeout,
            socket.gaierror,
            ConnectionError,
            ssl.SSLError,
            imaplib.IMAP4.abort,
        ),
    ):
        return True

    # Check for temporary IMAP errors
    if isinstance(error, imaplib.IMAP4.error):
        error_str = str(error).lower()
        if any(
            term in error_str
            for term in ["temporary", "try again", "connection", "timeout"]
        ):
            return True

    return False


def schedule_imap_polling(
    host_alias: Optional[str] = None,
    folder: str = "INBOX",
    limit: int = 50,
    callback_task: Optional[str] = None,
    callback_kwargs: Optional[Dict[str, Any]] = None,
    countdown: Optional[int] = None,
) -> Any:
    """Schedule IMAP polling task.

    Convenience function for scheduling poll task with options.

    Args:
        host_alias: Specific host to poll (None = default)
        folder: Mailbox folder
        limit: Max emails per poll
        callback_task: Task to process each email
        callback_kwargs: Additional callback kwargs
        countdown: Delay in seconds

    Returns:
        Celery AsyncResult
    """
    return poll_imap_task.apply_async(
        kwargs={
            "host_alias": host_alias,
            "folder": folder,
            "limit": limit,
            "callback_task": callback_task,
            "callback_kwargs": callback_kwargs,
        },
        countdown=countdown,
    )

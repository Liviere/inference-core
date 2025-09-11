"""
Tests for email Celery tasks
"""

import base64
import smtplib
import ssl
from unittest.mock import Mock, patch, MagicMock

import pytest
from celery.exceptions import Retry

from inference_core.celery.tasks.email_tasks import (
    send_email_task,
    send_email_async,
    encode_attachment,
    _is_retryable_error,
    EmailTaskError,
)


class TestSendEmailTask:
    """Test send_email_task Celery task"""

    @patch('inference_core.celery.tasks.email_tasks.get_email_service')
    def test_send_simple_email_task(self, mock_get_service):
        """Test successful email sending via task"""
        # Mock email service
        mock_service = Mock()
        mock_service.send_email.return_value = "test-message-id"
        mock_get_service.return_value = mock_service
        
        # Execute task
        result = send_email_task(
            to="test@example.com",
            subject="Test Subject",
            text="Test message"
        )
        
        # Verify result
        assert result["status"] == "success"
        assert result["message_id"] == "test-message-id"
        assert "task_id" in result
        assert "duration" in result
        
        # Verify email service was called correctly
        mock_service.send_email.assert_called_once_with(
            to="test@example.com",
            subject="Test Subject",
            text="Test message",
            html=None,
            attachments=None,
            host_alias=None,
            cc=None,
            bcc=None,
            reply_to=None,
            headers=None
        )

    @patch('inference_core.celery.tasks.email_tasks.get_email_service')
    def test_send_email_task_with_all_options(self, mock_get_service):
        """Test email task with all options"""
        mock_service = Mock()
        mock_service.send_email.return_value = "test-message-id"
        mock_get_service.return_value = mock_service
        
        # Prepare attachment
        attachment_content = b"test file content"
        attachment_b64 = base64.b64encode(attachment_content).decode('ascii')
        
        attachments = [{
            "filename": "test.txt",
            "content_b64": attachment_b64,
            "mime": "text/plain"
        }]
        
        result = send_email_task(
            to=["test1@example.com", "test2@example.com"],
            subject="Test Subject",
            text="Test message",
            html="<p>Test HTML</p>",
            attachments=attachments,
            host_alias="backup",
            cc=["cc@example.com"],
            bcc=["bcc@example.com"],
            reply_to="reply@example.com",
            headers={"X-Custom": "value"}
        )
        
        assert result["status"] == "success"
        
        # Verify attachments were processed correctly
        call_args = mock_service.send_email.call_args
        processed_attachments = call_args.kwargs["attachments"]
        assert len(processed_attachments) == 1
        assert processed_attachments[0] == ("test.txt", attachment_content, "text/plain")

    @patch('inference_core.celery.tasks.email_tasks.get_email_service')
    def test_email_service_not_available(self, mock_get_service):
        """Test task failure when email service not available"""
        mock_get_service.return_value = None
        
        with pytest.raises(EmailTaskError, match="Email service not available"):
            send_email_task(
                to="test@example.com",
                subject="Test",
                text="Test message"
            )

    @patch('inference_core.celery.tasks.email_tasks.get_email_service')
    def test_invalid_attachment_format(self, mock_get_service):
        """Test task failure with invalid attachment format"""
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        
        # Missing required keys in attachment
        invalid_attachments = [{"filename": "test.txt"}]  # Missing content_b64 and mime
        
        with pytest.raises(EmailTaskError, match="Invalid attachment format"):
            send_email_task(
                to="test@example.com",
                subject="Test",
                text="Test message",
                attachments=invalid_attachments
            )

    @patch('inference_core.celery.tasks.email_tasks.get_email_service')
    def test_invalid_base64_attachment(self, mock_get_service):
        """Test task failure with invalid base64 attachment content"""
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        
        # Invalid base64 content 
        invalid_attachments = [{
            "filename": "test.txt",
            "content_b64": "###invalid###",
            "mime": "text/plain"
        }]
        
        with pytest.raises(EmailTaskError, match="Failed to decode attachment"):
            send_email_task(
                to="test@example.com",
                subject="Test",
                text="Test message",
                attachments=invalid_attachments
            )

    @patch('inference_core.celery.tasks.email_tasks.get_email_service')
    def test_retryable_smtp_error(self, mock_get_service):
        """Test task retry on retryable SMTP errors"""
        mock_service = Mock()
        mock_service.send_email.side_effect = smtplib.SMTPConnectError(421, "Connection failed")
        mock_get_service.return_value = mock_service
        
        # Mock the task's retry method
        with patch.object(send_email_task, 'retry') as mock_retry:
            mock_retry.side_effect = Retry("Retrying")
            
            with pytest.raises(Retry):
                send_email_task(
                    to="test@example.com",
                    subject="Test",
                    text="Test message"
                )
            
            mock_retry.assert_called_once()

    @patch('inference_core.celery.tasks.email_tasks.get_email_service')
    def test_non_retryable_error(self, mock_get_service):
        """Test task failure on non-retryable errors"""
        mock_service = Mock()
        mock_service.send_email.side_effect = ValueError("Invalid recipient")
        mock_get_service.return_value = mock_service
        
        with pytest.raises(EmailTaskError, match="Email task failed"):
            send_email_task(
                to="test@example.com",
                subject="Test",
                text="Test message"
            )


class TestRetryableErrorDetection:
    """Test _is_retryable_error function"""

    def test_retryable_smtp_errors(self):
        """Test that SMTP connection errors are retryable"""
        assert _is_retryable_error(smtplib.SMTPConnectError(421, "Connection failed"))
        assert _is_retryable_error(smtplib.SMTPServerDisconnected())
        assert _is_retryable_error(smtplib.SMTPResponseException(421, "Service not available"))

    def test_retryable_network_errors(self):
        """Test that network errors are retryable"""
        import socket
        assert _is_retryable_error(socket.timeout())
        assert _is_retryable_error(socket.gaierror("Name resolution failed"))
        assert _is_retryable_error(ConnectionError("Connection failed"))
        assert _is_retryable_error(ssl.SSLError("SSL handshake failed"))

    def test_retryable_smtp_response_codes(self):
        """Test that temporary SMTP response codes are retryable"""
        # Temporary failures (4xx codes)
        assert _is_retryable_error(smtplib.SMTPException("421 Service not available"))
        assert _is_retryable_error(smtplib.SMTPException("450 Mailbox busy"))
        assert _is_retryable_error(smtplib.SMTPException("451 Local error"))
        assert _is_retryable_error(smtplib.SMTPException("452 Insufficient storage"))

    def test_non_retryable_smtp_errors(self):
        """Test that permanent SMTP errors are not retryable"""
        # Permanent failures (5xx codes)
        assert not _is_retryable_error(smtplib.SMTPException("550 Mailbox not found"))
        assert not _is_retryable_error(smtplib.SMTPException("553 Mailbox name invalid"))

    def test_non_retryable_other_errors(self):
        """Test that other errors are not retryable"""
        assert not _is_retryable_error(ValueError("Invalid input"))
        assert not _is_retryable_error(TypeError("Type error"))
        assert not _is_retryable_error(KeyError("Missing key"))


class TestAsyncEmailHelpers:
    """Test helper functions for async email sending"""

    @patch('inference_core.celery.tasks.email_tasks.send_email_task.apply_async')
    def test_send_email_async(self, mock_apply_async):
        """Test send_email_async helper function"""
        mock_result = Mock()
        mock_apply_async.return_value = mock_result
        
        result = send_email_async(
            to="test@example.com",
            subject="Test",
            text="Test message",
            html="<p>Test</p>",
            countdown=60
        )
        
        assert result == mock_result
        
        # Verify apply_async was called correctly
        mock_apply_async.assert_called_once_with(
            args=["test@example.com", "Test", "Test message"],
            kwargs={
                "html": "<p>Test</p>",
                "attachments": None,
                "host_alias": None,
                "cc": None,
                "bcc": None,
                "reply_to": None,
                "headers": None,
            },
            countdown=60,
            eta=None
        )

    def test_encode_attachment(self):
        """Test attachment encoding for Celery serialization"""
        filename = "test.txt"
        content = b"test file content"
        mime_type = "text/plain"
        
        encoded = encode_attachment(filename, content, mime_type)
        
        assert encoded["filename"] == filename
        assert encoded["mime"] == mime_type
        
        # Verify content can be decoded back
        decoded_content = base64.b64decode(encoded["content_b64"])
        assert decoded_content == content

    def test_encode_attachment_binary(self):
        """Test encoding binary attachment"""
        filename = "image.jpg"
        content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"  # PNG header
        mime_type = "image/png"
        
        encoded = encode_attachment(filename, content, mime_type)
        
        # Verify binary content is encoded correctly
        decoded_content = base64.b64decode(encoded["content_b64"])
        assert decoded_content == content


class TestTaskConfiguration:
    """Test Celery task configuration"""

    def test_task_configuration(self):
        """Test that task has correct Celery configuration"""
        task = send_email_task
        
        # Verify task configuration
        assert hasattr(task, 'autoretry_for')
        assert hasattr(task, 'retry_backoff')
        assert hasattr(task, 'max_retries')
        assert hasattr(task, 'acks_late')
        
        # Verify specific settings
        assert task.max_retries == 5
        assert task.acks_late is True
        assert task.retry_backoff is True
        assert task.retry_backoff_max == 120
        
        # Verify retryable exceptions
        assert smtplib.SMTPException in task.autoretry_for
        assert ssl.SSLError in task.autoretry_for
        assert ConnectionError in task.autoretry_for
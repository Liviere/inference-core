"""
Tests for EmailService functionality
"""

import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import tempfile

import pytest

from inference_core.core.email_config import EmailHostConfig, EmailConfig, FullEmailConfig, EmailSettings
from inference_core.services.email_service import EmailService, EmailSendError


class TestEmailService:
    """Test EmailService functionality"""

    def setup_method(self):
        """Set up test configuration"""
        self.host_config = EmailHostConfig(
            host="smtp.example.com",
            port=465,
            use_ssl=True,
            use_starttls=False,
            username="test@example.com",
            password_env="TEST_PASSWORD",
            from_email="no-reply@example.com",
            from_name="Test Service",
        )
        
        self.email_config = EmailConfig(
            default_host="primary",
            hosts={"primary": self.host_config}
        )
        
        self.settings = EmailSettings()
        
        self.full_config = FullEmailConfig(
            email=self.email_config,
            settings=self.settings
        )

    @patch.dict('os.environ', {'TEST_PASSWORD': 'secret123'})
    @patch('smtplib.SMTP_SSL')
    def test_send_simple_email(self, mock_smtp_ssl):
        """Test sending a simple text email"""
        # Mock SMTP connection
        mock_server = MagicMock()
        mock_smtp_ssl.return_value.__enter__.return_value = mock_server
        
        service = EmailService(self.full_config)
        
        message_id = service.send_email(
            to="recipient@example.com",
            subject="Test Subject",
            text="Test message body"
        )
        
        # Verify SMTP connection (don't check exact context object)
        mock_smtp_ssl.assert_called_once()
        call_args = mock_smtp_ssl.call_args
        assert call_args.kwargs["host"] == "smtp.example.com"
        assert call_args.kwargs["port"] == 465
        assert call_args.kwargs["timeout"] == 10
        assert call_args.kwargs["context"] is not None
        
        # Verify login
        mock_server.login.assert_called_once_with("test@example.com", "secret123")
        
        # Verify message was sent
        mock_server.send_message.assert_called_once()
        args, kwargs = mock_server.send_message.call_args
        sent_message = args[0]
        
        assert isinstance(sent_message, EmailMessage)
        assert sent_message["To"] == "recipient@example.com"
        assert sent_message["Subject"] == "Test Subject"
        assert sent_message["From"] == "Test Service <no-reply@example.com>"
        assert message_id is not None

    @patch.dict('os.environ', {'TEST_PASSWORD': 'secret123'})
    @patch('smtplib.SMTP_SSL')
    def test_send_html_email(self, mock_smtp_ssl):
        """Test sending email with both text and HTML"""
        mock_server = MagicMock()
        mock_smtp_ssl.return_value.__enter__.return_value = mock_server
        
        service = EmailService(self.full_config)
        
        service.send_email(
            to="recipient@example.com",
            subject="Test Subject",
            text="Plain text content",
            html="<p>HTML content</p>"
        )
        
        # Verify message has both text and HTML parts
        mock_server.send_message.assert_called_once()
        sent_message = mock_server.send_message.call_args[0][0]
        
        # Message should be multipart/alternative
        assert sent_message.is_multipart()

    @patch.dict('os.environ', {'TEST_PASSWORD': 'secret123'})
    @patch('smtplib.SMTP_SSL')
    def test_send_email_with_attachments(self, mock_smtp_ssl):
        """Test sending email with file attachments"""
        mock_server = MagicMock()
        mock_smtp_ssl.return_value.__enter__.return_value = mock_server
        
        service = EmailService(self.full_config)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
            f.write(b"Test file content")
            temp_path = Path(f.name)
        
        try:
            service.send_email(
                to="recipient@example.com",
                subject="Test with attachment",
                text="Message with attachment",
                attachments=[temp_path]
            )
            
            # Verify attachment was added
            mock_server.send_message.assert_called_once()
            sent_message = mock_server.send_message.call_args[0][0]
            assert sent_message.is_multipart()
            
        finally:
            temp_path.unlink()

    @patch.dict('os.environ', {'TEST_PASSWORD': 'secret123'})
    @patch('smtplib.SMTP_SSL')
    def test_send_email_with_memory_attachment(self, mock_smtp_ssl):
        """Test sending email with in-memory attachment"""
        mock_server = MagicMock()
        mock_smtp_ssl.return_value.__enter__.return_value = mock_server
        
        service = EmailService(self.full_config)
        
        attachment_content = b"Binary file content"
        
        service.send_email(
            to="recipient@example.com",
            subject="Test with memory attachment",
            text="Message with attachment",
            attachments=[("test.bin", attachment_content, "application/octet-stream")]
        )
        
        # Verify attachment was added
        mock_server.send_message.assert_called_once()
        sent_message = mock_server.send_message.call_args[0][0]
        assert sent_message.is_multipart()

    def test_attachment_size_limit_exceeded(self):
        """Test that large attachments are rejected"""
        # Create config with small attachment limit
        host_config = EmailHostConfig(
            host="smtp.example.com",
            port=465,
            use_ssl=True,
            username="test@example.com",
            password_env="TEST_PASSWORD",
            from_email="no-reply@example.com",
            max_attachment_mb=1  # 1MB limit
        )
        
        config = FullEmailConfig(
            email=EmailConfig(default_host="primary", hosts={"primary": host_config}),
            settings=self.settings
        )
        
        service = EmailService(config)
        
        # Create large attachment (2MB)
        large_content = b"x" * (2 * 1024 * 1024)
        
        with pytest.raises(EmailSendError, match="Failed to build email message"):
            service.send_email(
                to="recipient@example.com",
                subject="Test large attachment",
                text="Message",
                attachments=[("large.bin", large_content, "application/octet-stream")]
            )

    @patch.dict('os.environ', {'TEST_PASSWORD': 'secret123'})
    @patch('smtplib.SMTP_SSL')
    def test_send_email_with_cc_bcc(self, mock_smtp_ssl):
        """Test sending email with CC and BCC recipients"""
        mock_server = MagicMock()
        mock_smtp_ssl.return_value.__enter__.return_value = mock_server
        
        service = EmailService(self.full_config)
        
        service.send_email(
            to="recipient@example.com",
            subject="Test with CC/BCC",
            text="Test message",
            cc=["cc@example.com"],
            bcc=["bcc@example.com"]
        )
        
        # Verify message was sent
        mock_server.send_message.assert_called_once()
        args, kwargs = mock_server.send_message.call_args
        sent_message = args[0]
        
        # Check headers (BCC should not be in headers)
        assert sent_message["To"] == "recipient@example.com"
        assert sent_message["Cc"] == "cc@example.com"
        assert "Bcc" not in sent_message  # BCC should not be in headers
        
        # Check that to_addrs includes all recipients (including BCC)
        to_addrs = kwargs.get('to_addrs', [])
        assert "recipient@example.com" in to_addrs
        assert "cc@example.com" in to_addrs
        assert "bcc@example.com" in to_addrs

    @patch.dict('os.environ', {'TEST_PASSWORD': 'secret123'})
    @patch('smtplib.SMTP')
    def test_send_email_with_starttls(self, mock_smtp):
        """Test sending email with STARTTLS"""
        # Create STARTTLS config
        host_config = EmailHostConfig(
            host="smtp.example.com",
            port=587,
            use_ssl=False,
            use_starttls=True,
            username="test@example.com",
            password_env="TEST_PASSWORD",
            from_email="no-reply@example.com",
        )
        
        config = FullEmailConfig(
            email=EmailConfig(default_host="primary", hosts={"primary": host_config}),
            settings=self.settings
        )
        
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        service = EmailService(config)
        
        service.send_email(
            to="recipient@example.com",
            subject="Test STARTTLS",
            text="Test message"
        )
        
        # Verify STARTTLS was called
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()

    def test_missing_password_error(self):
        """Test error when password environment variable is missing"""
        service = EmailService(self.full_config)
        
        with pytest.raises(EmailSendError, match="Failed to send email"):
            service.send_email(
                to="recipient@example.com",
                subject="Test",
                text="Test message"
            )

    @patch.dict('os.environ', {'TEST_PASSWORD': 'secret123'})
    @patch('smtplib.SMTP_SSL')
    def test_smtp_connection_error(self, mock_smtp_ssl):
        """Test handling of SMTP connection errors"""
        mock_smtp_ssl.side_effect = ConnectionError("Connection failed")
        
        service = EmailService(self.full_config)
        
        with pytest.raises(EmailSendError, match="Failed to send email"):
            service.send_email(
                to="recipient@example.com",
                subject="Test",
                text="Test message"
            )

    def test_invalid_attachment_format(self):
        """Test error with invalid attachment format"""
        service = EmailService(self.full_config)
        
        with pytest.raises(EmailSendError, match="Failed to build email message"):
            service.send_email(
                to="recipient@example.com",
                subject="Test",
                text="Test message",
                attachments=["invalid_format"]  # Should be Path or tuple
            )

    def test_nonexistent_attachment_file(self):
        """Test error with nonexistent attachment file"""
        service = EmailService(self.full_config)
        
        nonexistent_path = Path("/nonexistent/file.txt")
        
        with pytest.raises(EmailSendError, match="Failed to build email message"):
            service.send_email(
                to="recipient@example.com",
                subject="Test",
                text="Test message",
                attachments=[nonexistent_path]
            )

    @patch.dict('os.environ', {'TEST_PASSWORD': 'secret123'})
    @patch('smtplib.SMTP_SSL')
    def test_send_email_with_custom_headers(self, mock_smtp_ssl):
        """Test sending email with custom headers"""
        mock_server = MagicMock()
        mock_smtp_ssl.return_value.__enter__.return_value = mock_server
        
        service = EmailService(self.full_config)
        
        custom_headers = {
            "X-Custom-Header": "test-value",
            "X-Priority": "high"
        }
        
        service.send_email(
            to="recipient@example.com",
            subject="Test with headers",
            text="Test message",
            headers=custom_headers
        )
        
        # Verify custom headers were added
        mock_server.send_message.assert_called_once()
        sent_message = mock_server.send_message.call_args[0][0]
        
        assert sent_message["X-Custom-Header"] == "test-value"
        assert sent_message["X-Priority"] == "high"

    def test_get_host_config_with_alias(self):
        """Test using specific host alias"""
        # Add second host
        backup_host = EmailHostConfig(
            host="backup.example.com",
            port=587,
            use_ssl=False,
            use_starttls=True,
            username="backup@example.com",
            password_env="BACKUP_PASSWORD",
            from_email="backup@example.com",
        )
        
        config = FullEmailConfig(
            email=EmailConfig(
                default_host="primary",
                hosts={"primary": self.host_config, "backup": backup_host}
            ),
            settings=self.settings
        )
        
        service = EmailService(config)
        
        # Should use backup host when specified
        with patch.dict('os.environ', {'BACKUP_PASSWORD': 'backup123'}):
            with patch('smtplib.SMTP') as mock_smtp:
                mock_server = MagicMock()
                mock_smtp.return_value.__enter__.return_value = mock_server
                
                service.send_email(
                    to="recipient@example.com",
                    subject="Test backup host",
                    text="Test message",
                    host_alias="backup"
                )
                
                # Verify backup host was used
                mock_smtp.assert_called_once_with(
                    host="backup.example.com",
                    port=587,
                    timeout=10,
                    context=None
                )


class TestEmailServiceUtilities:
    """Test utility functions for email service"""

    def test_get_email_service_with_config(self):
        """Test getting email service when configured"""
        with patch('inference_core.services.email_service.is_email_configured', return_value=True):
            with patch('inference_core.services.email_service.get_email_config') as mock_get_config:
                mock_config = Mock()
                mock_get_config.return_value = mock_config
                
                from inference_core.services.email_service import get_email_service
                service = get_email_service()
                
                assert service is not None
                assert isinstance(service, EmailService)

    def test_get_email_service_not_configured(self):
        """Test getting email service when not configured"""
        with patch('inference_core.services.email_service.is_email_configured', return_value=False):
            from inference_core.services.email_service import get_email_service
            service = get_email_service()
            
            assert service is None

    def test_get_email_service_init_error(self):
        """Test handling of email service initialization errors"""
        with patch('inference_core.services.email_service.is_email_configured', return_value=True):
            with patch('inference_core.services.email_service.EmailService') as mock_service:
                mock_service.side_effect = Exception("Init failed")
                
                from inference_core.services.email_service import get_email_service
                service = get_email_service()
                
                assert service is None
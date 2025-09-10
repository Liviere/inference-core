#!/usr/bin/env python3
"""
Email Service Demo

This script demonstrates how to use the email delivery service.
Run with: poetry run python scripts/email_demo.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference_core.core.email_config import is_email_configured, get_email_config
from inference_core.services.email_service import get_email_service
from inference_core.celery.tasks.email_tasks import send_email_async, encode_attachment


def main():
    print("🚀 Email Service Demo")
    print("=" * 50)
    
    # Check if email is configured
    print("1. Checking email configuration...")
    if not is_email_configured():
        print("❌ Email not configured. Please:")
        print("   - Copy email_config.example.yaml to email_config.yaml")
        print("   - Set SMTP credentials in environment variables")
        return
    
    config = get_email_config()
    print(f"✅ Email configured with default host: {config.email.default_host}")
    print(f"   Available hosts: {config.email.list_host_aliases()}")
    
    # Check if email service is available
    print("\n2. Initializing email service...")
    service = get_email_service()
    if not service:
        print("❌ Email service not available")
        return
    
    print("✅ Email service ready")
    
    # Demo: Check if we have credentials for demo
    print("\n3. Checking credentials...")
    host_config = config.email.get_host_config()
    if not host_config.get_password():
        print(f"⚠️  No password found for environment variable: {host_config.password_env}")
        print("   This demo will show the API but not actually send emails")
        demo_mode = True
    else:
        print("✅ SMTP credentials found")
        demo_mode = False
    
    # Demo: Basic email sending (direct)
    print("\n4. Email Service API Demo:")
    print("   service.send_email()")
    print("   ├── to: recipient@example.com")
    print("   ├── subject: Test Email")
    print("   ├── text: Plain text content")
    print("   ├── html: <p>HTML content</p>")
    print("   └── attachments: [file attachments]")
    
    if not demo_mode:
        try:
            # Uncomment to actually send email:
            # message_id = service.send_email(
            #     to="recipient@example.com",
            #     subject="Test Email from Inference Core",
            #     text="This is a test email from the Inference Core email service.",
            #     html="<p>This is a <strong>test email</strong> from the Inference Core email service.</p>"
            # )
            # print(f"✅ Email sent! Message ID: {message_id}")
            print("   (Email sending disabled in demo - uncomment code to actually send)")
        except Exception as e:
            print(f"❌ Email sending failed: {e}")
    
    # Demo: Celery async email
    print("\n5. Celery Async Email API Demo:")
    print("   send_email_async()")
    print("   ├── Queues email for background processing")
    print("   ├── Automatic retry on transient failures")
    print("   ├── Supports all email features (HTML, attachments)")
    print("   └── Returns Celery task object")
    
    # Demo: Attachment encoding
    print("\n6. Attachment Demo:")
    attachment_content = b"Hello, this is a test file!"
    encoded = encode_attachment("test.txt", attachment_content, "text/plain")
    print(f"   Original: {len(attachment_content)} bytes")
    print(f"   Encoded:  {len(encoded['content_b64'])} characters (base64)")
    print(f"   MIME:     {encoded['mime']}")
    
    # Demo: Template usage (from AuthService)
    print("\n7. Template System:")
    print("   Templates located in: templates/email/")
    print("   ├── reset_password.txt  (plain text)")
    print("   ├── reset_password.html (HTML with styling)")
    print("   └── Uses Jinja2 for variable substitution")
    
    # Demo: Configuration examples
    print("\n8. Configuration Examples:")
    print("\n   Gmail (SSL):")
    print("   host: smtp.gmail.com")
    print("   port: 465")
    print("   use_ssl: true")
    
    print("\n   Office 365 (STARTTLS):")
    print("   host: smtp.office365.com")
    print("   port: 587")
    print("   use_starttls: true")
    
    print("\n   Mailgun (STARTTLS):")
    print("   host: smtp.mailgun.org")
    print("   port: 587")
    print("   use_starttls: true")
    
    # Demo: Security features
    print("\n9. Security Features:")
    print("   ✅ No passwords in configuration files")
    print("   ✅ SSL/TLS encryption enforced")
    print("   ✅ Hostname verification by default")
    print("   ✅ Attachment size limits")
    print("   ✅ Rate limiting support")
    print("   ✅ Structured logging (no sensitive data)")
    
    # Demo: Production deployment
    print("\n10. Production Deployment:")
    print("    # Start mail queue worker")
    print("    poetry run celery -A inference_core.celery.celery_main:celery_app \\")
    print("                      worker --loglevel=info --queues=mail")
    print()
    print("    # Monitor with Flower")
    print("    poetry run celery -A inference_core.celery.celery_main:celery_app \\")
    print("                      flower --port=5555")
    
    print("\n" + "=" * 50)
    print("✨ Demo complete! Email service is ready for production use.")
    print("\nNext steps:")
    print("1. Configure your SMTP credentials")
    print("2. Start the Celery mail worker")
    print("3. Test with password reset: POST /api/v1/auth/reset-password")


if __name__ == "__main__":
    main()
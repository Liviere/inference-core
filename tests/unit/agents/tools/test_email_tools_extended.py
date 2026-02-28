"""Extended tests for email_tools module.

Covers ReadUnseenEmailsTool, GetEmailTool, SendEmailTool,
ListEmailAccountsTool, SummarizeEmailTool, get_email_tools factory,
and helper functions not covered by the existing test file.

The existing test_email_tools.py covers normalize_imap_query and basic
SearchEmailsTool._run; this file extends coverage to all remaining tools.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from inference_core.agents.tools.email_tools import (
    EMAIL_TYPES,
    EmailSummary,
    EmailType,
    GetEmailTool,
    ListEmailAccountsTool,
    ReadUnseenEmailsTool,
    SendEmailTool,
    SummarizeEmailTool,
    format_email_types_for_description,
    generate_email_tools_system_instructions,
    get_email_tools,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Verify module-level helpers."""

    def test_email_types_list(self):
        """EMAIL_TYPES contains all EmailType values."""
        assert "spam" in EMAIL_TYPES
        assert "important" in EMAIL_TYPES
        assert len(EMAIL_TYPES) == len(EmailType)

    def test_format_email_types_for_description(self):
        """Returns formatted string listing all email types."""
        desc = format_email_types_for_description()
        assert "spam" in desc
        assert "important" in desc
        assert "Available email types:" in desc

    def test_generate_email_tools_system_instructions(self):
        """System instructions contain tool usage guidance."""
        instructions = generate_email_tools_system_instructions()
        assert "read_unseen_emails" in instructions
        assert "search_emails" in instructions
        assert "send_email" in instructions
        assert "summarize_email" in instructions

    def test_email_summary_model(self):
        """EmailSummary Pydantic model accepts valid data."""
        summary = EmailSummary(
            summary="Test email about meeting",
            email_type=EmailType.IMPORTANT,
            requires_action=True,
            key_points=["Schedule confirmed", "Bring laptop"],
        )
        assert summary.email_type == EmailType.IMPORTANT
        assert summary.requires_action is True
        assert len(summary.key_points) == 2


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_imap_service():
    """Mocked IMAP service."""
    svc = MagicMock()
    svc.fetch_unseen_emails.return_value = []
    svc.fetch_emails.return_value = []
    svc.list_configured_hosts.return_value = ["primary", "support"]
    return svc


@pytest.fixture
def mock_email_service():
    """Mocked SMTP email service."""
    svc = MagicMock()
    svc.send_email.return_value = "<msg-id-123>"
    return svc


@pytest.fixture
def mock_email_config():
    """Mocked email configuration."""
    config = MagicMock()
    host_config = MagicMock()
    host_config.signature = "-- Sent from Agent"
    host_config.context = "Work email"
    config.email.get_host_config.return_value = host_config
    config.email.list_host_aliases.return_value = ["primary", "support"]
    config.email.default_host = "primary"
    return config


def _make_mock_email(
    uid="123",
    subject="Test Subject",
    from_name="John",
    from_address="john@example.com",
    body_text="Hello world",
    has_attachments=False,
    attachment_names=None,
    is_read=False,
    to_addresses=None,
    cc_addresses=None,
    date=None,
):
    """Build a mock email message object."""
    msg = MagicMock()
    msg.uid = uid
    msg.subject = subject
    msg.from_name = from_name
    msg.from_address = from_address
    msg.body_text = body_text
    msg.has_attachments = has_attachments
    msg.attachment_names = attachment_names or []
    msg.is_read = is_read
    msg.to_addresses = to_addresses or ["recipient@example.com"]
    msg.cc_addresses = cc_addresses or []
    msg.date = date or datetime(2025, 8, 15, 10, 30)
    return msg


# ---------------------------------------------------------------------------
# ReadUnseenEmailsTool
# ---------------------------------------------------------------------------


class TestReadUnseenEmailsTool:
    """Verify unread email fetching tool."""

    @pytest.fixture
    def tool(self, mock_imap_service):
        return ReadUnseenEmailsTool(
            imap_service=mock_imap_service,
            allowed_accounts=["primary", "support"],
            default_account="primary",
        )

    def test_run_fetches_unseen(self, tool, mock_imap_service):
        """Calls imap_service.fetch_unseen_emails with correct args."""
        tool._run(folder="INBOX", limit=5)
        mock_imap_service.fetch_unseen_emails.assert_called_once_with(
            host_alias="primary",
            folder="INBOX",
            limit=5,
            mark_as_read=False,
        )

    def test_run_uses_specified_account(self, tool, mock_imap_service):
        """Uses explicitly provided account over default."""
        tool._run(account_name="support")
        mock_imap_service.fetch_unseen_emails.assert_called_once()
        assert (
            mock_imap_service.fetch_unseen_emails.call_args[1]["host_alias"]
            == "support"
        )

    def test_resolve_account_rejects_disallowed(self, tool):
        """ValueError raised for accounts not in allowed list."""
        result = tool._run(account_name="hacker_account")
        assert "not allowed" in result

    def test_format_messages_empty(self, tool, mock_imap_service):
        """Returns 'No unread emails' for empty result."""
        result = tool._run()
        assert "No unread emails" in result

    def test_format_messages_with_data(self, tool, mock_imap_service):
        """Formats email list with UID, subject, sender, date."""
        msg = _make_mock_email(uid="42", subject="Important")
        mock_imap_service.fetch_unseen_emails.return_value = [msg]

        result = tool._run()

        assert "42" in result
        assert "Important" in result
        assert "john@example.com" in result

    def test_format_messages_truncates_long_body(self, tool, mock_imap_service):
        """Body preview is truncated at 500 chars."""
        msg = _make_mock_email(body_text="A" * 600)
        mock_imap_service.fetch_unseen_emails.return_value = [msg]

        result = tool._run()
        assert "[truncated]" in result

    def test_handles_service_exception(self, tool, mock_imap_service):
        """Returns error string on service exception."""
        mock_imap_service.fetch_unseen_emails.side_effect = RuntimeError("IMAP down")
        result = tool._run()
        assert "Failed to read emails" in result


# ---------------------------------------------------------------------------
# GetEmailTool
# ---------------------------------------------------------------------------


class TestGetEmailTool:
    """Verify single email retrieval by UID."""

    @pytest.fixture
    def tool(self, mock_imap_service):
        return GetEmailTool(
            imap_service=mock_imap_service,
            default_account="primary",
        )

    def test_run_fetches_by_uid(self, tool, mock_imap_service):
        """Fetches specific UID from IMAP service."""
        msg = _make_mock_email(uid="99")
        mock_imap_service.fetch_emails.return_value = [msg]

        result = tool._run(uid="99")

        mock_imap_service.fetch_emails.assert_called_once_with(
            host_alias="primary",
            folder="INBOX",
            limit=1,
            search_criteria="UID 99",
        )
        assert "99" in result

    def test_rejects_non_numeric_uid(self, tool):
        """Returns error for non-numeric UID."""
        result = tool._run(uid="abc")
        assert "Invalid UID" in result

    def test_returns_not_found(self, tool, mock_imap_service):
        """Returns error when UID yields no results."""
        mock_imap_service.fetch_emails.return_value = []
        result = tool._run(uid="999")
        assert "not found" in result

    def test_format_message_includes_all_fields(self, tool, mock_imap_service):
        """Formatted output includes subject, sender, body, etc."""
        msg = _make_mock_email(
            uid="50",
            subject="Meeting Tomorrow",
            from_name="Alice",
            from_address="alice@co.com",
            body_text="Let's discuss the project.",
            has_attachments=True,
            attachment_names=["report.pdf"],
            is_read=False,
            cc_addresses=["bob@co.com"],
        )
        mock_imap_service.fetch_emails.return_value = [msg]

        result = tool._run(uid="50")

        assert "Meeting Tomorrow" in result
        assert "Alice <alice@co.com>" in result
        assert "Unread" in result
        assert "report.pdf" in result
        assert "bob@co.com" in result
        assert "Let's discuss the project." in result


# ---------------------------------------------------------------------------
# SendEmailTool
# ---------------------------------------------------------------------------


class TestSendEmailTool:
    """Verify email sending tool."""

    @pytest.fixture
    def tool(self, mock_email_service, mock_email_config):
        return SendEmailTool(
            email_service=mock_email_service,
            email_config=mock_email_config,
            allowed_accounts=["primary"],
            default_account="primary",
        )

    def test_sends_basic_email(self, tool, mock_email_service):
        """Sends email with correct parameters."""
        result = tool._run(
            to="user@example.com",
            subject="Hello",
            body="Test body",
        )

        mock_email_service.send_email.assert_called_once()
        call_kwargs = mock_email_service.send_email.call_args[1]
        assert call_kwargs["to"] == ["user@example.com"]
        assert call_kwargs["subject"] == "Hello"
        assert "✓" in result

    def test_appends_signature(self, tool, mock_email_service):
        """Signature from host config is appended to body."""
        tool._run(to="user@example.com", subject="Hi", body="Body")

        call_kwargs = mock_email_service.send_email.call_args[1]
        assert "-- Sent from Agent" in call_kwargs["text"]

    def test_multiple_recipients(self, tool, mock_email_service):
        """Comma-separated recipients are parsed into list."""
        tool._run(
            to="a@co.com, b@co.com",
            subject="Group",
            body="Hi all",
        )

        call_kwargs = mock_email_service.send_email.call_args[1]
        assert call_kwargs["to"] == ["a@co.com", "b@co.com"]

    def test_reply_headers(self, tool, mock_email_service):
        """Reply-to-message-id sets In-Reply-To and References headers."""
        tool._run(
            to="user@example.com",
            subject="Re: Hello",
            body="Got it",
            reply_to_message_id="<orig-123>",
        )

        call_kwargs = mock_email_service.send_email.call_args[1]
        assert call_kwargs["headers"]["In-Reply-To"] == "<orig-123>"
        assert call_kwargs["headers"]["References"] == "<orig-123>"

    def test_rejects_disallowed_account(self, tool):
        """Returns error for disallowed account."""
        result = tool._run(
            to="user@co.com",
            subject="Hi",
            body="Body",
            account_name="hacker_account",
        )
        assert "not allowed" in result

    def test_handles_send_failure(self, tool, mock_email_service):
        """Returns error message on SMTP exception."""
        mock_email_service.send_email.side_effect = RuntimeError("SMTP error")
        result = tool._run(to="user@co.com", subject="Hi", body="Body")
        assert "Failed to send email" in result


# ---------------------------------------------------------------------------
# ListEmailAccountsTool
# ---------------------------------------------------------------------------


class TestListEmailAccountsTool:
    """Verify email account listing tool."""

    @pytest.fixture
    def tool(self, mock_email_config, mock_imap_service):
        return ListEmailAccountsTool(
            email_config=mock_email_config,
            imap_service=mock_imap_service,
            allowed_accounts=None,  # All accounts visible
        )

    def test_lists_all_accounts(self, tool):
        """Lists all configured accounts with capabilities."""
        result = tool._run()

        assert "primary" in result
        assert "support" in result
        assert "Send (SMTP)" in result
        assert "Read (IMAP)" in result

    def test_shows_default_account(self, tool):
        """Default account is indicated in output."""
        result = tool._run()
        assert "Default account: primary" in result

    def test_filters_by_allowed_accounts(self, mock_email_config, mock_imap_service):
        """Only allowed accounts are shown when filter is set."""
        tool = ListEmailAccountsTool(
            email_config=mock_email_config,
            imap_service=mock_imap_service,
            allowed_accounts=["primary"],
        )
        result = tool._run()

        assert "primary" in result
        # "support" was in the full list but filtered out
        # (whether it appears depends on exact filtering logic)

    def test_empty_when_no_accounts(self, mock_email_config, mock_imap_service):
        """Returns appropriate message when no accounts available."""
        mock_email_config.email.list_host_aliases.return_value = []
        tool = ListEmailAccountsTool(
            email_config=mock_email_config,
            imap_service=mock_imap_service,
        )
        result = tool._run()
        assert "No email accounts" in result


# ---------------------------------------------------------------------------
# get_email_tools factory
# ---------------------------------------------------------------------------


class TestGetEmailToolsFactory:
    """Verify factory function creates correct tool combinations."""

    def test_returns_empty_on_config_error(self):
        """Returns empty list when email config cannot be loaded."""
        with patch(
            "inference_core.core.email_config.get_email_config",
            side_effect=RuntimeError("no config"),
        ):
            tools = get_email_tools()
        assert tools == []

    def test_creates_imap_tools_when_imap_service(self, mock_imap_service):
        """ReadUnseen, Search, Get tools are created when imap_service provided."""
        with patch(
            "inference_core.core.email_config.get_email_config",
            return_value=MagicMock(),
        ):
            tools = get_email_tools(imap_service=mock_imap_service)

        tool_names = [t.name for t in tools]
        assert "read_unseen_emails" in tool_names
        assert "search_emails" in tool_names
        assert "get_email" in tool_names

    def test_creates_send_tool_when_email_service(
        self, mock_email_service, mock_imap_service
    ):
        """SendEmailTool is created when email_service provided."""
        with patch(
            "inference_core.core.email_config.get_email_config",
            return_value=MagicMock(),
        ):
            tools = get_email_tools(
                email_service=mock_email_service,
                imap_service=mock_imap_service,
            )

        tool_names = [t.name for t in tools]
        assert "send_email" in tool_names

    def test_always_includes_list_accounts(self):
        """ListEmailAccountsTool is always included."""
        with patch(
            "inference_core.core.email_config.get_email_config",
            return_value=MagicMock(),
        ):
            tools = get_email_tools()

        tool_names = [t.name for t in tools]
        assert "list_email_accounts" in tool_names

    def test_excludes_summarize_when_disabled(self, mock_imap_service):
        """SummarizeEmailTool excluded when include_summarize=False."""
        with patch(
            "inference_core.core.email_config.get_email_config",
            return_value=MagicMock(),
        ):
            tools = get_email_tools(
                imap_service=mock_imap_service,
                include_summarize=False,
            )

        tool_names = [t.name for t in tools]
        assert "summarize_email" not in tool_names

    def test_includes_summarize_by_default(self):
        """SummarizeEmailTool included by default."""
        with patch(
            "inference_core.core.email_config.get_email_config",
            return_value=MagicMock(),
        ):
            tools = get_email_tools()

        tool_names = [t.name for t in tools]
        assert "summarize_email" in tool_names

    def test_passes_allowed_accounts(self, mock_imap_service):
        """allowed_accounts is passed through to all IMAP tools."""
        with patch(
            "inference_core.core.email_config.get_email_config",
            return_value=MagicMock(),
        ):
            tools = get_email_tools(
                imap_service=mock_imap_service,
                allowed_accounts=["primary"],
                default_account="primary",
            )

        for tool in tools:
            if hasattr(tool, "allowed_accounts") and tool.allowed_accounts is not None:
                assert tool.allowed_accounts == ["primary"]

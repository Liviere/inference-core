"""Tests for EmailToolsProvider and register_email_tools_provider.

Covers: get_tools with/without services, register convenience function.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_core.agents.tools.email_provider import (
    EmailToolsProvider,
    register_email_tools_provider,
)


class TestEmailToolsProvider:
    """Test EmailToolsProvider.get_tools."""

    def test_init_stores_params(self):
        """Constructor stores configuration params."""
        provider = EmailToolsProvider(
            allowed_accounts=["primary"],
            default_account="primary",
            include_summarize=False,
            summarize_model="gpt-5",
        )
        assert provider.name == "email_tools"
        assert provider.allowed_accounts == ["primary"]
        assert provider.default_account == "primary"
        assert provider.include_summarize is False
        assert provider.summarize_model == "gpt-5"

    @patch("inference_core.agents.tools.email_tools.get_email_tools")
    @patch("inference_core.services.imap_service.get_imap_service")
    @patch("inference_core.services.email_service.get_email_service")
    async def test_get_tools_returns_tools(
        self, mock_email_svc, mock_imap_svc, mock_get_tools
    ):
        """get_tools returns the list from get_email_tools when services available."""
        mock_email_svc.return_value = MagicMock()
        mock_imap_svc.return_value = MagicMock()
        mock_get_tools.return_value = [MagicMock(), MagicMock()]

        provider = EmailToolsProvider(
            allowed_accounts=["primary"],
            default_account="primary",
        )
        tools = await provider.get_tools(task_type="email_agent")
        assert len(tools) == 2
        mock_get_tools.assert_called_once()

    @patch("inference_core.services.imap_service.get_imap_service")
    @patch("inference_core.services.email_service.get_email_service")
    async def test_get_tools_no_services(self, mock_email_svc, mock_imap_svc):
        """get_tools returns empty list when no services available."""
        mock_email_svc.return_value = None
        mock_imap_svc.return_value = None

        provider = EmailToolsProvider()
        tools = await provider.get_tools(task_type="test")
        assert tools == []

    @patch("inference_core.agents.tools.email_tools.get_email_tools")
    @patch("inference_core.services.imap_service.get_imap_service")
    @patch("inference_core.services.email_service.get_email_service")
    async def test_get_tools_exception_returns_empty(
        self, mock_email_svc, mock_imap_svc, mock_get_tools
    ):
        """get_tools returns empty list on unexpected exception."""
        mock_email_svc.side_effect = RuntimeError("import error")

        provider = EmailToolsProvider()
        tools = await provider.get_tools(task_type="test")
        assert tools == []


class TestRegisterEmailToolsProvider:
    """Test register_email_tools_provider convenience function."""

    @patch("inference_core.llm.tools.register_tool_provider")
    def test_registers_provider(self, mock_register):
        """Convenience function creates and registers EmailToolsProvider."""
        register_email_tools_provider(
            allowed_accounts=["main"],
            default_account="main",
        )
        mock_register.assert_called_once()
        provider = mock_register.call_args[0][0]
        assert isinstance(provider, EmailToolsProvider)
        assert provider.allowed_accounts == ["main"]

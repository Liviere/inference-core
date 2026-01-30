"""
Email Tools Provider

Pluggable tool provider for email operations (reading and sending).
Integrates with the LLM service tool provider system.

Register this provider during application startup to make email tools
available to agents configured with 'email_tools' in local_tool_providers.

Usage:
    from inference_core.llm.tools import register_tool_provider
    from inference_core.agents.tools.email_provider import EmailToolsProvider

    # Register with default settings
    register_tool_provider(EmailToolsProvider())

    # Or with specific allowed accounts
    register_tool_provider(EmailToolsProvider(
        allowed_accounts=['primary', 'support'],
        default_account='primary',
    ))

    # In llm_config.yaml:
    agents:
      email_agent:
        local_tool_providers: ['email_tools']
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class EmailToolsProvider:
    """Tool provider for email operations.

    Provides email tools (read, send, search, summarize) to agents.
    Respects account restrictions for security.

    Attributes:
        name: Provider name for registration ('email_tools')
        allowed_accounts: List of account aliases the agent can access
        default_account: Default account when not specified
        include_summarize: Whether to include email summarization tool
        summarize_model: Model to use for summarization
    """

    name: str = "email_tools"

    def __init__(
        self,
        allowed_accounts: Optional[List[str]] = None,
        default_account: Optional[str] = None,
        include_summarize: bool = True,
        summarize_model: str = "gpt-5-nano",
    ):
        """Initialize email tools provider.

        Args:
            allowed_accounts: List of account aliases agents can access.
                              If None, all configured accounts are allowed.
            default_account: Default account to use when not specified.
            include_summarize: Whether to include SummarizeEmailTool.
            summarize_model: Model name for email summarization.
        """
        self.allowed_accounts = allowed_accounts
        self.default_account = default_account
        self.include_summarize = include_summarize
        self.summarize_model = summarize_model

    async def get_tools(self, task_type: str, **kwargs) -> List[Any]:
        """Get email tools for the agent.

        Args:
            task_type: Task type or agent name (not used for filtering)
            **kwargs: Additional context (user_id, session_id, etc.)

        Returns:
            List of configured email BaseTool instances
        """
        try:
            from inference_core.agents.tools.email_tools import get_email_tools
            from inference_core.services.email_service import get_email_service
            from inference_core.services.imap_service import get_imap_service

            email_service = get_email_service()
            imap_service = get_imap_service()

            if not email_service and not imap_service:
                logger.warning(
                    "Neither email nor IMAP service available - "
                    "email tools will not be provided"
                )
                return []

            tools = get_email_tools(
                email_service=email_service,
                imap_service=imap_service,
                allowed_accounts=self.allowed_accounts,
                default_account=self.default_account,
                include_summarize=self.include_summarize,
                summarize_model=self.summarize_model,
            )

            logger.debug(
                "EmailToolsProvider returning %d tools for task '%s'",
                len(tools),
                task_type,
            )
            return tools

        except Exception as e:
            logger.error("Failed to get email tools: %s", e, exc_info=True)
            return []


def register_email_tools_provider(
    allowed_accounts: Optional[List[str]] = None,
    default_account: Optional[str] = None,
    include_summarize: bool = True,
    summarize_model: str = "gpt-5-nano",
) -> None:
    """Convenience function to register email tools provider.

    Args:
        allowed_accounts: List of account aliases agents can access
        default_account: Default account to use
        include_summarize: Whether to include summarization tool
        summarize_model: Model for summarization

    Example:
        # In application startup
        from inference_core.agents.tools.email_provider import (
            register_email_tools_provider
        )

        register_email_tools_provider(
            allowed_accounts=['primary'],
            default_account='primary',
        )
    """
    from inference_core.llm.tools import register_tool_provider

    provider = EmailToolsProvider(
        allowed_accounts=allowed_accounts,
        default_account=default_account,
        include_summarize=include_summarize,
        summarize_model=summarize_model,
    )

    register_tool_provider(provider)
    logger.info(
        "Registered EmailToolsProvider (accounts: %s, default: %s)",
        allowed_accounts or "all",
        default_account or "config default",
    )


__all__ = [
    "EmailToolsProvider",
    "register_email_tools_provider",
]

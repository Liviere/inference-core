"""Shared cancellation exception for agent execution."""


class AgentCancelled(Exception):
    """Raised when agent execution is cancelled via a cancel_check callback."""

"""Shared cancellation exception for agent execution."""

from typing import Any, Optional


class AgentCancelled(Exception):
    """Raised when agent execution is cancelled via a cancel_check callback.

    Attributes:
        partial_result: The last ``updates`` result dict collected before
            cancellation (contains ``messages`` with partial AI output).
            May be empty if no update chunks were received yet.
    """

    partial_result: dict[str, Any]

    def __init__(
        self,
        message: str = "Agent execution cancelled",
        partial_result: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.partial_result = partial_result or {}

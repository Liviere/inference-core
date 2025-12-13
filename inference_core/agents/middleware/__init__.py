"""
Middleware package for LangChain v1 agents.

This package provides middleware components for agent execution control,
including cost tracking, usage logging, memory injection, and other
cross-cutting concerns.

Usage:
    from inference_core.agents.middleware import (
        CostTrackingMiddleware,
        CostTrackingState,
        create_cost_tracking_middleware,
        MemoryMiddleware,
        MemoryState,
        create_memory_middleware,
    )

    # Create cost tracking middleware with defaults
    cost_middleware = create_cost_tracking_middleware(user_id=user_uuid)

    # Create memory middleware
    memory_middleware = create_memory_middleware(
        memory_service=memory_service,
        user_id="user-uuid",
    )
"""

from .cost_tracking import (
    CostTrackingMiddleware,
    CostTrackingState,
    create_cost_tracking_middleware,
)
from .memory import MemoryMiddleware, MemoryState, create_memory_middleware

__all__ = [
    # Cost tracking
    "CostTrackingMiddleware",
    "CostTrackingState",
    "create_cost_tracking_middleware",
    # Memory
    "MemoryMiddleware",
    "MemoryState",
    "create_memory_middleware",
]

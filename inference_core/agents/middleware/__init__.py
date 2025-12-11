"""
Middleware package for LangChain v1 agents.

This package provides middleware components for agent execution control,
including cost tracking, usage logging, and other cross-cutting concerns.

Usage:
    from inference_core.agents.middleware import (
        CostTrackingMiddleware,
        CostTrackingState,
        create_cost_tracking_middleware,
    )

    # Create middleware with defaults
    middleware = create_cost_tracking_middleware(user_id=user_uuid)

    # Or configure manually
    middleware = CostTrackingMiddleware(
        pricing_config=pricing_config,
        user_id=user_uuid,
        task_type="agent",
    )
"""

from .cost_tracking import (
    CostTrackingMiddleware,
    CostTrackingState,
    create_cost_tracking_middleware,
)

__all__ = [
    "CostTrackingMiddleware",
    "CostTrackingState",
    "create_cost_tracking_middleware",
]

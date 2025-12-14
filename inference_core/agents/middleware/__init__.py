"""
Middleware package for LangChain v1 agents.

This package provides middleware components for agent execution control,
including cost tracking, usage logging, memory injection, tool-based model
switching, and other cross-cutting concerns.

Usage:
    from inference_core.agents.middleware import (
        CostTrackingMiddleware,
        CostTrackingState,
        create_cost_tracking_middleware,
        MemoryMiddleware,
        MemoryState,
        create_memory_middleware,
        ToolBasedModelSwitchMiddleware,
        ToolModelSwitchConfig,
        ToolModelOverride,
        create_tool_model_switch_middleware,
    )

    # Create cost tracking middleware with defaults
    cost_middleware = create_cost_tracking_middleware(user_id=user_uuid)

    # Create memory middleware
    memory_middleware = create_memory_middleware(
        memory_service=memory_service,
        user_id="user-uuid",
    )

    # Create tool-based model switch middleware
    model_switch_middleware = create_tool_model_switch_middleware(
        overrides=[
            {
                'tool_name': 'complex_analysis',
                'model': 'claude-opus-4-1-20250805',
                'trigger': 'after_tool',
            }
        ],
        default_model='gpt-5-mini',
    )
"""

from .cost_tracking import (
    CostTrackingMiddleware,
    CostTrackingState,
    create_cost_tracking_middleware,
)
from .memory import MemoryMiddleware, MemoryState, create_memory_middleware
from .tool_model_switch import (
    ToolBasedModelSwitchMiddleware,
    ToolModelOverride,
    ToolModelSwitchConfig,
    ToolModelSwitchState,
    create_tool_model_switch_middleware,
)

__all__ = [
    # Cost tracking
    "CostTrackingMiddleware",
    "CostTrackingState",
    "create_cost_tracking_middleware",
    # Memory
    "MemoryMiddleware",
    "MemoryState",
    "create_memory_middleware",
    # Tool-based model switching
    "ToolBasedModelSwitchMiddleware",
    "ToolModelOverride",
    "ToolModelSwitchConfig",
    "ToolModelSwitchState",
    "create_tool_model_switch_middleware",
]

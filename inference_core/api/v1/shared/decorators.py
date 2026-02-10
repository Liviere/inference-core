"""
Shared decorators for API routes.

Provides common error handling and response processing decorators.
"""

import functools
import logging
from typing import Callable

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def handle_api_errors(func: Callable) -> Callable:
    """
    Decorator to handle common API errors with consistent formatting.
    
    Catches exceptions and converts them to appropriate HTTP responses:
    - HTTPException: Re-raises as-is to preserve status and detail
    - All other exceptions: Logs and converts to 500 Internal Server Error
    
    Usage:
        @router.get("/endpoint")
        @handle_api_errors
        async def my_endpoint():
            # ... endpoint logic
    
    Args:
        func: The async function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            # Re-raise HTTPException as-is to preserve status code and details
            raise
        except Exception as e:
            # Log unexpected errors and return generic 500
            logger.error(
                f"Unexpected error in {func.__name__}: {str(e)}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}",
            )
    
    return wrapper

"""
Jupyter Notebook Utilities

Helper functions for using inference_core in Jupyter notebooks where
event loops are frequently recreated.
"""

import logging

logger = logging.getLogger(__name__)


def reset_for_jupyter() -> None:
    """Reset all singletons for a fresh Jupyter notebook cell execution.

    Call this function at the START of your notebook cell when you encounter
    'Event loop is closed' or 'Task got Future attached to a different loop'
    errors after rerunning cells.

    This function clears:
    - Database engine and session maker
    - Usage logging pricing snapshot cache

    Example:
        >>> from inference_core.notebook_utils import reset_for_jupyter
        >>> reset_for_jupyter()
        >>>
        >>> # Now safely initialize your agent
        >>> from inference_core.services.agents_service import AgentService
        >>> agent_service = AgentService(agent_name="my_agent")
        >>> agent = await agent_service.create_agent()

    Alternative - use NullPool:
        For a more permanent solution, set DATABASE_POOL_CLASS=null in your
        .env file. This uses SQLAlchemy's NullPool which doesn't retain
        connections between requests, avoiding event loop issues entirely.

    Note:
        This is designed for development/notebook use only. In production,
        the application server manages a single event loop throughout its
        lifetime, so these issues don't occur.
    """
    # Reset database connections
    from inference_core.database.sql.connection import reset_database_for_new_event_loop

    reset_database_for_new_event_loop()

    # Reset usage logging cache
    from inference_core.llm.usage_logging import reset_usage_logging_cache

    reset_usage_logging_cache()

    # Clear settings cache to allow reconfiguration if needed
    from inference_core.core.config import get_settings

    get_settings.cache_clear()

    logger.info("Jupyter notebook environment reset complete")


def configure_for_jupyter() -> None:
    """Configure inference_core for optimal Jupyter notebook usage.

    Call this once at the start of your notebook to set up the environment
    for multi-cell execution without event loop issues.

    This function:
    - Sets DATABASE_POOL_CLASS=null to use NullPool
    - Clears any existing cached settings

    Example:
        >>> # First cell in notebook
        >>> from inference_core.notebook_utils import configure_for_jupyter
        >>> configure_for_jupyter()
        >>>
        >>> # Subsequent cells can create agents normally
        >>> from inference_core.services.agents_service import AgentService
        >>> agent_service = AgentService(agent_name="my_agent")

    Note:
        This modifies the environment variable, so it affects all subsequent
        imports and configurations in the notebook session.
    """
    import os

    # Set NullPool to avoid event loop binding issues
    os.environ["DATABASE_POOL_CLASS"] = "null"

    # Clear cached settings to pick up the new env var
    from inference_core.core.config import get_settings

    get_settings.cache_clear()

    # Reset any existing engine with old pool settings
    from inference_core.database.sql.connection import reset_database_for_new_event_loop

    reset_database_for_new_event_loop()

    logger.info(
        "Jupyter notebook environment configured with NullPool. "
        "Event loop issues should be avoided."
    )

"""
Celery Application Instance
"""

import logging
import os

from celery import Celery
from dotenv import load_dotenv

from inference_core.celery.config import CeleryConfig
from inference_core.core.config import get_settings

logger = logging.getLogger(__name__)


def create_celery_app() -> Celery:
    """Create and configure Celery application"""
    load_dotenv()
    load_dotenv("../../.env", override=True)
    settings = get_settings()

    celery_app = Celery(settings.app_name)

    # Load configuration
    celery_app.config_from_object(CeleryConfig)

    # Auto-discover tasks
    celery_app.autodiscover_tasks(["inference_core.celery.tasks"])

    # Explicitly include tasks modules (useful for testing/packaging)
    celery_app.conf.update(
        include=[
            "inference_core.celery.tasks.llm_tasks",
            "inference_core.celery.tasks.batch_tasks",
            "inference_core.celery.tasks.email_tasks",
        ]
    )

    return celery_app


# Create the Celery app instance
celery_app = create_celery_app()


# Task base configuration for shared functionality
class BaseTask(celery_app.Task):
    """Base task class with common functionality"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Task {task_id} failed: {exc}", exc_info=einfo)
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        logger.debug(f"Task {task_id} succeeded")
        super().on_success(retval, task_id, args, kwargs)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry"""
        logger.warning(f"Task {task_id} retrying: {exc}", exc_info=einfo)
        super().on_retry(exc, task_id, args, kwargs, einfo)


# Set the base task class
celery_app.Task = BaseTask

if os.getenv("DEBUG_CELERY") and int(os.getenv("DEBUG_CELERY")) == 1:
    import debugpy

    debugpy.listen(("0.0.0.0", 5678))
    print("🔴 Debugging is enabled. Waiting for debugger to attach...")
    debugpy.wait_for_client()

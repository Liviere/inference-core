"""
Celery task that keeps the model catalog current

Beat calls it every few minutes. Most runs read no provider at all: a
provider is read when its refresh interval has passed, and the task asks the
catalog itself whether it has. A schedule entry as long as the interval would
start counting anew with every restart of beat and might never come round.

Every run writes the catalog gauges from what is stored, so they stay fresh
between readings and also show what a command-line refresh changed.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from inference_core.celery.async_utils import run_in_worker_loop
from inference_core.celery.celery_main import celery_app
from inference_core.core.config import get_settings
from inference_core.core.redis_client import get_sync_redis
from inference_core.llm.catalog.drift import DRIFT_STATES
from inference_core.llm.catalog.reconcile import EVENT_TYPES
from inference_core.llm.catalog.service import refresh
from inference_core.llm.catalog.summary import summarize
from inference_core.llm.config import get_llm_config
from inference_core.observability.metrics import set_model_catalog_gauges

logger = logging.getLogger(__name__)

CATALOG_REFRESH_LOCK_KEY = "llm_catalog_refresh:lock"
CATALOG_REFRESH_LOCK_TIMEOUT = 600

# Thread pools do not enforce Celery's time limits, so the run bounds itself.
CATALOG_REFRESH_TIMEOUT_SECONDS = 240.0


async def _refresh_and_publish(
    interval: timedelta, read_timeout: float
) -> Dict[str, Any]:
    config = get_llm_config()
    now = datetime.now(timezone.utc)

    outcomes: Dict[str, str] = {}
    try:
        results = await asyncio.wait_for(
            refresh(
                config,
                only_due=True,
                interval=interval,
                read_timeout=read_timeout,
                now=now,
            ),
            CATALOG_REFRESH_TIMEOUT_SECONDS,
        )
        outcomes = {result.provider: result.outcome for result in results}
    except Exception as exc:
        # The gauges below still say what is stored, and how old it is.
        logger.warning("Model catalog refresh did not finish: %s", type(exc).__name__)

    summary = await summarize(config, interval=interval, now=now)
    set_model_catalog_gauges(
        providers=summary.providers,
        event_types=EVENT_TYPES,
        drift_states=DRIFT_STATES,
        last_success=summary.last_success,
        stale=summary.stale,
        model_counts=summary.model_counts,
        recent_events=summary.recent_events,
        configured=summary.configured,
        soonest_retirement=summary.soonest_retirement,
        now=now.timestamp(),
    )
    return {"status": "ok", "providers": outcomes}


@celery_app.task(bind=True, name="llm.catalog_refresh")
def catalog_refresh(self) -> Dict[str, Any]:
    """Read the providers that are due and write the catalog gauges."""
    settings = get_settings()
    if not settings.llm_model_catalog_enabled:
        return {"status": "disabled"}

    redis_client = get_sync_redis()
    if not redis_client.set(
        CATALOG_REFRESH_LOCK_KEY, "1", nx=True, ex=CATALOG_REFRESH_LOCK_TIMEOUT
    ):
        return {"status": "skipped", "reason": "lock"}
    try:
        return run_in_worker_loop(
            _refresh_and_publish(
                timedelta(seconds=settings.llm_model_catalog_refresh_interval_seconds),
                float(settings.llm_model_catalog_http_timeout_seconds),
            )
        )
    finally:
        redis_client.delete(CATALOG_REFRESH_LOCK_KEY)

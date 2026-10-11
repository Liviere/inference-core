"""The catalog in numbers: what the gauges show.

Computed from the stored catalog and the config, so it comes out the same
whoever read the providers last: the scheduled task or a command line.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func, select

from inference_core.database.sql.connection import get_async_session
from inference_core.database.sql.models.model_catalog import LLMCatalogEvent
from inference_core.llm.config import LLMConfig

from .drift import compute_drift
from .reconcile import as_utc
from .registry import listed_providers
from .service import SessionFactory, catalog_models, plan_providers, provider_states

RECENT_EVENTS_WINDOW = timedelta(hours=24)
STALE_AFTER_INTERVALS = 3


@dataclass
class CatalogSummary:
    """Numbers per provider; a provider without an entry has nothing to show."""

    providers: List[str] = field(default_factory=list)
    last_success: Dict[str, float] = field(default_factory=dict)
    stale: Dict[str, bool] = field(default_factory=dict)
    model_counts: Dict[str, int] = field(default_factory=dict)
    recent_events: Dict[Tuple[str, str], int] = field(default_factory=dict)
    configured: Dict[Tuple[str, str], int] = field(default_factory=dict)
    soonest_retirement: Dict[str, float] = field(default_factory=dict)


async def summarize(
    config: LLMConfig,
    *,
    interval: timedelta,
    now: Optional[datetime] = None,
    session_factory: SessionFactory = get_async_session,
) -> CatalogSummary:
    """Summarize the stored catalog against ``config``.

    ``providers`` is every provider with a lister, covered or not, so that the
    set of series does not depend on what is configured today. A provider is
    stale when it is covered, was tried, and has had no successful reading
    for three intervals, or none at all.
    """
    now = now or datetime.now(timezone.utc)
    summary = CatalogSummary(providers=listed_providers())

    states = await provider_states(session_factory=session_factory)
    by_provider = {state.provider: state for state in states}
    covered = {plan.provider for plan in plan_providers(config) if not plan.skip_reason}

    for provider, state in by_provider.items():
        summary.model_counts[provider] = state.model_count or 0
        last_success = as_utc(state.last_success_at)
        if last_success is not None:
            summary.last_success[provider] = last_success.timestamp()
    for provider in covered:
        state = by_provider.get(provider)
        if state is None or state.last_attempt_at is None:
            continue
        last_success = as_utc(state.last_success_at)
        # A provider that was tried and never read counts as stale at once:
        # how long it has been failing is not known.
        summary.stale[provider] = (
            last_success is None
            or now - last_success > STALE_AFTER_INTERVALS * interval
        )

    models = await catalog_models(session_factory=session_factory)
    for entry in compute_drift(config, models, states):
        key = (entry.provider, entry.state)
        summary.configured[key] = summary.configured.get(key, 0) + 1
        if entry.retires_at is not None:
            retires = entry.retires_at.timestamp()
            current = summary.soonest_retirement.get(entry.provider)
            if current is None or retires < current:
                summary.soonest_retirement[entry.provider] = retires

    async with session_factory() as session:
        rows = await session.execute(
            select(
                LLMCatalogEvent.provider,
                LLMCatalogEvent.event_type,
                func.count(),
            )
            .where(LLMCatalogEvent.detected_at >= now - RECENT_EVENTS_WINDOW)
            .group_by(LLMCatalogEvent.provider, LLMCatalogEvent.event_type)
        )
        for provider, event_type, count in rows:
            summary.recent_events[(provider, event_type)] = count

    return summary

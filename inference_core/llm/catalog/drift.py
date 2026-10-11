"""Configured models the catalog has something to say about.

Drift is about the models in the LLM config and nothing else: one the
provider does not know (any more), one it has deprecated, one it has set a
retirement date for. Models a provider offers that nobody configured are not
drift, they are the catalog's ``added`` events.

Everything here reads the catalog as stored. Nothing asks a provider: an
answer must not depend on the network being there.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from inference_core.database.sql.connection import get_async_session
from inference_core.database.sql.models.model_catalog import (
    LLMCatalogModel,
    LLMCatalogProviderState,
)
from inference_core.llm.config import LLMConfig

from .reconcile import as_utc
from .service import SessionFactory, catalog_models, plan_providers, provider_states
from .types import ModelLifecycle

STATE_MISSING = "missing"
STATE_DEPRECATED = "deprecated"
STATE_RETIRING = "retiring"
DRIFT_STATES = (STATE_MISSING, STATE_DEPRECATED, STATE_RETIRING)


@dataclass(frozen=True)
class DriftEntry:
    """One configured model that needs a look, and why.

    ``missing``: the provider does not list it and does not know it by id.
    ``retiring``: the provider has set the date it stops answering, or has
    retired it already. ``deprecated``: deprecated, with no date set.
    """

    provider: str
    model: str
    state: str
    deprecated_at: Optional[datetime] = None
    retires_at: Optional[datetime] = None
    replacement: Optional[str] = None


def compute_drift(
    config: LLMConfig,
    models: Sequence[LLMCatalogModel],
    states: Sequence[LLMCatalogProviderState],
) -> List[DriftEntry]:
    """Drift of ``config`` against the catalog's ``models``.

    A provider counts only once it has been read: before its first reading
    nothing is known, and "not in the catalog" would mean nothing.
    """
    read = {state.provider for state in states if state.baselined_at is not None}
    by_provider: Dict[str, List[LLMCatalogModel]] = {}
    for row in models:
        by_provider.setdefault(row.provider, []).append(row)

    entries: List[DriftEntry] = []
    for plan in plan_providers(config):
        if plan.skip_reason or plan.provider not in read:
            continue
        known: Dict[str, LLMCatalogModel] = {}
        for row in by_provider.get(plan.provider, []):
            if row.removed_at is not None:
                continue
            for alias in row.aliases or []:
                known.setdefault(alias, row)
        for row in by_provider.get(plan.provider, []):
            if row.removed_at is None:
                known[row.model_id] = row

        for model in plan.configured:
            row = known.get(model)
            if row is None:
                entries.append(DriftEntry(plan.provider, model, STATE_MISSING))
                continue
            state = _state(row)
            if state is not None:
                entries.append(
                    DriftEntry(
                        provider=plan.provider,
                        model=model,
                        state=state,
                        deprecated_at=as_utc(row.deprecated_at),
                        retires_at=as_utc(row.retires_at),
                        replacement=row.replacement,
                    )
                )
    return entries


def _state(row: LLMCatalogModel) -> Optional[str]:
    if row.retires_at is not None or row.lifecycle == ModelLifecycle.RETIRED.value:
        return STATE_RETIRING
    if row.lifecycle == ModelLifecycle.DEPRECATED.value:
        return STATE_DEPRECATED
    return None


async def current_drift(
    config: LLMConfig, *, session_factory: SessionFactory = get_async_session
) -> List[DriftEntry]:
    """Drift of ``config`` against the catalog as it is stored now."""
    return compute_drift(
        config,
        await catalog_models(session_factory=session_factory),
        await provider_states(session_factory=session_factory),
    )

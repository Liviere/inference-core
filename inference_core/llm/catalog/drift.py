"""Configured models the catalog has something to say about.

Drift is about the models in the LLM config and nothing else: one the
provider does not know (any more), one it has deprecated, one it has set a
retirement date for. Models a provider offers that nobody configured are not
drift, they are the catalog's ``added`` events.

The same reading of the catalog gives the list of everything configured
(``configured_models``): each model of the config with what the catalog says
about it, the current ones included.

Everything here reads the catalog as stored. Nothing asks a provider: an
answer must not depend on the network being there.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set

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
# A configured model the catalog has nothing to say against.
STATE_CURRENT = "ok"


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


@dataclass(frozen=True)
class ConfiguredModel:
    """One model of the config, as a list of them shows it.

    ``agents`` is how many agents of the config name the model: as their
    primary, a fallback, the model that reads images for them, or the model
    a tool switches them to. ``catalog_state`` is ``ok`` or one of the drift
    states, and ``None`` where the catalog cannot say: the provider has no
    lister, is skipped, or was not read yet.
    """

    provider: str
    model: str
    display_name: Optional[str] = None
    agents: int = 0
    catalog_state: Optional[str] = None


def compute_drift(
    config: LLMConfig,
    models: Sequence[LLMCatalogModel],
    states: Sequence[LLMCatalogProviderState],
) -> List[DriftEntry]:
    """Drift of ``config`` against the catalog's ``models``.

    A provider counts only once it has been read: before its first reading
    nothing is known, and "not in the catalog" would mean nothing.
    """
    answerable = _answerable(config, states)
    by_provider = _by_provider(models)

    entries: List[DriftEntry] = []
    for plan in plan_providers(config):
        if plan.provider not in answerable:
            continue
        known = _known(by_provider.get(plan.provider, []))

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


def configured_models(
    config: LLMConfig,
    models: Sequence[LLMCatalogModel] = (),
    states: Sequence[LLMCatalogProviderState] = (),
) -> List[ConfiguredModel]:
    """Every model of ``config``: providers by name, models in the config's order.

    Unlike drift this covers every provider, also one the catalog does not
    read. With no ``models`` and ``states`` it is the config alone, and no
    model has a catalog state.
    """
    answerable = _answerable(config, states)
    by_provider = _by_provider(models)
    used = _agents_by_model(config)

    configured: Dict[str, List[str]] = {}
    for name, model in config.models.items():
        configured.setdefault(str(model.provider), []).append(name)

    entries: List[ConfiguredModel] = []
    for provider in sorted(configured):
        known = _known(by_provider.get(provider, []))
        for name in configured[provider]:
            catalog_state = None
            if provider in answerable:
                row = known.get(name)
                catalog_state = (
                    STATE_MISSING if row is None else _state(row) or STATE_CURRENT
                )
            entries.append(
                ConfiguredModel(
                    provider=provider,
                    model=name,
                    display_name=_display_name(config.models[name]),
                    agents=len(used.get(name, ())),
                    catalog_state=catalog_state,
                )
            )
    return entries


def _answerable(
    config: LLMConfig, states: Sequence[LLMCatalogProviderState]
) -> Set[str]:
    """The providers the catalog can say something about: covered and read."""
    read = {state.provider for state in states if state.baselined_at is not None}
    return {
        plan.provider
        for plan in plan_providers(config)
        if not plan.skip_reason and plan.provider in read
    }


def _by_provider(models: Sequence[LLMCatalogModel]) -> Dict[str, List[LLMCatalogModel]]:
    by_provider: Dict[str, List[LLMCatalogModel]] = {}
    for row in models:
        by_provider.setdefault(row.provider, []).append(row)
    return by_provider


def _known(rows: Sequence[LLMCatalogModel]) -> Dict[str, LLMCatalogModel]:
    """A provider's models by id and by alias; an id wins over an alias."""
    known: Dict[str, LLMCatalogModel] = {}
    for row in rows:
        if row.removed_at is not None:
            continue
        for alias in row.aliases or []:
            known.setdefault(alias, row)
    for row in rows:
        if row.removed_at is None:
            known[row.model_id] = row
    return known


def _agents_by_model(config: LLMConfig) -> Dict[str, Set[str]]:
    """Model -> the agents of the config that name it, in whichever role."""
    used: Dict[str, Set[str]] = {}

    def name(model: Optional[str], agent: str) -> None:
        if model:
            used.setdefault(model, set()).add(agent)

    # The primary as it is in force: an environment variable may replace it.
    for agent, primary in (getattr(config, "agent_models", None) or {}).items():
        name(primary, agent)
    for agent, entry in (getattr(config, "agent_configs", None) or {}).items():
        for fallback in entry.fallback or []:
            name(fallback, agent)
        name(entry.multimodal_support_model, agent)
        for override in entry.tool_model_overrides or []:
            name(override.model, agent)
    return used


def _display_name(model: object) -> Optional[str]:
    # The schema does not declare the field: an entry carries it as an extra.
    label = getattr(model, "display_name", None)
    if isinstance(label, str) and label.strip():
        return label.strip()
    return None


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

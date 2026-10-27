"""Reading providers' model listings into the catalog.

A provider is part of the catalog when the config has a model of it, it has a
lister, and it can be asked: it has its key, or needs none. ``refresh`` reads
those providers, one listing each plus one request for every configured model
the listing does not show (an alias, a model of another account, a model that
was retired), and stores what :mod:`reconcile` makes of it.

Each model is read where it is called: a provider at its ``base_url``, and a
model entry with a ``base_url`` of its own (a regional endpoint, say) at that
one, since another endpoint may not serve the same models.

A reading that fails changes nothing but the provider's state: the models
stay as they were, and the failure is there to be seen.

Nothing here touches metrics: this runs from a command line as well.
"""

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import httpx
from sqlalchemy import select

from inference_core.database.sql.connection import get_async_session
from inference_core.database.sql.models.model_catalog import (
    LLMCatalogEvent,
    LLMCatalogModel,
    LLMCatalogProviderState,
)
from inference_core.llm.config import LLMConfig, provider_model_id

from .http import DEFAULT_READ_TIMEOUT_SECONDS, CatalogError, build_client
from .reconcile import as_utc, reconcile
from .registry import ModelLister, get_model_lister
from .types import DiscoveredModel

logger = logging.getLogger(__name__)

DEFAULT_REFRESH_INTERVAL = timedelta(days=1)
RETRY_AFTER_FAILURE = timedelta(hours=1)
PROVIDER_TIMEOUT_SECONDS = 90.0

OUTCOME_REFRESHED = "refreshed"
OUTCOME_FAILED = "failed"
OUTCOME_SKIPPED = "skipped"
OUTCOME_NOT_DUE = "not_due"

SessionFactory = Callable[[], Any]


@dataclass(frozen=True)
class ProviderPlan:
    """A provider with configured models and a lister, and whether it is read."""

    provider: str
    configured: Tuple[str, ...]
    skip_reason: Optional[str] = None
    # Configured models with a base_url of their own, and that base_url.
    own_endpoints: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ProviderRefresh:
    """What became of one provider in a refresh."""

    provider: str
    outcome: str
    detail: Optional[str] = None
    model_count: int = 0
    events: int = 0


@dataclass
class _Reading:
    listed: List[DiscoveredModel]
    looked_up: List[DiscoveredModel]
    not_found: Set[str]


class _EmptyListing(CatalogError):
    """A provider that had models listed none."""


# Providers whose lister has no address of its own to fall back on.
_NO_DEFAULT_ENDPOINT = frozenset({"custom_openai_compatible", "cortecs"})


def plan_providers(config: LLMConfig) -> List[ProviderPlan]:
    """The providers the catalog covers, sorted by name.

    A provider without a lister or without a configured model is not in the
    catalog at all. One that is switched off or cannot be asked is, with the
    reason.
    """
    # A model is looked for under the id its provider knows it by.
    configured: Dict[str, List[str]] = {}
    own: Dict[str, List[Tuple[str, Any]]] = {}
    for model in config.models.values():
        model_id = provider_model_id(model)
        provider = str(model.provider)
        if model_id not in configured.setdefault(provider, []):
            configured[provider].append(model_id)
        own.setdefault(provider, []).append((model_id, model.base_url))

    plans = []
    for provider in sorted(configured):
        if get_model_lister(provider) is None:
            continue
        plans.append(
            ProviderPlan(
                provider=provider,
                configured=tuple(configured[provider]),
                skip_reason=_skip_reason(config, provider),
                own_endpoints=_own_endpoints(config, provider, own[provider]),
            )
        )
    return plans


def _endpoint(url: Any) -> Optional[str]:
    if not isinstance(url, str) or not url.strip():
        return None
    return url.strip().rstrip("/")


def _own_endpoints(
    config: LLMConfig, provider: str, models: Sequence[Tuple[str, Any]]
) -> Tuple[Tuple[str, str], ...]:
    """The models (by provider id) whose ``base_url`` is not their
    provider's, with it."""
    provider_endpoint = _endpoint(
        config.get_provider_runtime_config(provider).get("base_url")
    )
    own = []
    for model_id, base_url in models:
        endpoint = _endpoint(base_url)
        if endpoint and endpoint != provider_endpoint:
            own.append((model_id, endpoint))
    return tuple(own)


def _skip_reason(config: LLMConfig, provider: str) -> Optional[str]:
    if not config.get_provider_catalog_config(provider).enabled:
        return "disabled"
    try:
        provider_config = config.get_provider_config(provider)
    except Exception:
        return "unreadable provider entry"
    runtime = config.get_provider_runtime_config(provider)
    if provider_config.requires_api_key and not runtime.get("api_key"):
        return "no API key"
    if provider in _NO_DEFAULT_ENDPOINT and not runtime.get("base_url"):
        return "no base_url"
    return None


def describe_error(exc: BaseException) -> str:
    """An error as it is stored: what kind, and the status code if any.

    Never the error's own text: for an HTTP client that can be a URL.
    """
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if isinstance(exc, _EmptyListing):
        return "EmptyListing"
    if isinstance(exc, CatalogError) and isinstance(status, int) and status != 200:
        return f"{name} (HTTP {status})"
    if isinstance(exc, CatalogError):
        return f"{name}: {exc}"[:200]
    return name[:200]


def _is_due(
    state: Optional[LLMCatalogProviderState], now: datetime, interval: timedelta
) -> bool:
    if state is None or state.last_attempt_at is None:
        return True
    last_attempt = as_utc(state.last_attempt_at)
    last_success = as_utc(state.last_success_at)
    if last_success is None or last_success < last_attempt:
        # The latest reading failed: try again sooner than a full interval.
        return now - last_attempt >= min(RETRY_AFTER_FAILURE, interval)
    return now - last_success >= interval


def _as_configured(found: DiscoveredModel, model_id: str) -> DiscoveredModel:
    """``found`` under the id the config uses, with what it stands for."""
    if found.model_id == model_id:
        return found
    return replace(
        found,
        model_id=model_id,
        attributes={**found.attributes, "resolves_to": found.model_id},
    )


async def _read_provider(
    lister: ModelLister,
    runtime: Dict[str, Any],
    configured: Sequence[str],
    client: httpx.AsyncClient,
    own_endpoints: Optional[Dict[str, str]] = None,
) -> _Reading:
    own_endpoints = own_endpoints or {}
    # What the provider's endpoint says of a model that is called elsewhere
    # does not count for it.
    listed = [
        model
        for model in await lister.list_models(runtime, client)
        if model.model_id not in own_endpoints
    ]
    known = {model.model_id for model in listed}
    for model in listed:
        known.update(model.aliases)

    looked_up: List[DiscoveredModel] = []
    not_found: Set[str] = set()
    if lister.get_model is not None:
        for model_id in configured:
            if model_id in known or model_id in own_endpoints:
                continue
            found = await lister.get_model(runtime, client, model_id)
            if found is None:
                not_found.add(model_id)
                continue
            looked_up.append(_as_configured(found, model_id))

    by_endpoint: Dict[str, List[str]] = {}
    for model_id, endpoint in own_endpoints.items():
        by_endpoint.setdefault(endpoint, []).append(model_id)
    for endpoint, model_ids in sorted(by_endpoint.items()):
        reading = await _read_provider(
            lister, {**runtime, "base_url": endpoint}, model_ids, client
        )
        looked_up.extend(_configured_of(reading, model_ids))
        not_found.update(reading.not_found)
    return _Reading(listed=listed, looked_up=looked_up, not_found=not_found)


def _configured_of(
    reading: _Reading, model_ids: Sequence[str]
) -> List[DiscoveredModel]:
    """The models ``model_ids`` name in a reading of their own endpoint.

    Only those: the rest of that endpoint's listing is not the provider's.
    """
    found: List[DiscoveredModel] = list(reading.looked_up)
    wanted = set(model_ids) - {model.model_id for model in found}
    for model in reading.listed:
        for model_id in sorted(wanted & ({model.model_id} | set(model.aliases))):
            found.append(_as_configured(model, model_id))
            wanted.discard(model_id)
    return found


async def _store_reading(
    session_factory: SessionFactory,
    provider: str,
    reading: _Reading,
    now: datetime,
    removal_after: timedelta,
) -> ProviderRefresh:
    async with session_factory() as session:
        state = await session.get(LLMCatalogProviderState, provider)
        if state is None:
            state = LLMCatalogProviderState(provider=provider, model_count=0)
            session.add(state)
        rows = (
            (
                await session.execute(
                    select(LLMCatalogModel).where(LLMCatalogModel.provider == provider)
                )
            )
            .scalars()
            .all()
        )
        changes = reconcile(
            provider,
            rows,
            reading.listed,
            reading.looked_up,
            not_found=reading.not_found,
            baseline=state.baselined_at is None,
            now=now,
            removal_after=removal_after,
        )
        session.add_all(changes.new_rows)
        session.add_all(changes.events)
        for row in changes.dropped_rows:
            await session.delete(row)

        state.last_attempt_at = now
        state.last_success_at = now
        state.last_error = None
        state.model_count = len(reading.listed)
        if state.baselined_at is None:
            state.baselined_at = now
        await session.commit()

    return ProviderRefresh(
        provider=provider,
        outcome=OUTCOME_REFRESHED,
        model_count=len(reading.listed),
        events=len(changes.events),
    )


async def _store_failure(
    session_factory: SessionFactory, provider: str, error: str, now: datetime
) -> None:
    async with session_factory() as session:
        state = await session.get(LLMCatalogProviderState, provider)
        if state is None:
            state = LLMCatalogProviderState(provider=provider, model_count=0)
            session.add(state)
        state.last_attempt_at = now
        state.last_error = error
        await session.commit()


async def refresh(
    config: LLMConfig,
    *,
    providers: Optional[Sequence[str]] = None,
    only_due: bool = False,
    interval: timedelta = DEFAULT_REFRESH_INTERVAL,
    read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
    now: Optional[datetime] = None,
    client: Optional[httpx.AsyncClient] = None,
    session_factory: SessionFactory = get_async_session,
) -> List[ProviderRefresh]:
    """Read providers' listings into the catalog.

    All covered providers, or those named in ``providers``. With ``only_due``
    a provider read less than ``interval`` ago is left alone (one whose latest
    reading failed is tried again after an hour). ``interval`` is also how far
    apart two readings must be before a model missing from both is taken for
    removed.

    One provider's failure never stops the others. Returns what became of
    each provider, in name order.
    """
    now = now or datetime.now(timezone.utc)
    plans = plan_providers(config)
    if providers is not None:
        wanted = set(providers)
        covered = {plan.provider for plan in plans}
        plans = [plan for plan in plans if plan.provider in wanted]
        plans += [
            ProviderPlan(name, (), "not in the catalog")
            for name in sorted(wanted - covered)
        ]

    results: Dict[str, ProviderRefresh] = {}
    to_read: List[ProviderPlan] = []

    async with session_factory() as session:
        states = {
            state.provider: state
            for state in (
                await session.execute(select(LLMCatalogProviderState))
            ).scalars()
        }
    for plan in plans:
        if plan.skip_reason:
            results[plan.provider] = ProviderRefresh(
                plan.provider, OUTCOME_SKIPPED, plan.skip_reason
            )
        elif only_due and not _is_due(states.get(plan.provider), now, interval):
            results[plan.provider] = ProviderRefresh(plan.provider, OUTCOME_NOT_DUE)
        else:
            to_read.append(plan)

    own_client = client is None
    http = client or build_client(read_timeout)
    try:
        readings = await asyncio.gather(
            *(
                asyncio.wait_for(
                    _read_provider(
                        get_model_lister(plan.provider),
                        config.get_provider_runtime_config(plan.provider),
                        plan.configured,
                        http,
                        dict(plan.own_endpoints),
                    ),
                    PROVIDER_TIMEOUT_SECONDS,
                )
                for plan in to_read
            ),
            return_exceptions=True,
        )
    finally:
        if own_client:
            await http.aclose()

    for plan, reading in zip(to_read, readings):
        try:
            if isinstance(reading, BaseException):
                raise reading
            previous = states.get(plan.provider)
            if not reading.listed and previous is not None and previous.model_count:
                raise _EmptyListing("empty listing")
            results[plan.provider] = await _store_reading(
                session_factory, plan.provider, reading, now, interval
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = describe_error(exc)
            logger.warning("Model catalog: reading %s failed: %s", plan.provider, error)
            try:
                await _store_failure(session_factory, plan.provider, error, now)
            except Exception:
                logger.exception(
                    "Model catalog: could not record the failure for %s", plan.provider
                )
            results[plan.provider] = ProviderRefresh(
                plan.provider, OUTCOME_FAILED, error
            )

    return [results[name] for name in sorted(results)]


async def provider_states(
    *, session_factory: SessionFactory = get_async_session
) -> List[LLMCatalogProviderState]:
    """Every provider's reading state, by provider name."""
    async with session_factory() as session:
        result = await session.execute(
            select(LLMCatalogProviderState).order_by(LLMCatalogProviderState.provider)
        )
        return list(result.scalars())


async def catalog_models(
    *,
    provider: Optional[str] = None,
    session_factory: SessionFactory = get_async_session,
) -> List[LLMCatalogModel]:
    """The catalog's models, removed ones included, by provider and id."""
    statement = select(LLMCatalogModel).order_by(
        LLMCatalogModel.provider, LLMCatalogModel.model_id
    )
    if provider is not None:
        statement = statement.where(LLMCatalogModel.provider == provider)
    async with session_factory() as session:
        return list((await session.execute(statement)).scalars())


async def catalog_events(
    *,
    since: Optional[datetime] = None,
    provider: Optional[str] = None,
    limit: int = 500,
    session_factory: SessionFactory = get_async_session,
) -> List[LLMCatalogEvent]:
    """What changed, newest first."""
    statement = select(LLMCatalogEvent).order_by(
        LLMCatalogEvent.detected_at.desc(), LLMCatalogEvent.model_id
    )
    if since is not None:
        statement = statement.where(LLMCatalogEvent.detected_at >= since)
    if provider is not None:
        statement = statement.where(LLMCatalogEvent.provider == provider)
    async with session_factory() as session:
        return list((await session.execute(statement.limit(limit))).scalars())

"""The rules that turn one reading of a provider's models into rows and events.

No I/O here: the rows are those the provider already has in the catalog, and
what comes back is what to add, what to drop and what to report. The rules
are about not crying wolf:

- The first reading is a baseline. Nothing in it is new.
- A model that is missing from one listing is not gone. It is taken for
  removed after two readings without it that are far enough apart, so two
  readings in a row prove nothing.
- When the provider was asked about a model by id and said there is none, the
  model is gone: that is an answer, not an absence.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Collection, Dict, List, Optional, Sequence, Tuple

from inference_core.database.sql.models.model_catalog import (
    LLMCatalogEvent,
    LLMCatalogModel,
)

from .types import DiscoveredModel

EVENT_ADDED = "added"
EVENT_REMOVED = "removed"
EVENT_REAPPEARED = "reappeared"
EVENT_LIFECYCLE_CHANGED = "lifecycle_changed"
EVENT_TYPES = (
    EVENT_ADDED,
    EVENT_REMOVED,
    EVENT_REAPPEARED,
    EVENT_LIFECYCLE_CHANGED,
)

SOURCE_LIST = "list"
SOURCE_LOOKUP = "lookup"

MISSES_BEFORE_REMOVAL = 2


@dataclass
class Reconciliation:
    """What one reading changes."""

    new_rows: List[LLMCatalogModel] = field(default_factory=list)
    dropped_rows: List[LLMCatalogModel] = field(default_factory=list)
    events: List[LLMCatalogEvent] = field(default_factory=list)


def as_utc(value: Optional[datetime]) -> Optional[datetime]:
    """``value`` as an aware UTC datetime; a naive one is taken to be UTC.

    Some databases hand back naive datetimes for timezone-aware columns.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _lifecycle_view(row: LLMCatalogModel) -> Dict[str, Any]:
    deprecated_at = as_utc(row.deprecated_at)
    retires_at = as_utc(row.retires_at)
    return {
        "lifecycle": row.lifecycle,
        "deprecated_at": deprecated_at.isoformat() if deprecated_at else None,
        "retires_at": retires_at.isoformat() if retires_at else None,
        "replacement": row.replacement,
    }


def _apply(
    row: LLMCatalogModel, model: DiscoveredModel, source: str, now: datetime
) -> None:
    row.source = source
    row.kind = model.kind.value
    row.lifecycle = model.lifecycle.value
    row.deprecated_at = model.deprecated_at
    row.retires_at = model.retires_at
    row.replacement = model.replacement
    row.aliases = list(model.aliases)
    row.attributes = dict(model.attributes)
    row.last_seen_at = now
    row.missed_runs = 0


def reconcile(
    provider: str,
    rows: Sequence[LLMCatalogModel],
    listed: Sequence[DiscoveredModel],
    looked_up: Sequence[DiscoveredModel] = (),
    *,
    not_found: Collection[str] = (),
    baseline: bool,
    now: datetime,
    removal_after: timedelta,
) -> Reconciliation:
    """Bring ``rows`` in line with one reading and say what changed.

    ``listed`` is the provider's listing, ``looked_up`` the models found by
    asking for them by id, ``not_found`` the ids the provider said do not
    exist. Existing rows are updated in place.
    """
    result = Reconciliation()

    def event(model_id: str, event_type: str, details: Dict[str, Any]) -> None:
        result.events.append(
            LLMCatalogEvent(
                provider=provider,
                model_id=model_id,
                event_type=event_type,
                details=details,
                detected_at=now,
            )
        )

    seen: Dict[str, Tuple[DiscoveredModel, str]] = {}
    for model in listed:
        seen.setdefault(model.model_id, (model, SOURCE_LIST))
    for model in looked_up:
        seen.setdefault(model.model_id, (model, SOURCE_LOOKUP))

    by_id = {row.model_id: row for row in rows}

    for model_id, (model, source) in seen.items():
        row = by_id.get(model_id)
        if row is None:
            row = LLMCatalogModel(
                provider=provider, model_id=model_id, first_seen_at=now
            )
            _apply(row, model, source, now)
            result.new_rows.append(row)
            # A model that was asked about by id is new to the config, not to
            # the provider.
            if not baseline and source == SOURCE_LIST:
                event(
                    model_id,
                    EVENT_ADDED,
                    {"kind": row.kind, "lifecycle": row.lifecycle},
                )
            continue

        if row.removed_at is not None:
            event(
                model_id,
                EVENT_REAPPEARED,
                {"removed_at": as_utc(row.removed_at).isoformat()},
            )
            row.removed_at = None
        before = _lifecycle_view(row)
        _apply(row, model, source, now)
        after = _lifecycle_view(row)
        if before != after:
            event(model_id, EVENT_LIFECYCLE_CHANGED, {"before": before, "after": after})

    for row in rows:
        if row.model_id in seen or row.removed_at is not None:
            continue
        last_seen = as_utc(row.last_seen_at)
        if row.model_id in not_found:
            row.removed_at = now
            event(
                row.model_id,
                EVENT_REMOVED,
                {"last_seen_at": last_seen.isoformat(), "reason": "not_found"},
            )
        elif row.source == SOURCE_LOOKUP:
            # Nobody asks about it any more, so nothing is known about it.
            result.dropped_rows.append(row)
        else:
            row.missed_runs = (row.missed_runs or 0) + 1
            if (
                row.missed_runs >= MISSES_BEFORE_REMOVAL
                and now - last_seen >= removal_after
            ):
                row.removed_at = now
                event(
                    row.model_id,
                    EVENT_REMOVED,
                    {"last_seen_at": last_seen.isoformat(), "reason": "not_listed"},
                )

    return result

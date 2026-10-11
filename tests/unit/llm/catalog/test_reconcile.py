"""
Unit tests for the rules that turn a reading of a listing into rows and events.

Covers:
- the first reading is a baseline: rows, no events
- a new model, a model that comes back, a lifecycle that changes
- a model missing from the listing is removed only after two readings far
  enough apart
- a model the provider says does not exist is removed at once
- a model that was looked up and is no longer asked about is dropped quietly
"""

from datetime import datetime, timedelta, timezone

from inference_core.database.sql.models.model_catalog import LLMCatalogModel
from inference_core.llm.catalog.reconcile import (
    EVENT_ADDED,
    EVENT_LIFECYCLE_CHANGED,
    EVENT_REAPPEARED,
    EVENT_REMOVED,
    reconcile,
)
from inference_core.llm.catalog.types import (
    DiscoveredModel,
    ModelKind,
    ModelLifecycle,
)

NOW = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)
DAY = timedelta(days=1)


def _model(model_id: str, **fields) -> DiscoveredModel:
    fields.setdefault("kind", ModelKind.CHAT)
    fields.setdefault("lifecycle", ModelLifecycle.ACTIVE)
    return DiscoveredModel(provider="acme", model_id=model_id, **fields)


def _row(model_id: str, **fields) -> LLMCatalogModel:
    fields.setdefault("source", "list")
    fields.setdefault("kind", "chat")
    fields.setdefault("lifecycle", "active")
    fields.setdefault("first_seen_at", NOW - 10 * DAY)
    fields.setdefault("last_seen_at", NOW - DAY)
    fields.setdefault("missed_runs", 0)
    return LLMCatalogModel(provider="acme", model_id=model_id, **fields)


def _run(rows, listed, looked_up=(), **options):
    options.setdefault("baseline", False)
    options.setdefault("now", NOW)
    options.setdefault("removal_after", DAY)
    return reconcile("acme", rows, listed, looked_up, **options)


def _events(result):
    return [(event.model_id, event.event_type) for event in result.events]


def test_first_reading_is_a_baseline():
    result = _run([], [_model("a"), _model("b")], baseline=True)

    assert [row.model_id for row in result.new_rows] == ["a", "b"]
    assert result.events == []
    row = result.new_rows[0]
    assert (row.first_seen_at, row.last_seen_at, row.missed_runs) == (NOW, NOW, 0)
    assert (row.source, row.kind, row.lifecycle) == ("list", "chat", "active")


def test_new_model_is_reported():
    result = _run([_row("a")], [_model("a"), _model("b", aliases=("b-latest",))])

    assert [row.model_id for row in result.new_rows] == ["b"]
    assert result.new_rows[0].aliases == ["b-latest"]
    assert _events(result) == [("b", EVENT_ADDED)]
    assert result.events[0].details == {"kind": "chat", "lifecycle": "active"}
    assert result.events[0].detected_at == NOW


def test_the_same_model_twice_in_a_listing_is_one_model():
    result = _run([], [_model("a"), _model("a")])

    assert [row.model_id for row in result.new_rows] == ["a"]
    assert _events(result) == [("a", EVENT_ADDED)]


def test_seen_again_resets_the_misses():
    row = _row("a", missed_runs=1)

    result = _run([row], [_model("a", attributes={"context_window": 8})])

    assert result.events == []
    assert (row.missed_runs, row.last_seen_at) == (0, NOW)
    assert row.attributes == {"context_window": 8}


def test_one_miss_is_not_a_removal():
    row = _row("a", last_seen_at=NOW - 3 * DAY)

    result = _run([row], [_model("b")])

    assert row.missed_runs == 1
    assert row.removed_at is None
    assert _events(result) == [("b", EVENT_ADDED)]


def test_two_misses_far_enough_apart_are_a_removal():
    row = _row("a", missed_runs=1, last_seen_at=NOW - 2 * DAY)

    result = _run([row], [_model("b")])

    assert row.removed_at == NOW
    assert ("a", EVENT_REMOVED) in _events(result)
    removed = next(e for e in result.events if e.event_type == EVENT_REMOVED)
    assert removed.details == {
        "last_seen_at": (NOW - 2 * DAY).isoformat(),
        "reason": "not_listed",
    }


def test_two_misses_in_quick_succession_are_not_a_removal():
    row = _row("a", missed_runs=1, last_seen_at=NOW - timedelta(minutes=5))

    result = _run([row], [])

    assert row.missed_runs == 2
    assert row.removed_at is None
    assert result.events == []


def test_removed_model_is_not_reported_again():
    row = _row("a", missed_runs=5, removed_at=NOW - DAY)

    result = _run([row], [])

    assert result.events == []
    assert row.missed_runs == 5


def test_model_that_comes_back_is_reported():
    row = _row("a", missed_runs=2, removed_at=NOW - DAY)

    result = _run([row], [_model("a")])

    assert _events(result) == [("a", EVENT_REAPPEARED)]
    assert row.removed_at is None
    assert row.missed_runs == 0


def test_lifecycle_change_is_reported_with_before_and_after():
    row = _row("a")
    retires = datetime(2026, 12, 1, tzinfo=timezone.utc)

    result = _run(
        [row],
        [
            _model(
                "a",
                lifecycle=ModelLifecycle.DEPRECATED,
                retires_at=retires,
                replacement="a2",
            )
        ],
    )

    assert _events(result) == [("a", EVENT_LIFECYCLE_CHANGED)]
    assert result.events[0].details == {
        "before": {
            "lifecycle": "active",
            "deprecated_at": None,
            "retires_at": None,
            "replacement": None,
        },
        "after": {
            "lifecycle": "deprecated",
            "deprecated_at": None,
            "retires_at": retires.isoformat(),
            "replacement": "a2",
        },
    }
    assert row.retires_at == retires


def test_a_date_read_back_without_a_zone_is_the_same_date():
    retires = datetime(2026, 12, 1, tzinfo=timezone.utc)
    row = _row("a", lifecycle="deprecated", retires_at=retires.replace(tzinfo=None))

    result = _run(
        [row], [_model("a", lifecycle=ModelLifecycle.DEPRECATED, retires_at=retires)]
    )

    assert result.events == []


def test_other_attributes_changing_is_not_an_event():
    row = _row("a", attributes={"pricing": {"input": 1.0}})

    result = _run([row], [_model("a", attributes={"pricing": {"input": 2.0}})])

    assert result.events == []
    assert row.attributes == {"pricing": {"input": 2.0}}


def test_looked_up_model_is_stored_without_an_event():
    result = _run([], [_model("a")], [_model("alias")])

    assert [(row.model_id, row.source) for row in result.new_rows] == [
        ("a", "list"),
        ("alias", "lookup"),
    ]
    assert _events(result) == [("a", EVENT_ADDED)]


def test_listed_wins_over_looked_up():
    result = _run([], [_model("a")], [_model("a")], baseline=True)

    assert [(row.model_id, row.source) for row in result.new_rows] == [("a", "list")]


def test_model_the_provider_says_is_gone_is_removed_at_once():
    row = _row("a", last_seen_at=NOW - timedelta(minutes=5))

    result = _run([row], [_model("b")], not_found={"a"})

    assert row.removed_at == NOW
    removed = next(e for e in result.events if e.event_type == EVENT_REMOVED)
    assert removed.model_id == "a"
    assert removed.details["reason"] == "not_found"


def test_looked_up_model_nobody_asks_about_is_dropped_quietly():
    row = _row("alias", source="lookup")

    result = _run([row], [_model("a")], baseline=True)

    assert result.dropped_rows == [row]
    assert result.events == []

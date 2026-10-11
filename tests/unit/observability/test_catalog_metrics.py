"""
Unit tests for the model catalog gauges.

Covers:
- every provider gets every series, with 0 where there is nothing to report
- a value that was set goes back to 0 when the next publish has none
"""

from prometheus_client import REGISTRY

from inference_core.observability.metrics import set_model_catalog_gauges

PROVIDERS = ["metrics_test_a", "metrics_test_b"]
EVENT_TYPES = ("added", "removed")
STATES = ("missing", "deprecated", "retiring")


def _publish(**values):
    fields = {
        "last_success": {},
        "stale": {},
        "model_counts": {},
        "recent_events": {},
        "configured": {},
        "soonest_retirement": {},
    }
    fields.update(values)
    set_model_catalog_gauges(
        providers=PROVIDERS,
        event_types=EVENT_TYPES,
        drift_states=STATES,
        now=1_790_000_000.0,
        **fields,
    )


def _value(name, **labels):
    return REGISTRY.get_sample_value(name, labels or None)


def test_every_series_is_written():
    _publish(
        last_success={"metrics_test_a": 1_789_000_000.0},
        stale={"metrics_test_b": True},
        model_counts={"metrics_test_a": 42},
        recent_events={("metrics_test_a", "added"): 3},
        configured={("metrics_test_a", "retiring"): 2},
        soonest_retirement={"metrics_test_a": 1_795_000_000.0},
    )

    assert _value("llm_catalog_publish_timestamp") == 1_790_000_000.0
    a, b = PROVIDERS
    assert (
        _value("llm_catalog_provider_last_success_timestamp", provider=a)
        == 1_789_000_000.0
    )
    assert _value("llm_catalog_provider_last_success_timestamp", provider=b) == 0
    assert _value("llm_catalog_provider_stale", provider=a) == 0
    assert _value("llm_catalog_provider_stale", provider=b) == 1
    assert _value("llm_catalog_provider_models", provider=a) == 42
    assert _value("llm_catalog_provider_models", provider=b) == 0
    assert _value("llm_catalog_events_recent", provider=a, type="added") == 3
    assert _value("llm_catalog_events_recent", provider=a, type="removed") == 0
    assert _value("llm_catalog_events_recent", provider=b, type="added") == 0
    assert _value("llm_catalog_configured_models", provider=a, state="retiring") == 2
    for state in ("missing", "deprecated"):
        assert _value("llm_catalog_configured_models", provider=a, state=state) == 0
    for state in STATES:
        assert _value("llm_catalog_configured_models", provider=b, state=state) == 0
    assert (
        _value("llm_catalog_configured_retirement_soonest_timestamp", provider=a)
        == 1_795_000_000.0
    )
    assert (
        _value("llm_catalog_configured_retirement_soonest_timestamp", provider=b) == 0
    )


def test_a_value_goes_back_to_zero():
    a = PROVIDERS[0]
    _publish(configured={(a, "missing"): 1}, stale={a: True})
    assert _value("llm_catalog_configured_models", provider=a, state="missing") == 1

    _publish()

    assert _value("llm_catalog_configured_models", provider=a, state="missing") == 0
    assert _value("llm_catalog_provider_stale", provider=a) == 0

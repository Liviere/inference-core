"""
Unit tests for the draft ``models:`` entry made from a catalog model.

Covers:
- the entry loads as YAML into a valid ModelConfig, pricing included
- what the listing does not say is said in a comment, never guessed
- a deprecated or retiring model carries a warning
"""

from datetime import datetime, timezone

import yaml

from inference_core.database.sql.models.model_catalog import LLMCatalogModel
from inference_core.llm.catalog.scaffold import render_model_entry
from inference_core.llm.config import ModelConfig, PricingConfig

NOW = datetime(2026, 10, 11, tzinfo=timezone.utc)


def _row(model_id="org/model-1", provider="deepinfra", **fields) -> LLMCatalogModel:
    fields.setdefault("lifecycle", "active")
    return LLMCatalogModel(
        provider=provider,
        model_id=model_id,
        source="list",
        kind="chat",
        first_seen_at=NOW,
        last_seen_at=NOW,
        **fields,
    )


def _load(text: str) -> dict:
    return yaml.safe_load(text)


def test_entry_with_everything_the_listing_can_say():
    row = _row(
        attributes={
            "display_name": "Model One",
            "context_window": 262144,
            "max_output_tokens": 8192,
            "capabilities": {"vision": True, "reasoning": True},
            "pricing": {"input": 0.35, "output": 0.4, "cache_read": 0.07},
        }
    )

    text = render_model_entry(row)

    entry = _load(text)["org/model-1"]
    assert entry == {
        "provider": "deepinfra",
        "display_name": "Model One",
        "max_tokens": 8192,
        "multimodal": True,
        "pricing": {
            "currency": "USD",
            "input": {"cost_per_1m": 0.35},
            "output": {"cost_per_1m": 0.4},
            "extras": {"cache_read_tokens": {"cost_per_1m": 0.07}},
        },
    }
    pricing = PricingConfig(**entry["pricing"])
    assert round(pricing.input.cost_per_1k * 1000, 6) == 0.35
    config = ModelConfig(
        name="org/model-1", **{k: v for k, v in entry.items() if k != "pricing"}
    )
    assert config.max_tokens == 8192
    assert "Context window: 262144 tokens." in text
    assert "reasoning_config" in text
    assert "WARNING" not in text
    assert all(line.startswith("  ") for line in text.splitlines())


def test_what_the_listing_does_not_say_is_left_out_and_said():
    text = render_model_entry(_row("gpt-5", "openai", attributes={}))

    assert _load(text) == {"gpt-5": {"provider": "openai"}}
    assert "# max_tokens: not in the listing" in text
    assert "# pricing: not in the listing." in text


def test_half_a_price_is_no_price():
    text = render_model_entry(_row(attributes={"pricing": {"input": 0.2}}))

    assert "pricing" not in _load(text)["org/model-1"]
    assert "# pricing: not in the listing." in text


def test_id_that_needs_quoting_still_loads():
    text = render_model_entry(_row("weird: id #1"))

    assert list(_load(text)) == ["weird: id #1"]


def test_deprecated_model_carries_a_warning():
    row = _row(lifecycle="deprecated", replacement="org/model-2")

    text = render_model_entry(row)

    assert "# WARNING: the provider has deprecated this model; it names" in text
    assert "org/model-2" in text


def test_retiring_model_says_when():
    row = _row(
        lifecycle="deprecated",
        retires_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
    )

    assert "# WARNING: the provider retires this model on 2026-12-01." in (
        render_model_entry(row)
    )


def test_retired_model():
    assert "has retired this model" in render_model_entry(_row(lifecycle="retired"))

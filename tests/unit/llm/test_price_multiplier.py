"""A provider's price_multiplier, read from llm_config.yaml.

The pricing of a model is written at list prices. A provider that charges
more than that (on a regional endpoint, say) has a multiplier, and every cost
computed from the model's pricing includes it. A model entry's own multiplier
wins over its provider's; without either, the prices are as written.
"""

import pytest
import yaml

from inference_core.llm.config import LLMConfig
from inference_core.llm.usage_logging import PricingCalculator

PRICING = {
    "input": {"cost_per_1m": 0.15},
    "output": {"cost_per_1m": 0.6},
    "extras": {"cache_read_tokens": {"cost_per_1m": 0.015}},
}


def _config(tmp_path, monkeypatch, providers, models) -> LLMConfig:
    path = tmp_path / "llm_config.yaml"
    path.write_text(
        yaml.safe_dump({"providers": providers, "models": models}), encoding="utf-8"
    )
    monkeypatch.setenv("LLM_CONFIG_PATH", str(path))
    return LLMConfig()


def _provider(**fields):
    return {"name": "Regional", "requires_api_key": False, **fields}


@pytest.fixture
def config(tmp_path, monkeypatch):
    return _config(
        tmp_path,
        monkeypatch,
        providers={
            "mistral": _provider(price_multiplier=1.1),
            "claude": _provider(),
        },
        models={
            "regional": {"provider": "mistral", "pricing": PRICING},
            "own-multiplier": {
                "provider": "mistral",
                "price_multiplier": 1.25,
                "pricing": PRICING,
            },
            "plain": {"provider": "claude", "pricing": PRICING},
            "plain-with-own": {
                "provider": "claude",
                "price_multiplier": 1.1,
                "pricing": PRICING,
            },
        },
    )


def _per_1m(config, model):
    pricing = config.models[model].pricing
    return (
        round(pricing.input.cost_per_1k * 1000, 9),
        round(pricing.output.cost_per_1k * 1000, 9),
        round(pricing.extras["cache_read_tokens"].cost_per_1k * 1000, 9),
    )


def test_the_providers_multiplier_scales_every_dimension(config):
    assert _per_1m(config, "regional") == (0.165, 0.66, 0.0165)


def test_without_one_the_prices_are_as_written(config):
    assert _per_1m(config, "plain") == (0.15, 0.6, 0.015)


def test_a_models_own_wins(config):
    assert _per_1m(config, "own-multiplier") == (0.1875, 0.75, 0.01875)
    assert _per_1m(config, "plain-with-own") == (0.165, 0.66, 0.0165)


def test_it_is_no_extra_parameter_of_the_model(config):
    assert "price_multiplier" not in (config.models["own-multiplier"].model_extra or {})


def test_the_computed_cost_includes_it(config):
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}

    regional = PricingCalculator.compute_cost(usage, config.models["regional"].pricing)
    plain = PricingCalculator.compute_cost(usage, config.models["plain"].pricing)

    assert plain["cost_total_usd"] == pytest.approx(0.75)
    assert regional["cost_total_usd"] == pytest.approx(0.825)


def test_the_provider_entry_reads_it(config):
    assert config.get_provider_config("mistral").price_multiplier == 1.1
    assert config.get_provider_config("claude").price_multiplier == 1.0


@pytest.mark.parametrize("value", [0, -1, "1.1", True])
def test_a_models_multiplier_that_is_no_positive_number_is_refused(
    tmp_path, monkeypatch, value
):
    with pytest.raises(ValueError, match="price_multiplier"):
        _config(
            tmp_path,
            monkeypatch,
            providers={"claude": _provider()},
            models={
                "m": {
                    "provider": "claude",
                    "price_multiplier": value,
                    "pricing": PRICING,
                }
            },
        )


def test_a_providers_multiplier_that_is_not_positive_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="price_multiplier"):
        _config(
            tmp_path,
            monkeypatch,
            providers={"claude": _provider(price_multiplier=0)},
            models={"m": {"provider": "claude", "pricing": PRICING}},
        )

"""Which tokens choose a model's context tier (``context_tier_basis``).

By default a tier is chosen by input and output together. A provider that
prices a request by the length of its prompt (Anthropic: "A request's prompt
length counts all of its input tokens, including cache reads and cache
writes") is priced with ``context_tier_basis: input``, so a reply near the
threshold is not charged the higher rate for its output.
"""

import pytest
import yaml

from inference_core.llm.config import (
    ContextTier,
    DimensionPrice,
    LLMConfig,
    PricingConfig,
)
from inference_core.llm.usage_logging import PricingCalculator

# Prices of a model that costs five times as much over 100K tokens.
TIERED = {
    "input": {"cost_per_1m": 0.1},
    "output": {"cost_per_1m": 0.5},
    "extras": {
        "cache_read_tokens": {"cost_per_1m": 0.01},
        "cache_write_tokens": {"cost_per_1m": 0.125},
    },
    "context_tiers": [
        {"max_context": 100000, "multiplier": 1.0},
        {"max_context": 1000000, "multiplier": 5.0},
    ],
}


def _pricing(basis=None) -> PricingConfig:
    fields = {} if basis is None else {"context_tier_basis": basis}
    return PricingConfig(
        input=DimensionPrice(cost_per_1m=0.1),
        output=DimensionPrice(cost_per_1m=0.5),
        context_tiers=[
            ContextTier(max_context=100000, multiplier=1.0),
            ContextTier(max_context=1000000, multiplier=5.0),
        ],
        **fields,
    )


NEAR_THE_THRESHOLD = {"input_tokens": 95000, "output_tokens": 10000}


def test_by_default_input_and_output_choose_the_tier():
    result = PricingCalculator.compute_cost(NEAR_THE_THRESHOLD, _pricing())

    assert result["context_multiplier"] == 5.0
    assert result["cost_total_usd"] == pytest.approx(0.0725)


def test_total_is_the_default():
    assert _pricing().context_tier_basis == "total"
    assert PricingCalculator.compute_cost(
        NEAR_THE_THRESHOLD, _pricing("total")
    ) == PricingCalculator.compute_cost(NEAR_THE_THRESHOLD, _pricing())


def test_by_input_a_prompt_under_the_threshold_pays_the_base_rate():
    result = PricingCalculator.compute_cost(NEAR_THE_THRESHOLD, _pricing("input"))

    assert result["context_multiplier"] == 1.0
    assert result["cost_input_usd"] == pytest.approx(0.0095)
    assert result["cost_output_usd"] == pytest.approx(0.005)
    assert result["cost_total_usd"] == pytest.approx(0.0145)


def test_by_input_a_prompt_of_exactly_the_threshold_stays_in_the_lower_tier():
    usage = {"input_tokens": 100000, "output_tokens": 30000}

    result = PricingCalculator.compute_cost(usage, _pricing("input"))

    assert result["context_multiplier"] == 1.0


def test_by_input_a_prompt_over_the_threshold_pays_the_higher_rate_on_everything():
    usage = {"input_tokens": 100001, "output_tokens": 10}

    result = PricingCalculator.compute_cost(usage, _pricing("input"))

    assert result["context_multiplier"] == 5.0


def test_by_input_cache_reads_and_writes_count_toward_the_prompt():
    """The prompt's cached part is in ``input_tokens`` (langchain-anthropic
    adds the cache reads and writes to it), so a prompt mostly read from the
    cache still pays the higher rate when it is long."""
    pricing = PricingConfig(
        **{
            **TIERED,
            "input": DimensionPrice(cost_per_1m=0.1),
            "output": DimensionPrice(cost_per_1m=0.5),
            "extras": {
                name: DimensionPrice(**price)
                for name, price in TIERED["extras"].items()
            },
            "context_tiers": [ContextTier(**t) for t in TIERED["context_tiers"]],
            "context_tier_basis": "input",
        }
    )
    usage = {
        "input_tokens": 120000,
        "cache_read_tokens": 110000,
        "output_tokens": 500,
    }

    result = PricingCalculator.compute_cost(usage, pricing)

    assert result["context_multiplier"] == 5.0
    # 10K uncached at 0.50, 110K read from the cache at 0.05, 500 out at 2.50.
    assert result["cost_total_usd"] == pytest.approx(0.005 + 0.0055 + 0.00125)


def _config(tmp_path, monkeypatch, pricing) -> LLMConfig:
    path = tmp_path / "llm_config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "claude": {"name": "Claude", "requires_api_key": False}
                },
                "models": {"tiered": {"provider": "claude", "pricing": pricing}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_CONFIG_PATH", str(path))
    return LLMConfig()


def test_the_basis_is_read_from_llm_config(tmp_path, monkeypatch):
    config = _config(
        tmp_path, monkeypatch, {**TIERED, "context_tier_basis": "input"}
    )

    assert config.models["tiered"].pricing.context_tier_basis == "input"


def test_an_entry_without_it_keeps_the_total(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, TIERED)

    assert config.models["tiered"].pricing.context_tier_basis == "total"

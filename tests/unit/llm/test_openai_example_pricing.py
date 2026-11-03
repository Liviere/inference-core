"""The GPT-6 prices of ``llm_config.example.yaml``, on what a call reports.

From GPT-5.6 OpenAI bills a prompt's tokens at one of three rates: uncached
input, a cache read (0.1x, 0.05x for GPT-6.1 Sol) and a cache write (1.25x for
the 30 minutes' cache). The usage names reads as
``prompt_tokens_details.cached_tokens`` and writes as ``cache_write_tokens``,
both inside ``prompt_tokens``; langchain-openai calls them ``cache_read`` and
``cache_creation``. A response's usage goes through langchain-openai's usage
metadata, the cost tracking's usage fragment, the key aliases and the
calculator, as in a real call, and must be priced at those three rates, with
nothing marked estimated.
"""

from pathlib import Path

import pytest
from langchain_openai.chat_models.base import _create_usage_metadata

from inference_core.agents.middleware.cost_tracking import CostTrackingMiddleware
from inference_core.llm.config import LLMConfig
from inference_core.llm.usage_logging import PricingCalculator, UsageNormalizer

EXAMPLE = Path(__file__).parents[3] / "llm_config.example.yaml"

#: Per million tokens: input, output, cache read, cache write (30 minutes).
#: https://developers.openai.com/api/docs/pricing
PRICES = {
    "gpt-6.1-sol": (2, 10, 0.1, 2.5),
    "gpt-6-luna": (0.1, 0.5, 0.01, 0.125),
}

USAGE = {
    "prompt_tokens": 5000,
    "completion_tokens": 300,
    "total_tokens": 5300,
    "prompt_tokens_details": {"cached_tokens": 3000, "cache_write_tokens": 1500},
    "completion_tokens_details": {"reasoning_tokens": 120},
}


@pytest.fixture(scope="module")
def example_config():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LLM_CONFIG_PATH", str(EXAMPLE))
        return LLMConfig()


def _cost(pricing, usage: dict) -> tuple[dict, dict]:
    metadata = _create_usage_metadata(usage)
    fragment, _extra = CostTrackingMiddleware._extract_usage_fragment(dict(metadata))
    normalized = UsageNormalizer.normalize_usage(fragment, pricing.key_aliases)
    return normalized, PricingCalculator.compute_cost(normalized, pricing)


@pytest.mark.parametrize("name", sorted(PRICES))
def test_reads_and_writes_at_their_rates(example_config, name):
    entry_input, output, read, write = PRICES[name]
    pricing = example_config.get_model_config(name).pricing

    normalized, cost = _cost(pricing, USAGE)

    assert normalized["cache_read_tokens"] == 3000
    assert normalized["cache_creation_tokens"] == 1500
    assert cost["cost_estimated"] is False
    expected = (500 * entry_input + 3000 * read + 1500 * write + 300 * output) / 1e6
    # The entries round to six decimals.
    assert cost["cost_total_usd"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("name", sorted(PRICES))
def test_a_call_that_writes_nothing_is_priced_alike(example_config, name):
    entry_input, output, read, _write = PRICES[name]
    pricing = example_config.get_model_config(name).pricing
    usage = {
        **USAGE,
        "prompt_tokens_details": {"cached_tokens": 3000, "cache_write_tokens": 0},
    }

    _normalized, cost = _cost(pricing, usage)

    assert cost["cost_estimated"] is False
    expected = (2000 * entry_input + 3000 * read + 300 * output) / 1e6
    # The entries round to six decimals.
    assert cost["cost_total_usd"] == pytest.approx(expected, abs=1e-6)

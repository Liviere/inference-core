"""The Claude prices of ``llm_config.example.yaml``, on what a call reports.

A Claude response's usage goes through langchain-anthropic's usage metadata,
the cost tracking's usage fragment, the pricing's key aliases and the
calculator, as in a real call. The example's Claude entries must then price:

- cache reads at the read rate, taken out of the input tokens;
- 5-minute and 1-hour cache writes at their own rates (1.25x and 2x input),
  taken out of the input tokens, which langchain-anthropic counts them in;
- the rest of the input at the input rate, the output (thinking included) at
  the output rate;

and a call that reasons is not marked estimated.
"""

from pathlib import Path

import pytest
from anthropic.types import CacheCreation, OutputTokensDetails, Usage
from langchain_anthropic.chat_models import _create_usage_metadata

from inference_core.agents.middleware.cost_tracking import CostTrackingMiddleware
from inference_core.llm.config import LLMConfig
from inference_core.llm.usage_logging import PricingCalculator, UsageNormalizer

EXAMPLE = Path(__file__).parents[3] / "llm_config.example.yaml"

#: Per million tokens: input, output, 5-minute write, 1-hour write, read.
#: https://platform.claude.com/docs/en/about-claude/pricing
PRICES = {
    "claude-opus-5-5": (4, 20, 5, 8, 0.2),
    "claude-sonnet-5-5": (2, 10, 2.5, 4, 0.1),
    "claude-haiku-5-5": (0.1, 0.5, 0.125, 0.2, 0.01),
}


@pytest.fixture(scope="module")
def example_config():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LLM_CONFIG_PATH", str(EXAMPLE))
        return LLMConfig()


def _cost(pricing, usage: Usage) -> dict:
    metadata = _create_usage_metadata(usage)
    fragment, _extra = CostTrackingMiddleware._extract_usage_fragment(dict(metadata))
    normalized = UsageNormalizer.normalize_usage(fragment, pricing.key_aliases)
    return PricingCalculator.compute_cost(normalized, pricing)


def test_every_claude_model_of_the_example_is_listed(example_config):
    claude = {
        name
        for name, model in example_config.models.items()
        if str(getattr(model.provider, "value", model.provider)) == "claude"
    }
    assert claude == set(PRICES)


@pytest.mark.parametrize("name", sorted(PRICES))
def test_cache_writes_reads_and_thinking_at_their_rates(example_config, name):
    inp, out, write_5m, write_1h, read = PRICES[name]
    pricing = example_config.models[name].pricing
    usage = Usage(
        input_tokens=1_000,  # Anthropic's: the input not read from or written to the cache
        cache_read_input_tokens=20_000,
        cache_creation_input_tokens=5_000,
        cache_creation=CacheCreation(
            ephemeral_5m_input_tokens=3_000, ephemeral_1h_input_tokens=2_000
        ),
        output_tokens=600,
        output_tokens_details=OutputTokensDetails(thinking_tokens=450),
    )

    result = _cost(pricing, usage)

    expected = (
        1_000 * inp
        + 20_000 * read
        + 3_000 * write_5m
        + 2_000 * write_1h
        + 600 * out
    ) / 1_000_000
    assert result["cost_total_usd"] == pytest.approx(expected, abs=1e-6)
    assert result["cost_input_usd"] == pytest.approx(1_000 * inp / 1e6, abs=1e-6)
    assert result["cost_output_usd"] == pytest.approx(600 * out / 1e6, abs=1e-6)
    assert result["cost_estimated"] is False


@pytest.mark.parametrize("name", sorted(PRICES))
def test_a_cache_write_without_the_ttl_breakdown(example_config, name):
    inp, out, write_5m, _write_1h, _read = PRICES[name]
    pricing = example_config.models[name].pricing
    usage = Usage(
        input_tokens=1_000,
        cache_creation_input_tokens=4_000,
        output_tokens=100,
    )

    result = _cost(pricing, usage)

    expected = (1_000 * inp + 4_000 * write_5m + 100 * out) / 1_000_000
    assert result["cost_total_usd"] == pytest.approx(expected, abs=1e-6)
    assert result["cost_estimated"] is False


def test_haiku_over_100k_tokens_at_five_times_the_prices(example_config):
    pricing = example_config.models["claude-haiku-5-5"].pricing
    usage = Usage(input_tokens=150_000, output_tokens=1_000)

    result = _cost(pricing, usage)

    assert result["cost_total_usd"] == pytest.approx(
        (150_000 * 0.5 + 1_000 * 2.5) / 1_000_000, abs=1e-6
    )


def test_every_priced_model_of_the_example_names_its_breakdown_keys(example_config):
    """A call that reports total and reasoning tokens is not estimated."""
    for name, model in example_config.models.items():
        if model.pricing is None:
            continue
        assert {"total_tokens", "reasoning_tokens"} <= set(
            model.pricing.extras_policy.breakdown_keys
        ), name

"""The Claude prices of ``llm_config.example.yaml``, on what a call reports.

A Claude response's usage goes through langchain-anthropic's usage metadata,
the cost tracking's usage fragment, the pricing's key aliases and the
calculator, as in a real call. The example's Claude entries must then price:

- cache reads at the read rate, taken out of the input tokens;
- 5-minute and 1-hour cache writes at their own rates (1.25x and 2x input),
  taken out of the input tokens, which langchain-anthropic counts them in;
- a write reported without its lifetime (a streamed response) at the rate of
  the entry's ``prompt_cache`` lifetime, an hour (OO-451);
- the rest of the input at the input rate, the output (thinking included) at
  the output rate;

and a call that reasons is not marked estimated.
"""

from pathlib import Path

import pytest
from anthropic.types import (
    CacheCreation,
    MessageDeltaUsage,
    OutputTokensDetails,
    Usage,
)
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
        1_000 * inp + 20_000 * read + 3_000 * write_5m + 2_000 * write_1h + 600 * out
    ) / 1_000_000
    assert result["cost_total_usd"] == pytest.approx(expected, abs=1e-6)
    assert result["cost_input_usd"] == pytest.approx(1_000 * inp / 1e6, abs=1e-6)
    assert result["cost_output_usd"] == pytest.approx(600 * out / 1e6, abs=1e-6)
    assert result["cost_estimated"] is False


@pytest.mark.parametrize("name", sorted(PRICES))
def test_the_claude_models_of_the_example_cache_for_an_hour(example_config, name):
    assert example_config.models[name].prompt_cache == "1h"


@pytest.mark.parametrize("name", sorted(PRICES))
def test_a_cache_write_without_the_ttl_breakdown_is_an_hours(example_config, name):
    inp, out, _write_5m, write_1h, _read = PRICES[name]
    pricing = example_config.models[name].pricing
    usage = Usage(
        input_tokens=1_000,
        cache_creation_input_tokens=4_000,
        output_tokens=100,
    )

    result = _cost(pricing, usage)

    expected = (1_000 * inp + 4_000 * write_1h + 100 * out) / 1_000_000
    assert result["cost_total_usd"] == pytest.approx(expected, abs=1e-6)
    assert result["cost_estimated"] is False


#: Haiku 5.5: 1,000 tokens of input, 4,000 written to the cache for an hour,
#: 500 of output: 0.0001 + 0.0008 + 0.00025 USD. At the 5-minute rate the
#: write would be 0.0005 and the call 0.00085; as plain input, 0.00075.
HOUR_WRITE_ON_HAIKU = 0.00115


def test_a_streamed_hours_write_on_haiku(example_config):
    """What a stream reports: the message_delta event's usage, no breakdown."""
    pricing = example_config.models["claude-haiku-5-5"].pricing
    usage = MessageDeltaUsage(
        input_tokens=1_000,
        cache_creation_input_tokens=4_000,
        cache_read_input_tokens=0,
        output_tokens=500,
    )

    result = _cost(pricing, usage)

    assert result["cost_total_usd"] == pytest.approx(HOUR_WRITE_ON_HAIKU, abs=1e-9)
    assert result["extra_costs"]["cache_write_1h_tokens"] == pytest.approx(0.0008)


def test_the_same_write_broken_down_costs_the_same(example_config):
    pricing = example_config.models["claude-haiku-5-5"].pricing
    usage = Usage(
        input_tokens=1_000,
        cache_creation_input_tokens=4_000,
        cache_creation=CacheCreation(
            ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=4_000
        ),
        output_tokens=500,
    )

    result = _cost(pricing, usage)

    assert result["cost_total_usd"] == pytest.approx(HOUR_WRITE_ON_HAIKU, abs=1e-9)


@pytest.mark.parametrize("name", sorted(PRICES))
def test_the_server_clears_tool_results_from_30k_tokens(example_config, name):
    [edit] = example_config.models[name].context_management["edits"]

    assert edit["trigger"] == {"type": "input_tokens", "value": 30000}


def test_the_example_reads_without_a_cache_price_warning(caplog):
    with caplog.at_level("WARNING"), pytest.MonkeyPatch.context() as mp:
        mp.setenv("LLM_CONFIG_PATH", str(EXAMPLE))
        LLMConfig()

    assert "caches its prompts" not in caplog.text


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


@pytest.mark.parametrize("name", sorted(PRICES))
def test_claude_models_of_the_example_drop_stale_thinking_blocks(example_config, name):
    """Each binds thinking blocks to the conversation; the example asks the
    API to drop a block whose conversation changed rather than refuse."""
    assert example_config.models[name].thinking_block_binding == "drop_block"

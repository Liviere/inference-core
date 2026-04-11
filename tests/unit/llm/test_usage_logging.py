"""
Unit tests for LLM usage logging functionality
"""

from decimal import Decimal

import pytest

from inference_core.llm.config import (
    ContextTier,
    DimensionPrice,
    ExtrasPolicy,
    PricingConfig,
    RoundingConfig,
    UsageLoggingConfig,
)
from inference_core.llm.usage_logging import (
    PricingCalculator,
    UsageNormalizer,
    UsageSession,
)


class TestUsageNormalizer:
    """Test usage data normalization"""

    def test_normalize_usage_with_aliases(self):
        """Test normalizing usage data with key aliases"""
        raw_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "reasoning_tokens": 25,
            "total_tokens": 175,
        }

        aliases = {
            "prompt_tokens": "input_tokens",
            "completion_tokens": "output_tokens",
        }

        normalized = UsageNormalizer.normalize_usage(raw_usage, aliases)

        assert normalized["input_tokens"] == 100
        assert normalized["output_tokens"] == 50
        assert normalized["reasoning_tokens"] == 25  # No alias, kept as-is
        assert normalized["total_tokens"] == 175

    def test_normalize_usage_no_aliases(self):
        """Test normalizing usage data without aliases"""
        raw_usage = {
            "input_tokens": 100,
            "output_tokens": 50,
        }

        normalized = UsageNormalizer.normalize_usage(raw_usage, {})

        assert normalized == raw_usage


class TestPricingCalculator:
    """Test pricing calculations"""

    def test_compute_cost_basic(self):
        """Test basic cost calculation with input and output tokens"""
        pricing_config = PricingConfig(
            input=DimensionPrice(cost_per_1k=2.0),
            output=DimensionPrice(cost_per_1k=10.0),
        )

        usage = {
            "input_tokens": 1000,
            "output_tokens": 500,
        }

        result = PricingCalculator.compute_cost(usage, pricing_config)

        assert result["cost_input_usd"] == 2.0  # 1000/1000 * 2.0
        assert result["cost_output_usd"] == 5.0  # 500/1000 * 10.0
        assert result["cost_extras_usd"] == 0.0
        assert result["cost_total_usd"] == 7.0
        assert result["cost_estimated"] is False
        assert result["context_multiplier"] == 1.0

    def test_compute_cost_with_extras(self):
        """Test additive extra dimensions explicitly marked as separate charges."""
        pricing_config = PricingConfig(
            input=DimensionPrice(cost_per_1k=2.0),
            output=DimensionPrice(cost_per_1k=10.0),
            extras={
                "reasoning_tokens": DimensionPrice(
                    cost_per_1k=15.0,
                    billed_separately=True,
                ),
                "cache_write_tokens": DimensionPrice(
                    cost_per_1k=1.0,
                    billed_separately=True,
                ),
            },
        )

        usage = {
            "input_tokens": 1000,
            "output_tokens": 500,
            "reasoning_tokens": 200,
            "cache_write_tokens": 100,
        }

        result = PricingCalculator.compute_cost(usage, pricing_config)

        assert result["cost_input_usd"] == 2.0
        assert result["cost_output_usd"] == 5.0
        assert result["cost_extras_usd"] == 3.1  # (200/1000 * 15.0) + (100/1000 * 1.0)
        assert result["cost_total_usd"] == 10.1
        assert result["extra_costs"]["reasoning_tokens"] == 3.0
        assert result["extra_costs"]["cache_write_tokens"] == 0.1

    def test_compute_cost_defaults_reasoning_to_output_breakdown(self):
        """Reasoning tokens are treated as part of output_tokens by default."""
        pricing_config = PricingConfig(
            input=DimensionPrice(cost_per_1k=2.0),
            output=DimensionPrice(cost_per_1k=10.0),
            extras={
                "reasoning_tokens": DimensionPrice(cost_per_1k=15.0),
            },
        )

        usage = {
            "input_tokens": 1000,
            "output_tokens": 500,
            "reasoning_tokens": 200,
        }

        result = PricingCalculator.compute_cost(usage, pricing_config)

        assert result["cost_input_usd"] == 2.0
        assert result["cost_output_usd"] == 3.0
        assert result["extra_costs"]["reasoning_tokens"] == 3.0
        assert result["cost_total_usd"] == 8.0

    def test_compute_cost_with_context_multiplier(self):
        """Test cost calculation with context tier multipliers"""
        pricing_config = PricingConfig(
            input=DimensionPrice(cost_per_1k=2.0),
            output=DimensionPrice(cost_per_1k=10.0),
            context_tiers=[
                ContextTier(max_context=10000, multiplier=1.0),
                ContextTier(max_context=50000, multiplier=1.5),
                ContextTier(max_context=100000, multiplier=2.0),
            ],
        )

        # Usage that fits in the second tier
        usage = {
            "input_tokens": 30000,
            "output_tokens": 5000,
        }

        result = PricingCalculator.compute_cost(usage, pricing_config)

        assert result["context_multiplier"] == 1.5
        assert result["cost_input_usd"] == 90.0  # 30000/1000 * 2.0 * 1.5
        assert result["cost_output_usd"] == 75.0  # 5000/1000 * 10.0 * 1.5

    def test_compute_cost_with_unpriced_tokens(self):
        """Test cost calculation with unpriced token dimensions"""
        pricing_config = PricingConfig(
            input=DimensionPrice(cost_per_1k=2.0),
            output=DimensionPrice(cost_per_1k=10.0),
            extras_policy=ExtrasPolicy(passthrough_unpriced=True),
        )

        usage = {
            "input_tokens": 1000,
            "output_tokens": 500,
            "unknown_tokens": 100,  # Not priced
        }

        result = PricingCalculator.compute_cost(usage, pricing_config)

        assert result["cost_estimated"] is True
        assert result["cost_total_usd"] == 7.0  # Only priced tokens

    def test_compute_cost_with_rounding(self):
        """Test cost calculation with rounding"""
        pricing_config = PricingConfig(
            input=DimensionPrice(cost_per_1k=2.333333),
            output=DimensionPrice(cost_per_1k=10.777777),
            rounding=RoundingConfig(decimals=2),
        )

        usage = {
            "input_tokens": 1000,
            "output_tokens": 500,
        }

        result = PricingCalculator.compute_cost(usage, pricing_config)

        assert result["cost_input_usd"] == 2.33
        assert result["cost_output_usd"] == 5.39
        assert result["cost_total_usd"] == 7.72


class TestUsageSession:
    """Test usage session management"""

    def test_session_creation(self):
        """Test creating a usage session"""
        session = UsageSession(
            task_type="completion",
            request_mode="sync",
            model_name="gpt-4",
            provider="openai",
        )

        assert session.task_type == "completion"
        assert session.request_mode == "sync"
        assert session.model_name == "gpt-4"
        assert session.provider == "openai"
        assert session.accumulated_usage == {}

    def test_accumulate_usage(self):
        """Test accumulating usage data"""
        session = UsageSession(
            task_type="completion",
            request_mode="sync",
            model_name="gpt-4",
            provider="openai",
        )

        # First fragment
        session.accumulate({"input_tokens": 100, "output_tokens": 50})
        assert session.accumulated_usage["input_tokens"] == 100
        assert session.accumulated_usage["output_tokens"] == 50

        # Second fragment (streaming)
        session.accumulate({"output_tokens": 25, "reasoning_tokens": 10})
        assert session.accumulated_usage["input_tokens"] == 100
        assert session.accumulated_usage["output_tokens"] == 75  # 50 + 25
        assert session.accumulated_usage["reasoning_tokens"] == 10

    def test_accumulate_non_numeric_values(self):
        """Test accumulating non-numeric values (latest wins)"""
        session = UsageSession(
            task_type="completion",
            request_mode="sync",
            model_name="gpt-4",
            provider="openai",
        )

        session.accumulate({"model_name": "gpt-4-old", "tokens": 100})
        session.accumulate({"model_name": "gpt-4-new", "tokens": 50})

        assert session.accumulated_usage["model_name"] == "gpt-4-new"
        assert session.accumulated_usage["tokens"] == 150


class TestUsageLoggingConfig:
    """Test usage logging configuration"""

    def test_default_config(self):
        """Test default usage logging configuration"""
        config = UsageLoggingConfig()

        assert config.enabled is True
        assert config.base_currency == "USD"
        assert config.fail_open is True
        assert config.default_rounding_decimals == 6

    def test_custom_config(self):
        """Test custom usage logging configuration"""
        config = UsageLoggingConfig(
            enabled=False,
            base_currency="EUR",
            fail_open=False,
            default_rounding_decimals=4,
        )

        assert config.enabled is False
        assert config.base_currency == "EUR"
        assert config.fail_open is False
        assert config.default_rounding_decimals == 4


class TestComputeCostWithDefaultBreakdowns:
    """Test default and explicit parent resolution for priced token breakdowns."""

    def test_cache_read_deducted_from_input(self):
        """Cached input is treated as an input breakdown by default."""
        pricing_config = PricingConfig(
            input=DimensionPrice(cost_per_1k=0.00005),  # $0.05 / 1M
            output=DimensionPrice(cost_per_1k=0.0004),  # $0.40 / 1M
            extras={
                "cache_read_tokens": DimensionPrice(
                    cost_per_1k=0.000005,  # $0.005 / 1M
                ),
            },
            rounding=RoundingConfig(decimals=6),
        )

        usage = {
            "input_tokens": 28225,
            "output_tokens": 2670,
            "cache_read_tokens": 24576,
        }

        result = PricingCalculator.compute_cost(usage, pricing_config)

        # fresh input = 28225 - 24576 = 3649
        expected_input = round(3649 / 1000.0 * 0.00005, 6)
        expected_cache = round(24576 / 1000.0 * 0.000005, 6)
        expected_output = round(2670 / 1000.0 * 0.0004, 6)

        assert result["cost_input_usd"] == expected_input
        assert result["extra_costs"]["cache_read_tokens"] == expected_cache
        assert result["cost_output_usd"] == expected_output
        assert result["cost_total_usd"] == round(
            expected_input + expected_cache + expected_output, 6
        )
        assert result["cost_estimated"] is False

    def test_billed_separately_preserves_additive_behaviour(self):
        """billed_separately keeps an extra additive instead of deducting it."""
        pricing_config = PricingConfig(
            input=DimensionPrice(cost_per_1k=2.0),
            output=DimensionPrice(cost_per_1k=10.0),
            extras={
                "reasoning_tokens": DimensionPrice(
                    cost_per_1k=15.0,
                    billed_separately=True,
                ),
            },
        )

        usage = {
            "input_tokens": 1000,
            "output_tokens": 500,
            "reasoning_tokens": 200,
        }

        result = PricingCalculator.compute_cost(usage, pricing_config)

        assert result["cost_input_usd"] == 2.0  # unchanged, no deduction
        assert result["cost_output_usd"] == 5.0
        assert result["extra_costs"]["reasoning_tokens"] == 3.0
        assert result["cost_total_usd"] == 10.0

    def test_custom_provider_key_can_be_normalized_via_alias(self):
        """Custom provider counters should be mapped to standardized extra keys."""
        pricing_config = PricingConfig(
            input=DimensionPrice(cost_per_1k=2.0),
            output=DimensionPrice(cost_per_1k=10.0),
            extras={
                "input_special_tokens": DimensionPrice(
                    cost_per_1k=1.0,
                ),
            },
            key_aliases={
                "provider_special_tokens": "input_special_tokens",
            },
        )

        raw_usage = {
            "input_tokens": 1000,
            "provider_special_tokens": 400,
        }

        usage = UsageNormalizer.normalize_usage(raw_usage, pricing_config.key_aliases)

        result = PricingCalculator.compute_cost(usage, pricing_config)

        assert result["cost_input_usd"] == 1.2
        assert result["extra_costs"]["input_special_tokens"] == 0.4
        assert result["cost_total_usd"] == 1.6

    def test_cache_read_larger_than_input_clamped_to_zero(self):
        """If cache_read > input_tokens (edge case), fresh input is clamped to 0."""
        pricing_config = PricingConfig(
            input=DimensionPrice(cost_per_1k=0.00005),
            output=DimensionPrice(cost_per_1k=0.0004),
            extras={
                "cache_read_tokens": DimensionPrice(
                    cost_per_1k=0.000005,
                ),
            },
        )

        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_tokens": 200,  # more than input
        }

        result = PricingCalculator.compute_cost(usage, pricing_config)

        assert result["cost_input_usd"] == 0.0  # clamped

    def test_exact_sample_payload_matches_langsmith(self):
        """Reproduce the exact bug-report payload and assert expected total.

        Model: gpt-5-nano pricing ($0.05 / 1M input, $0.40 / 1M output,
        $0.005 / 1M cached input).  Expected total ≈ $0.001373.
        """
        pricing_config = PricingConfig(
            input=DimensionPrice(cost_per_1m=0.05),
            output=DimensionPrice(cost_per_1m=0.4),
            extras={
                "cache_read_tokens": DimensionPrice(
                    cost_per_1m=0.005,
                ),
            },
            key_aliases={
                "prompt_tokens": "input_tokens",
                "completion_tokens": "output_tokens",
                "cached_input": "cache_read_tokens",
            },
            rounding=RoundingConfig(decimals=6),
        )

        raw_usage = {
            "input_tokens": 28225,
            "output_tokens": 2670,
            "total_tokens": 30895,
            "cache_read_tokens": 24576,
            "reasoning_tokens": 2432,
        }

        normalized = UsageNormalizer.normalize_usage(
            raw_usage, pricing_config.key_aliases
        )
        result = PricingCalculator.compute_cost(normalized, pricing_config)

        # fresh input = 28225 - 24576 = 3649
        # input cost  = 3649 * 0.05 / 1_000_000 = 0.00018245 → rounded 0.000182
        # cache cost  = 24576 * 0.005 / 1_000_000 = 0.00012288 → rounded 0.000123
        # output cost = 2670 * 0.4 / 1_000_000 = 0.001068
        # total ≈ 0.001373
        assert result["cost_input_usd"] == 0.000182
        assert result["extra_costs"]["cache_read_tokens"] == 0.000123
        assert result["cost_output_usd"] == 0.001068
        assert result["cost_total_usd"] == 0.001373
        # reasoning_tokens is unpriced but is a known informational breakdown
        assert result["cost_estimated"] is True  # reasoning_tokens present but unpriced

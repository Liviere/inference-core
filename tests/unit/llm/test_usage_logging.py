"""
Unit tests for LLM usage logging functionality
"""

import pytest
from decimal import Decimal

from inference_core.llm.config import (
    PricingConfig, 
    DimensionPrice, 
    ContextTier, 
    RoundingConfig, 
    ExtrasPolicy,
    UsageLoggingConfig
)
from inference_core.llm.usage_logging import (
    UsageNormalizer, 
    PricingCalculator, 
    UsageSession
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
        """Test cost calculation with extra dimensions"""
        pricing_config = PricingConfig(
            input=DimensionPrice(cost_per_1k=2.0),
            output=DimensionPrice(cost_per_1k=10.0),
            extras={
                "reasoning_tokens": DimensionPrice(cost_per_1k=15.0),
                "cache_write_tokens": DimensionPrice(cost_per_1k=1.0),
            }
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
    
    def test_compute_cost_with_context_multiplier(self):
        """Test cost calculation with context tier multipliers"""
        pricing_config = PricingConfig(
            input=DimensionPrice(cost_per_1k=2.0),
            output=DimensionPrice(cost_per_1k=10.0),
            context_tiers=[
                ContextTier(max_context=10000, multiplier=1.0),
                ContextTier(max_context=50000, multiplier=1.5),
                ContextTier(max_context=100000, multiplier=2.0),
            ]
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
            extras_policy=ExtrasPolicy(passthrough_unpriced=True)
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
            rounding=RoundingConfig(decimals=2)
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
            task_type="explain",
            request_mode="sync",
            model_name="gpt-4",
            provider="openai",
        )
        
        assert session.task_type == "explain"
        assert session.request_mode == "sync"
        assert session.model_name == "gpt-4"
        assert session.provider == "openai"
        assert session.accumulated_usage == {}
    
    def test_accumulate_usage(self):
        """Test accumulating usage data"""
        session = UsageSession(
            task_type="explain",
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
            task_type="explain",
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
            default_rounding_decimals=4
        )
        
        assert config.enabled is False
        assert config.base_currency == "EUR"
        assert config.fail_open is False
        assert config.default_rounding_decimals == 4
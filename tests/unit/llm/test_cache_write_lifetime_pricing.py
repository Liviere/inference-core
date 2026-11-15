"""A cache write priced at the lifetime the called model asked for.

A streamed Anthropic response reports its cache write without a lifetime
(``message_delta`` names only ``cache_creation_input_tokens``), and the
pricing's alias of ``cache_creation_tokens`` prices it at one rate, the
entry's ``prompt_cache``. A model built with another lifetime than its
entry's (a config with the entry's ``prompt_cache`` overridden) asks for that
lifetime, and with ``CostTrackingMiddleware.cache_write_by_model_lifetime``
its write is priced at it, in the usage log's row and in the state's cost.

The usage goes through langchain-anthropic's usage metadata and the real
middleware, on the Haiku 5.5 prices of ``llm_config.example.yaml``.
"""

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import CacheCreation, MessageDeltaUsage, Usage
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import _create_usage_metadata
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from inference_core.agents.middleware.cost_tracking import (
    CostTrackingMiddleware,
    _MiddlewareContext,
    cache_write_of_lifetime,
)
from inference_core.llm.config import DimensionPrice, LLMConfig, PricingConfig
from inference_core.llm.model_identity import tag_model_identity
from inference_core.llm.usage_logging import PricingCalculator, UsageNormalizer

EXAMPLE = Path(__file__).parents[3] / "llm_config.example.yaml"
HAIKU = "claude-haiku-5-5"

#: Haiku 5.5: 1,000 tokens of input, 4,000 written to the cache, 500 of
#: output. An hour's write: 0.0001 + 0.0008 + 0.00025 USD; five minutes':
#: 0.0001 + 0.0005 + 0.00025.
HOUR = 0.00115
FIVE_MINUTES = 0.00085


@pytest.fixture(scope="module")
def example_config():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LLM_CONFIG_PATH", str(EXAMPLE))
        return LLMConfig()


@pytest.fixture
def options(monkeypatch):
    """Both options as a consumer that prices by the lifetime sets them."""

    def set_options(by_lifetime: bool) -> None:
        monkeypatch.setattr(
            CostTrackingMiddleware, "cache_write_by_model_lifetime", by_lifetime
        )
        monkeypatch.setattr(CostTrackingMiddleware, "state_cost_with_key_aliases", True)

    return set_options


def _claude(lifetime):
    """A Claude model as the factory builds it for a ``prompt_cache``."""
    kwargs = (
        {"model_kwargs": {"cache_control": {"type": "ephemeral", "ttl": lifetime}}}
        if lifetime
        else {}
    )
    model = ChatAnthropic(model=HAIKU, api_key="test", **kwargs)
    tag_model_identity(model, name=HAIKU, provider="claude")
    return model


def _streamed():
    """What a stream reports: the message_delta event's usage."""
    return MessageDeltaUsage(
        input_tokens=1_000,
        cache_creation_input_tokens=4_000,
        cache_read_input_tokens=0,
        output_tokens=500,
    )


def _broken_down(five_minutes: int, hour: int):
    """What a response without a stream reports: the write by lifetime."""
    return Usage(
        input_tokens=1_000,
        cache_creation_input_tokens=five_minutes + hour,
        cache_creation=CacheCreation(
            ephemeral_5m_input_tokens=five_minutes, ephemeral_1h_input_tokens=hour
        ),
        output_tokens=500,
    )


def _llm_config(models):
    config = MagicMock()
    config.models = {
        name: MagicMock(provider=provider, pricing=pricing)
        for name, (provider, pricing) in models.items()
    }
    return config


def _logged(pricing, model, usage, name=HAIKU, provider="claude"):
    """Log one call; the usage the row got, its cost and the state's cost."""
    middleware = CostTrackingMiddleware(
        pricing_config=pricing,
        provider=provider,
        model_name=name,
        llm_config=_llm_config({name: (provider, pricing)}),
    )
    middleware._ctx = _MiddlewareContext()
    message = MagicMock()
    message.usage_metadata = dict(_create_usage_metadata(usage)) if usage else {
        "input_tokens": 1_000,
        "output_tokens": 500,
        "total_tokens": 1_500,
    }
    message.response_metadata = {"model_name": name}
    message.content = "ok"
    response = MagicMock()
    response.result = [message]
    request = MagicMock()
    request.messages = []
    request.system_message = None
    request.model = model
    with patch(
        "inference_core.agents.middleware.cost_tracking.UsageSession"
    ) as MockSession:
        MockSession.return_value.finalize_sync.return_value = uuid.uuid4()
        middleware._persist_from_response(response, request)
    final_usage = MockSession.return_value.finalize_sync.call_args.kwargs[
        "final_usage"
    ]
    normalized = UsageNormalizer.normalize_usage(final_usage, pricing.key_aliases)
    row_cost = float(
        PricingCalculator.compute_cost(normalized, pricing)["cost_total_usd"]
    )
    return final_usage, row_cost, middleware._ctx.persisted.cost_usd


@pytest.mark.parametrize(
    ("lifetime", "expected"), [("5m", FIVE_MINUTES), ("1h", HOUR)]
)
def test_a_streamed_write_is_priced_at_the_models_lifetime(
    example_config, options, lifetime, expected
):
    options(True)
    pricing = example_config.models[HAIKU].pricing

    usage, row_cost, state_cost = _logged(pricing, _claude(lifetime), _streamed())

    assert row_cost == pytest.approx(expected, abs=1e-9)
    assert state_cost == pytest.approx(row_cost, abs=1e-12)
    # The row says which lifetime was written.
    assert "cache_creation_tokens" not in usage
    assert usage[f"ephemeral_{lifetime}_input_tokens_tokens"] == 4_000


def test_by_default_the_entrys_alias_prices_it(example_config, options):
    options(False)
    pricing = example_config.models[HAIKU].pricing

    usage, row_cost, state_cost = _logged(pricing, _claude("5m"), _streamed())

    assert row_cost == pytest.approx(HOUR, abs=1e-9)
    assert state_cost == pytest.approx(HOUR, abs=1e-9)
    assert usage["cache_creation_tokens"] == 4_000


def test_a_model_without_the_automatic_cache_keeps_the_entrys_alias(
    example_config, options
):
    """A model whose cache_control was taken off (a breakpoint of its own on
    the system prompt, say) says no lifetime."""
    options(True)
    pricing = example_config.models[HAIKU].pricing

    usage, row_cost, _state = _logged(pricing, _claude(None), _streamed())

    assert row_cost == pytest.approx(HOUR, abs=1e-9)
    assert usage["cache_creation_tokens"] == 4_000


@pytest.mark.parametrize("lifetime", ["5m", "1h"])
def test_a_write_broken_down_by_the_response_is_priced_as_it_says(
    example_config, options, lifetime
):
    options(True)
    pricing = example_config.models[HAIKU].pricing

    _usage, five, _ = _logged(pricing, _claude(lifetime), _broken_down(4_000, 0))
    _usage, hour, _ = _logged(pricing, _claude(lifetime), _broken_down(0, 4_000))

    assert five == pytest.approx(FIVE_MINUTES, abs=1e-9)
    assert hour == pytest.approx(HOUR, abs=1e-9)


def test_a_pricing_without_the_lifetimes_key_is_priced_as_before(options):
    """An entry that prices only the generic write has no rate of its own for
    a lifetime: its alias prices it, as before."""
    options(True)
    pricing = PricingConfig(
        input=DimensionPrice(cost_per_1m=0.1),
        output=DimensionPrice(cost_per_1m=0.5),
        extras={"cache_write_1h_tokens": DimensionPrice(cost_per_1m=0.2)},
        key_aliases={"cache_creation_tokens": "cache_write_1h_tokens"},
    )

    usage, row_cost, state_cost = _logged(pricing, _claude("5m"), _streamed())

    assert row_cost == pytest.approx(HOUR, abs=1e-9)
    assert state_cost == pytest.approx(HOUR, abs=1e-9)
    assert usage["cache_creation_tokens"] == 4_000


def test_a_fallback_of_another_provider_is_unchanged(options):
    options(True)
    pricing = PricingConfig(
        input=DimensionPrice(cost_per_1m=0.1),
        output=DimensionPrice(cost_per_1m=0.3),
    )
    model = FakeListChatModel(responses=["ok"])
    tag_model_identity(model, name="mistral-small-latest", provider="mistral")

    usage, row_cost, state_cost = _logged(
        pricing, model, None, name="mistral-small-latest", provider="mistral"
    )

    assert usage == {"input_tokens": 1_000, "output_tokens": 500, "total_tokens": 1_500}
    assert row_cost == pytest.approx(1_000 * 0.1e-6 + 500 * 0.3e-6)
    assert state_cost == pytest.approx(row_cost)


def test_the_extra_tokens_follow_the_fragment(example_config):
    pricing = example_config.models[HAIKU].pricing
    fragment = {"input_tokens": 5_000, "cache_creation_tokens": 4_000}
    extra = {"cache_creation_tokens": 4_000, "cache_read_tokens": 10}

    renamed, renamed_extra = cache_write_of_lifetime(fragment, extra, "5m", pricing)

    assert renamed == {
        "input_tokens": 5_000,
        "ephemeral_5m_input_tokens_tokens": 4_000,
    }
    assert renamed_extra == {
        "ephemeral_5m_input_tokens_tokens": 4_000,
        "cache_read_tokens": 10,
    }
    # The arguments are not changed.
    assert fragment == {"input_tokens": 5_000, "cache_creation_tokens": 4_000}
    assert cache_write_of_lifetime(fragment, extra, None, pricing) == (fragment, extra)

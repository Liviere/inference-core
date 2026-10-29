"""A cache write reported without its lifetime, priced for the entry's.

A streamed Claude response names how many tokens it wrote to the cache
(``cache_creation_tokens``) but not for how long. A model whose
``prompt_cache`` is ``1h`` should price that key as an hour's write (2x
input). When the file prices it at another rate, a warning names the model;
the pricing is used as it is.
"""

import pytest
import yaml

from inference_core.llm.config import LLMConfig

PRICING = {
    "input": {"cost_per_1m": 0.1},
    "output": {"cost_per_1m": 0.5},
    "extras": {
        "cache_write_tokens": {"cost_per_1m": 0.125},
        "cache_write_1h_tokens": {"cost_per_1m": 0.2},
        "cache_read_tokens": {"cost_per_1m": 0.01},
    },
}


def _load(tmp_path, monkeypatch, entry) -> LLMConfig:
    path = tmp_path / "llm_config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "providers": {"claude": {"name": "Claude", "requires_api_key": False}},
                "models": {"claude-haiku-5-5": {"provider": "claude", **entry}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_CONFIG_PATH", str(path))
    return LLMConfig()


def _entry(prompt_cache, alias, *, price_multiplier=None):
    pricing = {**PRICING}
    if alias is not None:
        pricing["key_aliases"] = {"cache_creation_tokens": alias}
    entry = {"prompt_cache": prompt_cache, "pricing": pricing}
    if price_multiplier is not None:
        entry["price_multiplier"] = price_multiplier
    return entry


@pytest.mark.parametrize(
    "prompt_cache, alias",
    [
        ("1h", "cache_write_1h_tokens"),
        ("5m", "cache_write_tokens"),
    ],
)
def test_a_write_priced_for_the_lifetime_is_no_warning(
    tmp_path, monkeypatch, caplog, prompt_cache, alias
):
    with caplog.at_level("WARNING"):
        config = _load(tmp_path, monkeypatch, _entry(prompt_cache, alias))

    assert config.models["claude-haiku-5-5"].prompt_cache == prompt_cache
    assert "caches its prompts" not in caplog.text


def test_an_hour_priced_as_five_minutes_is_named(tmp_path, monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        config = _load(tmp_path, monkeypatch, _entry("1h", "cache_write_tokens"))

    assert "'claude-haiku-5-5' caches its prompts for 1h" in caplog.text
    assert "1.25x input, not 2.00x" in caplog.text
    # The pricing is used as the file says.
    pricing = config.models["claude-haiku-5-5"].pricing
    assert pricing.key_aliases["cache_creation_tokens"] == "cache_write_tokens"


def test_an_hour_without_an_alias_is_priced_as_input_and_named(
    tmp_path, monkeypatch, caplog
):
    with caplog.at_level("WARNING"):
        _load(tmp_path, monkeypatch, _entry("1h", None))

    assert "1.00x input, not 2.00x" in caplog.text


def test_a_price_multiplier_does_not_change_the_rate(tmp_path, monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        _load(
            tmp_path,
            monkeypatch,
            _entry("1h", "cache_write_1h_tokens", price_multiplier=1.1),
        )

    assert "caches its prompts" not in caplog.text


@pytest.mark.parametrize("prompt_cache", [None, "1h"])
def test_no_cache_or_no_pricing_is_no_warning(
    tmp_path, monkeypatch, caplog, prompt_cache
):
    entry = (
        {"pricing": PRICING} if prompt_cache is None else {"prompt_cache": prompt_cache}
    )
    with caplog.at_level("WARNING"):
        _load(tmp_path, monkeypatch, entry)

    assert "caches its prompts" not in caplog.text

"""
Unit tests for a provider's ``catalog:`` block in the LLM config.

Covers:
- the defaults for a provider without the block
- switching a provider off
- a block that cannot be read is ignored, and never breaks the provider
"""

import logging

import pytest

from .conftest import make_config


def _config(block):
    entry = {"name": "OpenAI", "requires_api_key": True, "api_key_env": "X_KEY"}
    if block is not ...:
        entry["catalog"] = block
    return make_config({"openai": entry}, {"gpt-5": "openai"})


def test_default_is_enabled():
    assert _config(...).get_provider_catalog_config("openai").enabled is True


def test_unknown_provider_gets_the_defaults():
    assert _config(...).get_provider_catalog_config("nope").enabled is True


def test_can_be_switched_off():
    config = _config({"enabled": False})

    assert config.get_provider_catalog_config("openai").enabled is False


@pytest.mark.parametrize(
    "block", [{"enabled": "sometimes"}, {"enabeld": False}, "off", ["enabled"], 0]
)
def test_unreadable_block_is_ignored_with_a_warning(block, caplog):
    config = _config(block)

    with caplog.at_level(logging.WARNING):
        assert config.get_provider_catalog_config("openai").enabled is True

    assert "catalog" in caplog.text
    assert "openai" in caplog.text


@pytest.mark.parametrize("block", [{"enabled": "sometimes"}, "off", {"policy": "all"}])
def test_unreadable_block_does_not_break_the_provider(block, monkeypatch):
    monkeypatch.setenv("X_KEY", "sk-test")
    config = _config(block)

    assert config.get_provider_config("openai").requires_api_key is True
    assert config.get_provider_runtime_config("openai")["api_key"] == "sk-test"

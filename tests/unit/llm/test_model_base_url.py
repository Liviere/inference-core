"""A model entry's own base_url, read from llm_config.yaml.

One provider can serve a model from another endpoint than its own (a regional
one, say): the model's base_url wins over the provider's, and a model without
one keeps the provider's.
"""

import pytest
import yaml

from inference_core.llm.config import LLMConfig
from inference_core.llm.models import LLMModelFactory

CONFIG = {
    "providers": {
        "mistral": {
            "name": "Mistral AI",
            "api_key_env": "TEST_MISTRAL_KEY",
            "requires_api_key": True,
        },
        "custom_openai_compatible": {
            "name": "Gateway",
            "openai_compatible": True,
            "base_url": "https://gateway.example.com/v1",
            "api_key_env": "TEST_GATEWAY_KEY",
            "requires_api_key": True,
        },
    },
    "models": {
        "mistral-small-latest": {"provider": "mistral"},
        "mistral-small-2603": {
            "provider": "mistral",
            "base_url": "https://api.eu.example.com",
        },
        "gateway-model": {"provider": "custom_openai_compatible"},
        "gateway-model-eu": {
            "provider": "custom_openai_compatible",
            "base_url": "https://eu.gateway.example.com/v1",
        },
    },
}


@pytest.fixture
def config(tmp_path, monkeypatch):
    path = tmp_path / "llm_config.yaml"
    path.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    monkeypatch.setenv("LLM_CONFIG_PATH", str(path))
    monkeypatch.setenv("TEST_MISTRAL_KEY", "test-key")
    monkeypatch.setenv("TEST_GATEWAY_KEY", "test-key")
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled", lambda: False
    )
    return LLMConfig()


def test_a_model_without_one_keeps_the_providers(config):
    assert config.models["mistral-small-latest"].base_url is None
    assert (
        config.models["gateway-model"].base_url == "https://gateway.example.com/v1"
    )


def test_a_models_own_base_url_wins(config):
    assert config.models["mistral-small-2603"].base_url == "https://api.eu.example.com"
    assert (
        config.models["gateway-model-eu"].base_url
        == "https://eu.gateway.example.com/v1"
    )


def test_it_is_no_extra_parameter_of_the_model(config):
    assert "base_url" not in (config.models["mistral-small-2603"].model_extra or {})


def test_the_model_is_built_on_it(config):
    model = LLMModelFactory(config).create_model("mistral-small-2603")
    assert str(model.client.base_url).startswith("https://api.eu.example.com")

    default = LLMModelFactory(config).create_model("mistral-small-latest")
    assert "api.eu.example.com" not in str(default.client.base_url)

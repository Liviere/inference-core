"""The Cortecs gateway as a provider of chat models.

Its address is the provider's ``base_url`` and nothing else, its key comes
from the provider's ``api_key_env``, a model is called by its
``provider_model_id`` and the provider's ``request_options`` go into the body
of every request.
"""

import logging

import pytest
import yaml
from langchain_openai import ChatOpenAI

from inference_core.llm.config import LLMConfig, ModelProvider, provider_model_id
from inference_core.llm.models import LLMModelFactory

BASE_URL = "https://gateway.cortecs.example/v1"


def _config(tmp_path, monkeypatch, provider=None, models=None, key="test-key"):
    provider_entry = {
        "name": "Cortecs",
        "openai_compatible": True,
        "base_url": BASE_URL,
        "api_key_env": "TEST_CORTECS_KEY",
        "requires_api_key": True,
        "request_options": {"eu_native": True, "allow_zero_data_retention": True},
    }
    provider_entry.update(provider or {})
    config = {
        "providers": {"cortecs": provider_entry},
        "models": models
        or {
            "cortecs-gpt-5-nano": {
                "provider": "cortecs",
                "provider_model_id": "gpt-5-nano",
            },
            "mistral-small-2603": {"provider": "cortecs"},
        },
    }
    path = tmp_path / "llm_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("LLM_CONFIG_PATH", str(path))
    if key is None:
        monkeypatch.delenv("TEST_CORTECS_KEY", raising=False)
    else:
        monkeypatch.setenv("TEST_CORTECS_KEY", key)
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled", lambda: False
    )
    return LLMConfig()


def test_it_is_a_builtin_provider():
    assert ModelProvider("cortecs") is ModelProvider.CORTECS


def test_the_model_goes_to_the_providers_address_with_its_key(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch)

    model = LLMModelFactory(config).create_model("mistral-small-2603")

    assert isinstance(model, ChatOpenAI)
    assert str(model.openai_api_base).rstrip("/") == BASE_URL
    assert model.openai_api_key.get_secret_value() == "test-key"
    assert model.model_name == "mistral-small-2603"


def test_a_model_is_called_by_its_provider_model_id(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch)

    model = LLMModelFactory(config).create_model("cortecs-gpt-5-nano")

    assert model.model_name == "gpt-5-nano"
    assert provider_model_id(config.models["cortecs-gpt-5-nano"]) == "gpt-5-nano"
    assert provider_model_id(config.models["mistral-small-2603"]) == (
        "mistral-small-2603"
    )


def test_provider_model_id_is_no_parameter_of_the_model(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch)

    model = LLMModelFactory(config).create_model("cortecs-gpt-5-nano")

    assert "provider_model_id" not in (model.model_kwargs or {})


def test_the_request_options_go_into_every_request(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch)

    model = LLMModelFactory(config).create_model("mistral-small-2603")

    assert model.extra_body == {
        "eu_native": True,
        "allow_zero_data_retention": True,
    }
    # The client hands ``extra_body`` to the OpenAI SDK, which merges it into
    # the JSON body of the request.
    payload = model._get_request_payload([("user", "hi")])
    assert payload["extra_body"] == model.extra_body


def test_usage_is_asked_for_when_streaming(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch)

    model = LLMModelFactory(config).create_model("mistral-small-2603")

    assert model.stream_usage is True


def test_without_request_options_nothing_is_added(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch, provider={"request_options": {}})

    model = LLMModelFactory(config).create_model("mistral-small-2603")

    assert not model.extra_body


def test_without_a_key_there_is_no_model(tmp_path, monkeypatch, caplog):
    config = _config(tmp_path, monkeypatch, key=None)

    with caplog.at_level(logging.ERROR):
        model = LLMModelFactory(config).create_model("mistral-small-2603")

    assert model is None
    assert "CORTECS_API_KEY" in caplog.text


def test_without_an_address_there_is_no_model(tmp_path, monkeypatch, caplog):
    config = _config(tmp_path, monkeypatch, provider={"base_url": None})

    with caplog.at_level(logging.ERROR):
        model = LLMModelFactory(config).create_model("mistral-small-2603")

    assert model is None
    assert "base_url" in caplog.text


def test_a_byok_key_is_used_like_the_platforms(tmp_path, monkeypatch):
    config = _config(tmp_path, monkeypatch).with_overrides(
        model_overrides={"mistral-small-2603": {"api_key": "user-key"}}
    )

    model = LLMModelFactory(config).create_model("mistral-small-2603")

    assert model.openai_api_key.get_secret_value() == "user-key"
    assert model.extra_body["eu_native"] is True


def test_request_options_default_to_none():
    from inference_core.llm.config import ProviderConfig

    assert ProviderConfig(name="any").request_options == {}

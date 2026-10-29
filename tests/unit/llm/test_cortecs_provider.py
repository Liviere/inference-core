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


# --- request_options of a model entry -------------------------------------


def _pinned(**entry):
    return {
        "cortecs-pinned": {
            "provider": "cortecs",
            "provider_model_id": "mistral-small-2603",
            **entry,
        },
        "mistral-small-2603": {"provider": "cortecs"},
    }


def test_an_entrys_request_options_go_on_top_of_the_providers(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        monkeypatch,
        models=_pinned(request_options={"allowed_providers": ["ionos"]}),
    )

    factory = LLMModelFactory(config)
    pinned = factory.create_model("cortecs-pinned")
    other = factory.create_model("mistral-small-2603")

    assert pinned.extra_body == {
        "eu_native": True,
        "allow_zero_data_retention": True,
        "allowed_providers": ["ionos"],
    }
    # The provider's other models are not pinned.
    assert other.extra_body == {"eu_native": True, "allow_zero_data_retention": True}


def test_the_entry_wins_key_by_key_and_a_list_is_replaced(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        monkeypatch,
        provider={
            "request_options": {
                "eu_native": True,
                "allowed_providers": ["ovh", "ionos"],
            }
        },
        models=_pinned(request_options={"allowed_providers": ["ionos"]}),
    )

    model = LLMModelFactory(config).create_model("cortecs-pinned")

    assert model.extra_body == {"eu_native": True, "allowed_providers": ["ionos"]}


def test_request_options_are_no_parameter_of_the_model(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        monkeypatch,
        models=_pinned(request_options={"allowed_providers": ["ionos"]}),
    )

    model = LLMModelFactory(config).create_model("cortecs-pinned")

    assert "request_options" not in (model.model_kwargs or {})
    assert "allowed_providers" not in (model.model_kwargs or {})


def test_an_entry_may_not_change_a_locked_key(tmp_path, monkeypatch):
    with pytest.raises(ValueError) as raised:
        _config(
            tmp_path,
            monkeypatch,
            provider={"locked_request_options": ["eu_native"]},
            models=_pinned(request_options={"eu_native": False}),
        )

    message = str(raised.value)
    assert "cortecs-pinned" in message and "eu_native" in message
    assert "False" not in message


def test_an_entry_may_repeat_a_locked_key_with_the_same_value(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        monkeypatch,
        provider={"locked_request_options": ["eu_native", "allow_zero_data_retention"]},
        models=_pinned(
            request_options={"eu_native": True, "allowed_providers": ["ionos"]}
        ),
    )

    model = LLMModelFactory(config).create_model("cortecs-pinned")

    assert model.extra_body["eu_native"] is True
    assert model.extra_body["allowed_providers"] == ["ionos"]


def test_an_entry_may_not_add_a_locked_key_its_provider_does_not_set(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="allowed_providers"):
        _config(
            tmp_path,
            monkeypatch,
            provider={"locked_request_options": ["allowed_providers"]},
            models=_pinned(request_options={"allowed_providers": ["ionos"]}),
        )


def test_entry_request_options_and_locks_default_to_none():
    from inference_core.llm.config import EmbeddingConfig, ModelConfig, ProviderConfig

    assert ModelConfig(name="m", provider="cortecs").request_options is None
    assert EmbeddingConfig(provider="cortecs", model="bge-m3").request_options is None
    assert ProviderConfig(name="any").locked_request_options == []


def test_merged_request_options_without_entry_options_are_the_providers():
    from inference_core.llm.config import ProviderConfig, merged_request_options

    provider = ProviderConfig(name="Cortecs", request_options={"eu_native": True})

    assert merged_request_options(provider, None, entry="e") == {"eu_native": True}
    assert merged_request_options({"request_options": {"a": 1}}, {}, entry="e") == {
        "a": 1
    }
    assert merged_request_options({}, None, entry="e") == {}

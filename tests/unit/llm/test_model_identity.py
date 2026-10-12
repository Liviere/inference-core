"""Which configured model a chat-model object is, written on it by the factory."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from inference_core.llm.config import ModelConfig, ModelProvider
from inference_core.llm.model_identity import (
    ModelIdentity,
    get_model_identity,
    provider_name,
    tag_model_identity,
)
from inference_core.llm.models import LLMModelFactory


@tool
def _noop() -> str:
    """Do nothing."""
    return ""


@pytest.mark.parametrize(
    "model",
    [
        ChatOpenAI(model="gpt-5", api_key="k"),
        ChatAnthropic(model="claude-x", api_key="k"),
        ChatMistralAI(model="mistral-small-latest", api_key="k"),
        FakeListChatModel(responses=["ok"]),
    ],
    ids=["openai", "anthropic", "mistral", "fake"],
)
def test_a_tagged_model_says_what_it_is(model):
    tag_model_identity(model, name="configured-name", provider="deepinfra")

    assert get_model_identity(model) == ModelIdentity(
        name="configured-name", provider="deepinfra"
    )


def test_the_tag_keeps_what_the_metadata_already_held():
    model = ChatOpenAI(model="gpt-5", api_key="k")
    before = dict(model.metadata or {})

    tag_model_identity(model, name="gpt-5", provider="openai")

    for key, value in before.items():
        assert model.metadata[key] == value


def test_the_tag_is_read_through_a_tool_binding_and_a_copy():
    model = ChatOpenAI(model="gpt-5", api_key="k")
    tag_model_identity(model, name="gpt-5", provider="openai")

    expected = ModelIdentity(name="gpt-5", provider="openai")
    assert get_model_identity(model.bind_tools([_noop])) == expected
    assert get_model_identity(model.model_copy()) == expected


def test_the_tag_is_not_sent_to_the_provider():
    model = ChatOpenAI(model="gpt-5", api_key="k")
    tag_model_identity(model, name="configured-name", provider="openai")

    payload = model._get_request_payload([("user", "hi")])

    assert "configured-name" not in str(payload)
    assert "metadata" not in payload


def test_a_provider_may_be_an_enum_or_missing():
    model = FakeListChatModel(responses=["ok"])

    tag_model_identity(model, name="m", provider=ModelProvider.OPENAI)
    assert get_model_identity(model) == ModelIdentity(name="m", provider="openai")

    # A second tag without a provider does not keep the first one's.
    tag_model_identity(model, name="m")
    assert get_model_identity(model) == ModelIdentity(name="m", provider=None)

    assert provider_name(None) is None
    assert provider_name("fireworks") == "fireworks"


@pytest.mark.parametrize("model", [None, object(), MagicMock(), "gpt-5"])
def test_an_untagged_object_has_no_identity(model):
    assert get_model_identity(model) is None


def test_tagging_what_cannot_be_tagged_is_quiet():
    tag_model_identity(None, name="m", provider="openai")
    tag_model_identity(object(), name="m", provider="openai")
    tag_model_identity(FakeListChatModel(responses=["ok"]), name="")


def test_the_factory_tags_what_it_builds_with_the_name_it_was_asked_for():
    """The name is the key in the config, whatever the object calls its model."""
    config = MagicMock()
    config.enable_caching = False
    config.get_model_config.return_value = ModelConfig(
        name="accounts/fireworks/models/minimax-m3",
        provider=ModelProvider.FIREWORKS,
        api_key="k",
    )
    factory = LLMModelFactory(config)
    built = FakeListChatModel(responses=["ok"])

    with patch.object(factory, "_create_model_instance", return_value=built):
        model = factory.create_model("accounts/fireworks/models/minimax-m3")

    assert get_model_identity(model) == ModelIdentity(
        name="accounts/fireworks/models/minimax-m3", provider="fireworks"
    )


def test_the_factory_tags_an_emulated_model_it_has_no_config_for(monkeypatch):
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled", lambda: True
    )
    config = MagicMock()
    config.enable_caching = False
    config.get_model_config.return_value = None

    model = LLMModelFactory(config).create_model("not-configured")

    assert get_model_identity(model) == ModelIdentity(
        name="not-configured", provider=None
    )

"""
Unit tests for chat model providers registered by the host application.

Covers:
- registration rules (name, builder, built-in names are protected)
- ModelConfig accepting a registered provider and rejecting an unknown one
- the model factory building such a model, as a primary and as a fallback
- parameter normalization through the provider's own or the default policy
- availability and provider config for a provider with no YAML entry
"""

from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from inference_core.agents.middleware.model_fallback import (
    build_model_fallback_middleware,
)
from inference_core.llm.config import (
    LLMConfig,
    ModelConfig,
    ModelProvider,
    resolve_provider_name,
)
from inference_core.llm.models import LLMModelFactory
from inference_core.llm.param_policy import ProviderParamPolicy, normalize_params
from inference_core.llm.provider_registry import (
    get_registered_provider,
    is_registered_provider,
    register_chat_model_provider,
    unregister_chat_model_provider,
)

PROVIDER = "in_house_gateway"


@pytest.fixture(autouse=True)
def _disable_llm_emulation(monkeypatch):
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled",
        lambda: False,
    )


@pytest.fixture
def built():
    """Register PROVIDER with a builder that records what it was given."""
    calls = []

    def builder(config, params):
        calls.append((config, params))
        return FakeListChatModel(responses=[f"from {config.name}"])

    register_chat_model_provider(PROVIDER, builder)
    yield calls
    unregister_chat_model_provider(PROVIDER)


def _config_with(*models: ModelConfig) -> LLMConfig:
    with patch.object(LLMConfig, "_load_config"):
        config = LLMConfig()
    config.providers = {}
    config.models = {model.name: model for model in models}
    config.agent_models = {}
    config.agent_configs = {}
    config._yaml_config = {}
    return config


class TestRegistration:
    def test_registered_provider_is_known(self, built):
        assert is_registered_provider(PROVIDER)
        assert get_registered_provider(PROVIDER).name == PROVIDER
        assert resolve_provider_name(PROVIDER) == PROVIDER

    def test_unregistering_forgets_it(self, built):
        unregister_chat_model_provider(PROVIDER)

        assert not is_registered_provider(PROVIDER)
        with pytest.raises(ValueError, match="Unknown provider"):
            resolve_provider_name(PROVIDER)

    def test_registering_again_replaces_the_builder(self, built):
        def replacement(config, params):
            return FakeListChatModel(responses=["new"])

        register_chat_model_provider(PROVIDER, replacement)

        assert get_registered_provider(PROVIDER).builder is replacement

    @pytest.mark.parametrize("name", ["", "   ", None])
    def test_empty_name_is_refused(self, name):
        with pytest.raises(ValueError):
            register_chat_model_provider(name, lambda config, params: None)

    def test_built_in_name_cannot_be_taken_over(self):
        with pytest.raises(ValueError, match="built-in"):
            register_chat_model_provider("openai", lambda config, params: None)
        assert not is_registered_provider("openai")

    def test_builder_must_be_callable(self):
        with pytest.raises(TypeError):
            register_chat_model_provider(PROVIDER, "not-a-function")
        assert not is_registered_provider(PROVIDER)


class TestModelConfig:
    def test_accepts_a_registered_provider(self, built):
        model = ModelConfig(name="gateway-large", provider=PROVIDER)

        assert model.provider == PROVIDER

    def test_rejects_an_unknown_provider(self):
        with pytest.raises(ValidationError, match="Unknown provider"):
            ModelConfig(name="gateway-large", provider="nobody_registered_this")

    def test_built_in_provider_is_stored_by_name(self):
        by_enum = ModelConfig(name="a", provider=ModelProvider.OLLAMA)
        by_name = ModelConfig(name="b", provider="ollama")

        assert by_enum.provider == by_name.provider == "ollama"
        assert by_enum.provider == ModelProvider.OLLAMA


class TestFactory:
    def test_builds_the_model_through_the_registered_builder(self, built):
        model_config = ModelConfig(name="gateway-large", provider=PROVIDER)
        factory = LLMModelFactory(_config_with(model_config))

        model = factory.create_model("gateway-large")

        assert isinstance(model, FakeListChatModel)
        assert model.invoke("hi").content == "from gateway-large"
        config, params = built[0]
        assert config.name == "gateway-large"
        assert params["temperature"] == 0.7
        assert params["request_timeout"] == 60

    def test_extra_config_fields_reach_the_builder_on_the_config(self, built):
        model_config = ModelConfig(
            name="gateway-large", provider=PROVIDER, tenant="acme"
        )
        factory = LLMModelFactory(_config_with(model_config))

        factory.create_model("gateway-large")

        config, params = built[0]
        assert config.tenant == "acme"
        # Not a call parameter the default policy knows, so it is not forwarded.
        assert "tenant" not in params

    def test_model_added_at_runtime_is_built(self, built):
        config = _config_with()
        config.add_custom_model(ModelConfig(name="added-later", provider=PROVIDER))

        model = LLMModelFactory(config).create_model("added-later")

        assert model.invoke("hi").content == "from added-later"

    def test_a_failing_builder_yields_no_model(self):
        def broken(config, params):
            raise RuntimeError("gateway is down")

        register_chat_model_provider(PROVIDER, broken)
        try:
            model_config = ModelConfig(name="gateway-large", provider=PROVIDER)
            factory = LLMModelFactory(_config_with(model_config))

            assert factory.create_model("gateway-large") is None
        finally:
            unregister_chat_model_provider(PROVIDER)

    def test_serves_as_a_fallback_model(self, built):
        model_config = ModelConfig(name="gateway-large", provider=PROVIDER)
        factory = LLMModelFactory(_config_with(model_config))

        middleware = build_model_fallback_middleware(
            model_factory=factory,
            fallback_models=["gateway-large"],
            primary_model="something-else",
        )

        assert middleware is not None
        assert [config.name for config, _ in built] == ["gateway-large"]


class TestParameters:
    def test_default_policy_keeps_the_common_parameters(self, built):
        params = normalize_params(
            PROVIDER,
            {"temperature": 0.2, "max_tokens": 64, "top_k": 40, "request_timeout": 5},
        )

        assert params == {"temperature": 0.2, "max_tokens": 64, "request_timeout": 5}

    def test_own_policy_decides_what_is_forwarded(self):
        policy = ProviderParamPolicy(
            allowed={"temperature", "top_k", "timeout"},
            renamed={"request_timeout": "timeout"},
            dropped={"max_tokens"},
        )
        register_chat_model_provider(
            PROVIDER, lambda config, params: None, param_policy=policy
        )
        try:
            params = normalize_params(
                PROVIDER,
                {
                    "temperature": 0.2,
                    "max_tokens": 64,
                    "top_k": 40,
                    "request_timeout": 5,
                },
            )
        finally:
            unregister_chat_model_provider(PROVIDER)

        assert params == {"temperature": 0.2, "top_k": 40, "timeout": 5}

    def test_unregistered_provider_is_still_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            normalize_params("nobody_registered_this", {"temperature": 0.2})


class TestConfigHelpers:
    def test_available_without_a_key_or_a_yaml_entry(self, built):
        config = _config_with(ModelConfig(name="gateway-large", provider=PROVIDER))

        assert config.is_model_available("gateway-large")
        assert config.get_provider_config(PROVIDER).name == PROVIDER
        assert config.build_model_provider_map() == {"gateway-large": PROVIDER}

    def test_yaml_entry_still_describes_the_provider(self, built):
        config = _config_with(ModelConfig(name="gateway-large", provider=PROVIDER))
        config.providers = {PROVIDER: {"name": "In-house gateway"}}

        assert config.get_provider_config(PROVIDER).name == "In-house gateway"

    def test_yaml_model_of_a_registered_provider_loads(
        self, built, tmp_path, monkeypatch
    ):
        path = tmp_path / "llm_config.yaml"
        path.write_text(
            "providers: {}\n"
            "models:\n"
            "  gateway-large:\n"
            f"    provider: '{PROVIDER}'\n"
            "    max_tokens: 512\n"
            "agents: {}\n"
        )

        monkeypatch.setenv("LLM_CONFIG_PATH", str(path))

        config = LLMConfig()

        assert config.models["gateway-large"].provider == PROVIDER
        assert config.models["gateway-large"].max_tokens == 512

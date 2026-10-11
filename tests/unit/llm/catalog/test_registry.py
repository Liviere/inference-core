"""
Unit tests for the model lister registry.

Covers:
- every built-in provider that has a listing endpoint has a lister
- registration rules, replacement and removal
"""

import pytest

from inference_core.llm.catalog import (
    get_model_lister,
    listed_providers,
    register_model_lister,
    unregister_model_lister,
)

PROVIDER = "in_house_gateway"


async def _list_models(runtime, client):
    return []


async def _get_model(runtime, client, model_id):
    return None


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    unregister_model_lister(PROVIDER)


def test_builtin_providers_have_listers():
    assert listed_providers() == [
        "claude",
        "custom_openai_compatible",
        "deepinfra",
        "fireworks",
        "gemini",
        "mistral",
        "openai",
        "xai",
    ]
    for provider in listed_providers():
        assert get_model_lister(provider).get_model is not None


def test_ollama_has_no_lister():
    assert get_model_lister("ollama") is None


def test_register_and_unregister():
    register_model_lister(PROVIDER, _list_models, get_model=_get_model)

    lister = get_model_lister(PROVIDER)
    assert lister.provider == PROVIDER
    assert lister.list_models is _list_models
    assert lister.get_model is _get_model
    assert PROVIDER in listed_providers()

    unregister_model_lister(PROVIDER)
    assert get_model_lister(PROVIDER) is None


def test_registering_again_replaces():
    register_model_lister(PROVIDER, _list_models, get_model=_get_model)
    register_model_lister(PROVIDER, _list_models)

    assert get_model_lister(PROVIDER).get_model is None


@pytest.mark.parametrize("name", ["", "   ", None])
def test_name_must_be_a_non_empty_string(name):
    with pytest.raises(ValueError):
        register_model_lister(name, _list_models)


def test_callables_are_required():
    with pytest.raises(ValueError):
        register_model_lister(PROVIDER, "not callable")
    with pytest.raises(ValueError):
        register_model_lister(PROVIDER, _list_models, get_model="not callable")

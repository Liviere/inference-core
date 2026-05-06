"""Tests for no-cost LLM emulation primitives."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from inference_core.llm.config import ModelConfig, ModelProvider
from inference_core.llm.emulation import (
    EmulatedChatModel,
    build_tool_emulation_middleware,
)
from inference_core.llm.models import LLMModelFactory


def _emulation_settings(**overrides):
    data = {
        "llm_emulation_enabled": True,
        "llm_emulation_response": "emulated response",
        "llm_emulation_profile": "deterministic",
        "llm_emulation_latency_ms": 0,
        "llm_emulation_error_rate": 0.0,
        "llm_tool_emulation_mode": "off",
        "llm_tool_emulation_include": None,
        "llm_tool_emulation_exclude": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_emulated_chat_model_invoke_and_stream_include_usage_metadata():
    model = EmulatedChatModel(model_name="unit-model", response="hello from tests")

    message = model.invoke([HumanMessage(content="ping")])
    chunks = list(model.stream([HumanMessage(content="ping")]))

    assert message.content == "hello from tests"
    assert message.response_metadata["provider"] == "emulated"
    assert message.usage_metadata["total_tokens"] > 0
    assert "".join(chunk.content for chunk in chunks) == "hello from tests"


@pytest.mark.asyncio
async def test_emulated_chat_model_async_invoke():
    model = EmulatedChatModel(model_name="unit-model", response="async response")

    message = await model.ainvoke([HumanMessage(content="ping")])

    assert message.content == "async response"
    assert message.response_metadata["model_name"] == "unit-model"


@patch("inference_core.core.config.get_settings")
@patch("inference_core.llm.models.ChatOpenAI")
def test_model_factory_returns_emulated_model_without_provider_call(
    mock_chat_openai,
    mock_get_settings,
):
    mock_get_settings.return_value = _emulation_settings()
    config = MagicMock()
    config.enable_caching = False
    config.get_model_config.return_value = ModelConfig(
        name="gpt-real",
        provider=ModelProvider.OPENAI,
        api_key=None,
    )
    factory = LLMModelFactory(config)

    model = factory.create_model("gpt-real")

    assert isinstance(model, EmulatedChatModel)
    assert model.model_name == "gpt-real"
    mock_chat_openai.assert_not_called()


@patch("inference_core.core.config.get_settings")
def test_model_factory_emulates_unknown_model_name(mock_get_settings):
    mock_get_settings.return_value = _emulation_settings()
    config = MagicMock()
    config.enable_caching = False
    config.get_model_config.return_value = None
    factory = LLMModelFactory(config)

    model = factory.create_model("db-override-model")

    assert isinstance(model, EmulatedChatModel)
    assert model.model_name == "db-override-model"


@patch("inference_core.core.config.get_settings")
def test_tool_emulator_uses_emulated_model(mock_get_settings):
    mock_get_settings.return_value = _emulation_settings(
        llm_tool_emulation_mode="all"
    )

    middleware = build_tool_emulation_middleware([])

    assert middleware is not None
    assert isinstance(middleware.model, EmulatedChatModel)
    assert middleware.model.model_name == "tool-emulator"
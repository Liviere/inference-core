"""Tests for no-cost LLM emulation primitives."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from inference_core.llm.config import ModelConfig, ModelProvider
from inference_core.llm.emulation import (
    EmulatedChatModel,
    EmulationSessionOverrides,
    activate_emulation_session,
    build_tool_emulation_middleware,
    create_emulated_chat_model,
)
from inference_core.llm.models import LLMModelFactory


def _emulation_settings(**overrides):
    data = {
        "llm_emulation_enabled": True,
        "llm_emulation_response": "emulated response",
        "llm_emulation_profile": "deterministic",
        "llm_emulation_latency_ms": 0,
        "llm_emulation_latency_jitter_ms": 0,
        "llm_emulation_session_scale_min": 1.0,
        "llm_emulation_session_scale_max": 1.0,
        "llm_emulation_step_latency_growth": 0.0,
        "llm_emulation_stream_first_chunk_ratio": 1.0,
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


def test_emulated_chat_model_respects_system_prompt_override_marker():
    model = EmulatedChatModel(model_name="unit-model", response="fallback response")

    message = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a test responder. Regardless of the user's message, "
                    "reply with exactly this single token and nothing else: MARKER-123"
                )
            ),
            HumanMessage(content="Say hi"),
        ]
    )

    assert message.content == "MARKER-123"


def test_emulated_chat_model_appends_suffix_rule_from_system_prompt():
    model = EmulatedChatModel(model_name="unit-model", response="fallback response")

    message = model.invoke(
        [
            SystemMessage(
                content=(
                    "ADDITIONAL STRICT RULE: end every reply with the literal token "
                    "SUFFIX-321"
                )
            ),
            HumanMessage(content="Reply with a short greeting."),
        ]
    )

    assert "SUFFIX-321" in message.content


def test_emulated_chat_model_recalls_number_from_prior_user_message():
    model = EmulatedChatModel(model_name="unit-model", response="fallback response")

    message = model.invoke(
        [
            HumanMessage(
                content="Remember this number: 42. Just confirm you noted it."
            ),
            HumanMessage(content="What number did I ask you to remember?"),
        ]
    )

    assert "42" in message.content


@patch("inference_core.core.config.get_settings")
def test_emulated_chat_model_latency_plan_is_deterministic_per_session_seed(
    mock_get_settings,
):
    mock_get_settings.return_value = _emulation_settings(
        llm_emulation_latency_ms=1000,
        llm_emulation_latency_jitter_ms=200,
        llm_emulation_session_scale_min=0.75,
        llm_emulation_session_scale_max=1.25,
        llm_emulation_step_latency_growth=0.2,
    )
    model = create_emulated_chat_model("timed-model")

    with patch("inference_core.llm.emulation.time.sleep") as mock_sleep:
        with activate_emulation_session(seed=17):
            model.invoke([HumanMessage(content="first")])
            model.invoke([HumanMessage(content="second")])
        first_run_delays = [call.args[0] for call in mock_sleep.call_args_list]

        mock_sleep.reset_mock()

        with activate_emulation_session(seed=17):
            model.invoke([HumanMessage(content="first")])
            model.invoke([HumanMessage(content="second")])
        second_run_delays = [call.args[0] for call in mock_sleep.call_args_list]

    assert len(first_run_delays) == 2
    assert first_run_delays == second_run_delays
    assert first_run_delays[0] != first_run_delays[1]


@patch("inference_core.core.config.get_settings")
def test_emulated_stream_spreads_latency_across_chunks(mock_get_settings):
    mock_get_settings.return_value = _emulation_settings(
        llm_emulation_latency_ms=900,
        llm_emulation_stream_first_chunk_ratio=0.5,
    )
    model = create_emulated_chat_model("stream-model", response="hello from tests")

    with patch("inference_core.llm.emulation.time.sleep") as mock_sleep:
        with activate_emulation_session(seed=3):
            chunks = list(model.stream([HumanMessage(content="ping")]))

    sleep_durations = [call.args[0] for call in mock_sleep.call_args_list]

    assert "".join(chunk.content for chunk in chunks) == "hello from tests"
    assert sleep_durations == [0.45, 0.225, 0.225]


def test_emulated_session_overrides_change_delay_without_rebuilding_model():
    model = EmulatedChatModel(
        model_name="override-model",
        response="override response",
        latency_ms=100,
        latency_jitter_ms=0,
        session_scale_min=1.0,
        session_scale_max=1.0,
        step_latency_growth=0.0,
    )
    overrides = EmulationSessionOverrides(
        profile="performance-realistic",
        latency_ms=2000,
        session_scale_min=2.0,
        session_scale_max=2.0,
        step_latency_growth=0.5,
    )

    with patch("inference_core.llm.emulation.time.sleep") as mock_sleep:
        with activate_emulation_session(seed=11, overrides=overrides):
            first_message = model.invoke([HumanMessage(content="first")])
            second_message = model.invoke([HumanMessage(content="second")])

    sleep_durations = [call.args[0] for call in mock_sleep.call_args_list]

    assert sleep_durations == [4.0, 6.0]
    assert first_message.response_metadata["profile"] == "performance-realistic"
    assert second_message.response_metadata["emulation_call_index"] == 2


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
    mock_get_settings.return_value = _emulation_settings(llm_tool_emulation_mode="all")

    middleware = build_tool_emulation_middleware([])

    assert middleware is not None
    assert isinstance(middleware.model, EmulatedChatModel)
    assert middleware.model.model_name == "tool-emulator"

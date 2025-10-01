from typing import Any, Dict

from inference_core.llm.callbacks import LLMUsageCallbackHandler
from inference_core.llm.usage_logging import UsageLoggingConfig, UsageSession


class DummyAIMessage:
    def __init__(self, usage_metadata, model_name):
        self.usage_metadata = usage_metadata
        self.response_metadata = {"model_name": model_name}


class DummyChatGeneration:
    def __init__(self, message):
        self.message = message


class DummyLLMResult:
    def __init__(
        self, usage: Dict[str, Any], usage_meta=None, model_name="dummy-model"
    ):
        self.llm_output = {"token_usage": usage}
        if usage_meta is not None:
            ai_msg = DummyAIMessage(usage_meta, model_name)
            gen = DummyChatGeneration(ai_msg)
            self.generations = [[gen]]
        else:
            self.generations = [[]]


def make_session():
    return UsageSession(
        task_type="unit-test",
        request_mode="sync",
        model_name="dummy-model",
        provider="dummy",
        pricing_config=None,
        logging_config=UsageLoggingConfig(enabled=False),  # disable DB commit
    )


def test_usage_callback_accumulates_tokens_llm_output_only():
    session = make_session()
    handler = LLMUsageCallbackHandler(usage_session=session)
    result = DummyLLMResult(
        {"prompt_tokens": 10, "completion_tokens": 5, "input_tokens": 10}
    )
    handler.on_llm_end(result)
    assert session.accumulated_usage.get("prompt_tokens") == 10
    assert session.accumulated_usage.get("completion_tokens") == 5
    assert session.accumulated_usage.get("input_tokens") == 10


def test_usage_callback_handles_missing_output():
    session = make_session()
    handler = LLMUsageCallbackHandler(usage_session=session)

    class NoOutput:
        llm_output = None

    handler.on_llm_end(NoOutput())
    assert session.accumulated_usage == {}

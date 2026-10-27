"""``context_management`` and ``prompt_cache`` of a Claude model entry.

Both are off by default. ``context_management`` has the API clear old tool
results from what the model reads, while the conversation sent stays as it
was (a thinking block bound to it stays valid); ``prompt_cache`` has every
request cache its prompt. Checked on the request the real ChatAnthropic
sends, through a mocked transport.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from inference_core.llm.config import ModelConfig, ModelProvider
from inference_core.llm.models import CONTEXT_MANAGEMENT_BETA, LLMModelFactory

from tests.unit.llm.test_anthropic_thinking_blocks import (
    OMITTED_STREAM,
    _build_model,
    _header_transport,
)

CLEAR_TOOL_USES = {
    "edits": [
        {
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": 15000},
            "keep": {"type": "tool_uses", "value": 3},
            "clear_at_least": {"type": "input_tokens", "value": 5000},
            "clear_tool_inputs": True,
            "exclude_tools": ["write_todos"],
        }
    ]
}


@pytest.fixture(autouse=True)
def _disable_llm_emulation(monkeypatch):
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled", lambda: False
    )


def _config(**fields: Any) -> ModelConfig:
    return ModelConfig(
        name="claude-haiku-5-5",
        provider=ModelProvider.CLAUDE,
        api_key="test-key",
        max_tokens=256,
        **fields,
    )


def _history():
    return [
        HumanMessage("Check the weather in Krakow."),
        AIMessage(
            content=[{"type": "text", "text": "Checking."}],
            tool_calls=[
                {"name": "weather", "args": {"city": "Krakow"}, "id": "toolu_01"}
            ],
        ),
        ToolMessage("Rain, 12 C", tool_call_id="toolu_01"),
    ]


async def _send(config: ModelConfig, messages=None):
    sent: list = []
    model = _build_model(config, _header_transport(sent, OMITTED_STREAM))
    async for _ in model.astream(messages or "2+2?"):
        pass
    payload, beta = sent[-1]
    return payload, (beta or "").split(",")


class TestContextManagement:
    async def test_the_edits_and_the_beta_reach_the_request(self):
        payload, betas = await _send(_config(context_management=CLEAR_TOOL_USES))

        assert payload["context_management"] == CLEAR_TOOL_USES
        assert CONTEXT_MANAGEMENT_BETA in betas

    async def test_the_history_is_sent_as_it_is(self):
        """The edits are the API's: the tool result goes out unchanged."""
        payload, _ = await _send(
            _config(context_management=CLEAR_TOOL_USES), _history()
        )

        tool_result = payload["messages"][-1]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert "Rain, 12 C" in str(tool_result["content"])

    async def test_unset_sends_nothing(self):
        payload, betas = await _send(_config())

        assert "context_management" not in payload
        assert CONTEXT_MANAGEMENT_BETA not in betas

    async def test_it_goes_with_thinking_block_binding(self):
        payload, betas = await _send(
            _config(
                context_management=CLEAR_TOOL_USES,
                thinking_block_binding="drop_block",
            )
        )

        assert payload["context_management"] == CLEAR_TOOL_USES
        assert payload["thinking"]["block_binding"] == {
            "prefix_mismatch_behavior": "drop_block"
        }
        assert CONTEXT_MANAGEMENT_BETA in betas
        assert "thinking-binding-controls-2026-08-01" in betas

    async def test_the_model_object_carries_it(self):
        """What the application asks to tell whether the API edits the
        context of a model's requests."""
        model = _build_model(
            _config(context_management=CLEAR_TOOL_USES),
            _header_transport([], OMITTED_STREAM),
        )

        assert model.context_management == CLEAR_TOOL_USES


class TestPromptCache:
    async def test_every_request_carries_cache_control(self):
        payload, _ = await _send(_config(prompt_cache="5m"))

        assert payload["cache_control"] == {"type": "ephemeral", "ttl": "5m"}

    async def test_an_hour(self):
        payload, _ = await _send(_config(prompt_cache="1h"))

        assert payload["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    async def test_unset_sends_nothing(self):
        payload, _ = await _send(_config())

        assert "cache_control" not in payload

    def test_a_lifetime_the_api_does_not_take_is_refused(self):
        with pytest.raises(ValueError):
            _config(prompt_cache="10m")


def test_another_provider_ignores_them(caplog, monkeypatch):
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled", lambda: False
    )
    config = ModelConfig(
        name="mistral-small-latest",
        provider=ModelProvider.MISTRAL,
        api_key="test-key",
        context_management=CLEAR_TOOL_USES,
        prompt_cache="5m",
    )

    class _Config:
        enable_caching = False

        def get_model_config(self, name):
            return config

    with caplog.at_level("WARNING"):
        model = LLMModelFactory(_Config()).create_model(config.name)

    assert model is not None
    assert "context_management" not in model.model_kwargs
    assert "cache_control" not in model.model_kwargs
    assert "context_management is ignored" in caplog.text
    assert "prompt_cache is ignored" in caplog.text

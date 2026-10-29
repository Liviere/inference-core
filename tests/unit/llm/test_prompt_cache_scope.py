"""A model built or used without the prompt cache of its entry.

A Claude entry's ``prompt_cache`` puts ``cache_control`` on every request of
the model. A side job whose prompt is different every time (an analysis of
the run, a summary) writes that cache and never reads it, and a write costs
more than plain input. ``create_model(prompt_cache=False)``,
``without_prompt_cache`` and ``MemoryMiddleware(analysis_prompt_cache=False)``
leave it out for such a call; without them nothing changes. Checked on the
request the real ChatAnthropic sends, through a mocked transport.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import tool

from inference_core.agents.middleware.memory import MemoryMiddleware
from inference_core.llm.config import ModelConfig, ModelProvider
from inference_core.llm.model_identity import get_model_identity
from inference_core.llm.models import LLMModelFactory, without_prompt_cache

from tests.unit.llm.test_anthropic_thinking_blocks import (
    OMITTED_STREAM,
    _build_model,
    _header_transport,
)


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


def _thinking_config(**fields: Any) -> ModelConfig:
    return _config(
        reasoning_config={"thinking": {"type": "adaptive", "display": "summarized"}},
        thinking_block_binding="drop_block",
        **fields,
    )


async def _payload(model) -> dict:
    sent: list = []
    _point_at(model, sent)
    async for _ in model.astream("2+2?"):
        pass
    return sent[-1][0]


def _point_at(model, sent: list) -> None:
    probe = _build_model(_config(), _header_transport(sent, OMITTED_STREAM))
    model.__dict__["_client"] = probe.__dict__["_client"]
    model.__dict__["_async_client"] = probe.__dict__["_async_client"]


class TestCreateModel:
    async def test_false_builds_the_model_without_the_cache(self):
        sent: list = []
        model = _build_model(
            _config(prompt_cache="1h"),
            _header_transport(sent, OMITTED_STREAM),
            prompt_cache=False,
        )
        async for _ in model.astream("2+2?"):
            pass

        assert "cache_control" not in sent[-1][0]

    async def test_false_with_thinking_keeps_the_thinking(self):
        sent: list = []
        model = _build_model(
            _thinking_config(prompt_cache="1h"),
            _header_transport(sent, OMITTED_STREAM),
            prompt_cache=False,
            reasoning_output=True,
        )
        async for _ in model.astream("2+2?"):
            pass

        payload = sent[-1][0]
        assert "cache_control" not in payload
        assert payload["thinking"]["type"] == "adaptive"

    async def test_a_lifetime_overrides_the_entrys(self):
        sent: list = []
        model = _build_model(
            _config(prompt_cache="5m"),
            _header_transport(sent, OMITTED_STREAM),
            prompt_cache="1h",
        )
        async for _ in model.astream("2+2?"):
            pass

        assert sent[-1][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    @pytest.mark.parametrize("given", [{}, {"prompt_cache": None}])
    async def test_left_out_the_entry_decides(self, given):
        sent: list = []
        model = _build_model(
            _config(prompt_cache="1h"),
            _header_transport(sent, OMITTED_STREAM),
            **given,
        )
        async for _ in model.astream("2+2?"):
            pass

        assert sent[-1][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_a_value_it_does_not_take_builds_nothing(self):
        llm_config = MagicMock()
        llm_config.enable_caching = False
        llm_config.get_model_config.return_value = _config(prompt_cache="1h")

        model = LLMModelFactory(llm_config).create_model(
            "claude-haiku-5-5", prompt_cache="10m"
        )

        assert model is None

    def test_with_and_without_the_cache_are_two_models(self):
        llm_config = MagicMock()
        llm_config.enable_caching = True
        llm_config.get_model_config.return_value = _config(prompt_cache="1h")
        factory = LLMModelFactory(llm_config)

        cached = factory.create_model("claude-haiku-5-5")
        plain = factory.create_model("claude-haiku-5-5", prompt_cache=False)

        assert cached is not plain
        assert "cache_control" in cached.model_kwargs
        assert "cache_control" not in plain.model_kwargs
        assert factory.create_model("claude-haiku-5-5", prompt_cache=False) is plain

    def test_the_entry_itself_is_left_as_it_was(self):
        entry = _config(prompt_cache="1h")
        llm_config = MagicMock()
        llm_config.enable_caching = False
        llm_config.get_model_config.return_value = entry

        LLMModelFactory(llm_config).create_model("claude-haiku-5-5", prompt_cache=False)

        assert entry.prompt_cache == "1h"

    def test_false_on_another_provider_is_no_warning(self, caplog):
        entry = ModelConfig(
            name="mistral-small-latest",
            provider=ModelProvider.MISTRAL,
            api_key="test-key",
        )
        llm_config = MagicMock()
        llm_config.enable_caching = False
        llm_config.get_model_config.return_value = entry

        with caplog.at_level("WARNING"):
            model = LLMModelFactory(llm_config).create_model(
                entry.name, prompt_cache=False
            )

        assert model is not None
        assert "prompt_cache is ignored" not in caplog.text


class TestWithoutPromptCache:
    async def test_a_copy_sends_no_cache_control(self):
        model = _build_model(
            _config(prompt_cache="1h"), _header_transport([], OMITTED_STREAM)
        )

        copy = without_prompt_cache(model)

        assert copy is not model
        assert "cache_control" not in await _payload(copy)
        # The model it was copied from keeps its cache.
        assert "cache_control" in await _payload(model)

    def test_the_copy_says_which_model_it_is(self):
        model = _build_model(
            _config(prompt_cache="1h"), _header_transport([], OMITTED_STREAM)
        )

        copy = without_prompt_cache(model)

        assert get_model_identity(copy) == get_model_identity(model)
        assert get_model_identity(copy).name == "claude-haiku-5-5"

    def test_a_model_without_the_cache_is_returned_as_it_is(self):
        model = _build_model(_config(), _header_transport([], OMITTED_STREAM))

        assert without_prompt_cache(model) is model

    def test_something_that_is_no_model_is_returned_as_it_is(self):
        thing = object()

        assert without_prompt_cache(thing) is thing


@tool
def save_note(text: str) -> str:
    """Save a note."""
    return "saved"


class TestMemoryAnalysis:
    async def _first_request(self, **middleware_kwargs) -> dict:
        sent: list = []
        model = _build_model(
            _thinking_config(prompt_cache="1h"),
            _header_transport(sent, OMITTED_STREAM),
            reasoning_output=True,
        )
        middleware = MemoryMiddleware(memory_service=MagicMock(), **middleware_kwargs)

        answered = await middleware._first_answer(
            model, [save_note], [("user", "Remember that I like tea.")]
        )

        assert answered is not None
        return sent[-1][0]

    async def test_without_the_cache(self):
        payload = await self._first_request(analysis_prompt_cache=False)

        assert "cache_control" not in payload
        assert payload["thinking"]["type"] == "adaptive"

    async def test_by_default_the_model_is_used_as_it_is(self):
        payload = await self._first_request()

        assert payload["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_a_subagents_copy_keeps_the_choice(self):
        middleware = MemoryMiddleware(
            memory_service=MagicMock(), analysis_prompt_cache=False
        )

        assert middleware.copy_for_subagent().analysis_prompt_cache is False

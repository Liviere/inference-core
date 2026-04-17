"""ChatFireworks subclass with full reasoning support.

The upstream ``langchain-fireworks`` adapter drops ``reasoning_content``
from both non-streaming and streaming responses.  This module provides
:class:`ChatFireworksReasoning` which plugs that gap by:

* Extracting ``reasoning_content`` from API responses into
  ``AIMessage.additional_kwargs["reasoning_content"]``.
* Extracting ``delta.reasoning_content`` from streaming chunks into
  ``AIMessageChunk.additional_kwargs["reasoning_content"]``.
* Serialising ``reasoning_content`` back into request dicts so that
  interleaved / preserved thinking works in multi-turn conversations.
* Exposing ``reasoning_effort``, ``thinking`` and ``reasoning_history``
  as first-class constructor parameters.

``langchain-core`` already converts
``additional_kwargs["reasoning_content"]`` into normalised
``content_blocks`` (type ``"reasoning"``), so the downstream streaming
infrastructure works without changes.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_fireworks.chat_models import (
    ChatFireworks,
    _convert_chunk_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reasoning-aware conversion helpers
# ---------------------------------------------------------------------------


def _convert_dict_to_message_with_reasoning(
    _dict: Mapping[str, Any],
) -> BaseMessage:
    """Like the upstream converter but preserves ``reasoning_content``."""
    message = _convert_dict_to_message(_dict)
    if isinstance(message, AIMessage):
        reasoning_content = _dict.get("reasoning_content")
        if reasoning_content:
            message.additional_kwargs["reasoning_content"] = reasoning_content
    return message


def _convert_chunk_to_message_chunk_with_reasoning(
    chunk: Mapping[str, Any],
    default_class: type[BaseMessageChunk],
) -> BaseMessageChunk:
    """Like the upstream converter but preserves ``delta.reasoning_content``."""
    message_chunk = _convert_chunk_to_message_chunk(chunk, default_class)
    if isinstance(message_chunk, AIMessageChunk):
        delta = chunk["choices"][0]["delta"]
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content:
            message_chunk.additional_kwargs["reasoning_content"] = reasoning_content
    return message_chunk


def _convert_message_to_dict_with_reasoning(message: BaseMessage) -> dict:
    """Like the upstream converter but includes ``reasoning_content`` in the dict.

    This is required for interleaved / preserved thinking: the Fireworks
    API expects ``reasoning_content`` to be sent back alongside ``content``
    and ``tool_calls`` in subsequent turns.
    """
    message_dict = _convert_message_to_dict(message)
    if isinstance(message, AIMessage):
        reasoning_content = message.additional_kwargs.get("reasoning_content")
        if reasoning_content:
            message_dict["reasoning_content"] = reasoning_content
    return message_dict


# ---------------------------------------------------------------------------
# ChatFireworksReasoning
# ---------------------------------------------------------------------------


class ChatFireworksReasoning(ChatFireworks):
    """``ChatFireworks`` with native reasoning support.

    Drop-in replacement for :class:`ChatFireworks` that correctly
    extracts ``reasoning_content`` from API responses (both streaming
    and non-streaming) and serialises it back for multi-turn
    conversations.

    Extra constructor parameters
    ----------------------------
    reasoning_effort : str | None
        Controls reasoning depth (``"low"``, ``"medium"``, ``"high"``).
    thinking : dict | None
        Anthropic-compatible thinking control
        (e.g. ``{"type": "enabled", "budget_tokens": 4096}``).
        Mutually exclusive with ``reasoning_effort``.
    reasoning_history : str | None
        How to handle historical reasoning content in subsequent
        requests (e.g. ``"preserved"``).
    """

    reasoning_effort: str | None = None
    """Controls reasoning depth: ``"low"``, ``"medium"``, or ``"high"``."""

    thinking: dict | None = None
    """Anthropic-compatible thinking parameter.

    Example: ``{"type": "enabled", "budget_tokens": 4096}``.
    Cannot be combined with ``reasoning_effort``.
    """

    reasoning_history: str | None = None
    """How historical reasoning content is included in subsequent requests.

    Example: ``"preserved"`` to retain all previous reasoning content.
    """

    # -- params ---------------------------------------------------------------

    @property
    def _default_params(self) -> dict[str, Any]:
        params = super()._default_params
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.thinking is not None:
            params["thinking"] = self.thinking
        if self.reasoning_history is not None:
            params["reasoning_history"] = self.reasoning_history
        return params

    # -- message dict creation ------------------------------------------------

    def _create_message_dicts(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict_with_reasoning(m) for m in messages]
        return message_dicts, params

    # -- non-streaming result -------------------------------------------------

    def _create_chat_result(self, response: dict | BaseModel) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.model_dump()
        token_usage = response.get("usage", {})
        for res in response["choices"]:
            message = _convert_dict_to_message_with_reasoning(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = {
                    "input_tokens": token_usage.get("prompt_tokens", 0),
                    "output_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
                message.response_metadata["model_provider"] = "fireworks"
                message.response_metadata["model_name"] = self.model_name
            generation_info = {"finish_reason": res.get("finish_reason")}
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        llm_output = {
            "token_usage": token_usage,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    # -- sync streaming -------------------------------------------------------

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        for chunk in self.client.create(messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk_with_reasoning(
                chunk, default_chunk_class
            )
            generation_info: dict[str, Any] = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                generation_info["model_name"] = self.model_name
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk,
                generation_info=generation_info or None,
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=logprobs,
                )
            yield generation_chunk

    # -- async streaming ------------------------------------------------------

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        async for chunk in self.async_client.acreate(messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            message_chunk = _convert_chunk_to_message_chunk_with_reasoning(
                chunk, default_chunk_class
            )
            generation_info: dict[str, Any] = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                generation_info["model_name"] = self.model_name
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs
            default_chunk_class = message_chunk.__class__
            generation_chunk = ChatGenerationChunk(
                message=message_chunk,
                generation_info=generation_info or None,
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    token=generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=logprobs,
                )
            yield generation_chunk

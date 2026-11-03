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
    _usage_to_metadata,
)
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reasoning-aware conversion helpers
# ---------------------------------------------------------------------------


def _normalize_content_blocks(content: Any) -> Any:
    """Convert LangChain-internal content blocks to Fireworks/OpenAI API format.

    LangChain stores images as ``{"type": "image", "base64": "...",
    "mime_type": "image/jpeg"}`` internally, but the Fireworks (and OpenAI)
    Chat Completions API expects
    ``{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}``.

    Plain strings and already-conformant blocks pass through unchanged.
    """
    if not isinstance(content, list):
        return content
    normalized: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            normalized.append(block)
            continue
        btype = block.get("type", "")
        if btype == "image" and "base64" in block:
            mime = block.get("mime_type", "image/png")
            b64 = block["base64"]
            normalized.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
        elif btype == "image" and "url" in block:
            normalized.append({"type": "image_url", "image_url": {"url": block["url"]}})
        else:
            normalized.append(block)
    return normalized


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

    Also normalises LangChain-internal image blocks (``type: "image"``)
    to the OpenAI/Fireworks ``image_url`` format expected by the API.
    """
    message_dict = _convert_message_to_dict(message)
    # Normalise multimodal content blocks
    if isinstance(message_dict.get("content"), list):
        message_dict["content"] = _normalize_content_blocks(message_dict["content"])
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

    include_stream_usage: bool = False
    """Whether a streamed request asks for the provider's usage and reads it.

    With it, a streamed request carries ``stream_options.include_usage`` (unless
    the call names its own ``stream_options``), and the final chunk, which has
    no ``choices`` and only ``usage``, becomes an ``AIMessageChunk`` with
    ``usage_metadata``. A response that is not streamed reports its usage the
    same way: reasoning tokens, and cache reads when :attr:`usage_cache_reads`
    is set. Without it, usage is read as before: not at all from a stream (the
    usage chunk is skipped and ``stream_usage`` is not honoured), and only
    input, output and total tokens from a response that is not streamed.
    """

    usage_cache_reads: bool = False
    """Whether the usage read under :attr:`include_stream_usage` reports the
    prompt cache's reads (``input_token_details.cache_read``). For a model whose
    cache reads have a price; elsewhere the reported reads are an unpriced key
    and would mark the cost estimated."""

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

    # -- usage ----------------------------------------------------------------

    def _provider_usage(self, usage: Mapping[str, Any]) -> UsageMetadata:
        """The provider's ``usage`` as usage metadata, under
        :attr:`include_stream_usage`: reasoning tokens always, cache reads only
        with :attr:`usage_cache_reads`."""
        usage_metadata = _usage_to_metadata(usage)
        if not self.usage_cache_reads:
            usage_metadata.pop("input_token_details", None)
        return usage_metadata

    # -- non-streaming result -------------------------------------------------

    def _create_chat_result(self, response: dict | BaseModel) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.model_dump()
        token_usage = response.get("usage", {})
        for res in response["choices"]:
            message = _convert_dict_to_message_with_reasoning(res["message"])
            if token_usage and isinstance(message, AIMessage):
                if self.include_stream_usage:
                    message.usage_metadata = self._provider_usage(token_usage)
                else:
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

    # -- streaming helpers ----------------------------------------------------

    def _stream_params(
        self, params: dict[str, Any], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """The body of a streamed request: usage asked for only under
        :attr:`include_stream_usage`, and never over the call's own
        ``stream_options``.

        The Fireworks SDK's ``create`` takes no ``stream_options`` argument,
        so it goes in ``extra_body``, which lands in the JSON body; a value
        the call put there itself wins.
        """
        params = {**params, **kwargs, "stream": True}
        if not self.include_stream_usage:
            return params
        extra_body = dict(params.pop("extra_body", None) or {})
        stream_options = params.pop("stream_options", None)
        if "stream_options" not in extra_body:
            extra_body["stream_options"] = (
                stream_options
                if stream_options is not None
                else {"include_usage": True}
            )
        params["extra_body"] = extra_body
        return params

    def _usage_chunk(self, chunk: Mapping[str, Any]) -> ChatGenerationChunk | None:
        """The final chunk of a stream that asked for usage: no ``choices``,
        only ``usage``. ``None`` for any other chunk without choices, or
        without :attr:`include_stream_usage` (such a chunk was always
        skipped)."""
        usage = chunk.get("usage")
        if not self.include_stream_usage or not usage:
            return None
        return ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                usage_metadata=self._provider_usage(usage),
                response_metadata={"model_provider": "fireworks"},
            )
        )

    # -- sync streaming -------------------------------------------------------

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = self._stream_params(params, kwargs)

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        for chunk in self.client.create(messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if len(chunk["choices"]) == 0:
                usage_chunk = self._usage_chunk(chunk)
                if usage_chunk is None:
                    continue
                if run_manager:
                    run_manager.on_llm_new_token(
                        usage_chunk.text, chunk=usage_chunk
                    )
                yield usage_chunk
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
        params = self._stream_params(params, kwargs)

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        async for chunk in self.async_client.acreate(messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if len(chunk["choices"]) == 0:
                usage_chunk = self._usage_chunk(chunk)
                if usage_chunk is None:
                    continue
                if run_manager:
                    await run_manager.on_llm_new_token(
                        token=usage_chunk.text, chunk=usage_chunk
                    )
                yield usage_chunk
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

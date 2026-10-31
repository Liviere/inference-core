"""ChatMistralAI that reports what a response read from the prompt cache.

Mistral caches the start of a prompt by itself and bills what it read at a
tenth of the input price. It says how much in
``usage.prompt_tokens_details.cached_tokens``, also in a stream's last chunk,
and ``prompt_tokens`` includes those tokens. ``langchain-mistralai`` keeps only
the three totals in ``usage_metadata``, so a price for cached input never
applied.

:class:`ChatMistralAICachedTokens` adds ``input_token_details.cache_read``, the
standard LangChain key, which usage logging records as ``cache_read_tokens``.
The factory builds it only for a model entry that prices
``extras.cache_read_tokens``; any other Mistral model stays a plain
``ChatMistralAI``, whose cost a reported but unpriced key would mark estimated.
"""

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages import BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.chat_models import (
    _convert_chunk_to_message_chunk,
    acompletion_with_retry,
)


def _cached_tokens(usage: Optional[Dict[str, Any]]) -> int:
    details = (usage or {}).get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    return cached if isinstance(cached, int) else 0


def _with_cache_read(message: Any, usage: Optional[Dict[str, Any]]) -> None:
    """Add the cached tokens of *usage* to the message's ``usage_metadata``."""
    cached = _cached_tokens(usage)
    if cached <= 0 or not getattr(message, "usage_metadata", None):
        return
    message.usage_metadata = {
        **message.usage_metadata,
        "input_token_details": {"cache_read": cached},
    }


class ChatMistralAICachedTokens(ChatMistralAI):
    """``ChatMistralAI`` with ``input_token_details.cache_read`` in its usage.

    The stream's two loops follow ``ChatMistralAI._stream`` and ``_astream``:
    the raw usage is gone once ``_convert_chunk_to_message_chunk`` returns.
    ``tests/unit/llm/test_mistral_cached_tokens.py`` runs them on the library's
    real client, so a change there fails the test.
    """

    def _create_chat_result(self, response: dict) -> ChatResult:
        result = super()._create_chat_result(response)
        for generation in result.generations:
            if isinstance(generation.message, AIMessage):
                _with_cache_read(generation.message, response.get("usage"))
        return result

    def _chunk(
        self,
        chunk: dict,
        default_class: type[BaseMessageChunk],
        index: int,
        index_type: str,
    ) -> tuple[ChatGenerationChunk, type[BaseMessageChunk], int, str]:
        message, index, index_type = _convert_chunk_to_message_chunk(
            chunk, default_class, index, index_type, self.output_version
        )
        if isinstance(message, AIMessageChunk):
            _with_cache_read(message, chunk.get("usage"))
        return (
            ChatGenerationChunk(message=message),
            message.__class__,
            index,
            index_type,
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_class: type[BaseMessageChunk] = AIMessageChunk
        index, index_type = -1, ""
        for chunk in self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            if len(chunk.get("choices", [])) == 0:
                continue
            generation, default_class, index, index_type = self._chunk(
                chunk, default_class, index, index_type
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    token=cast(str, generation.message.content), chunk=generation
                )
            yield generation

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_class: type[BaseMessageChunk] = AIMessageChunk
        index, index_type = -1, ""
        async for chunk in await acompletion_with_retry(
            self, messages=message_dicts, run_manager=run_manager, **params
        ):
            if len(chunk.get("choices", [])) == 0:
                continue
            generation, default_class, index, index_type = self._chunk(
                chunk, default_class, index, index_type
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    token=cast(str, generation.message.content), chunk=generation
                )
            yield generation

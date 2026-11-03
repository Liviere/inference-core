"""Reading a response only up to its output limit, for a provider that ignores it.

A provider stops a response at ``max_tokens`` (``max_completion_tokens``) and
bills what it generated. A gateway that does not apply the limit lets a
response run on: a call meant to cost a few hundred tokens of output costs
thousands, mostly reasoning. A model entry's ``enforces_max_tokens: false``
builds its model with :class:`OutputLengthGuard`, which reads the stream only
up to the call's limit:

- **Counting:** each raw chunk of the stream adds the tokens of the text it
  carries: the answer, the reasoning (``reasoning_content`` or ``reasoning``,
  which ``ChatOpenAI`` itself does not read) and tool call arguments, counted
  with tiktoken's ``cl100k_base`` (four characters a token without it). A chunk
  whose usage names ``completion_tokens`` gives the provider's own count. The
  count lives in a per-call ``ContextVar``, so a model shared by concurrent
  calls counts each apart.
- **The cut:** once the count reaches the limit the stream is closed, which
  closes the HTTP response, and the response ends with one more chunk:
  ``finish_reason: length`` and a usage of what was received (output: the
  count; input: the prompt counted the same way), marked
  ``response_metadata.usage_estimated``. What a provider generates after the
  connection closed is not known here.
- **Without streaming:** a call made without streaming (``streaming=False``,
  ``disable_streaming``) is streamed for this and handed back as one response.
- **A response within the limit** is read to its end; its usage is the
  provider's.

Only the Chat Completions stream is counted; a call that goes to the Responses
API is read as it comes.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, AsyncIterator, Iterator, Mapping, Optional

from langchain_core.language_models.chat_models import (
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult

logger = logging.getLogger(__name__)

#: Where the response says its usage is an estimate.
USAGE_ESTIMATED = "usage_estimated"

#: Delta fields whose text is output: the answer and the reasoning.
_TEXT_FIELDS = ("content", "reasoning_content", "reasoning")

#: Per message of a prompt, for its role and framing (as cost tracking counts).
_MESSAGE_OVERHEAD = 4


@lru_cache(maxsize=1)
def _encoding() -> Any:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_text_tokens(text: str) -> int:
    """Tokens of *text*: tiktoken's ``cl100k_base``, else four characters each."""
    if not text:
        return 0
    encoding = _encoding()
    if encoding is not None:
        return len(encoding.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)


def usage_is_estimated(message: Any) -> bool:
    """Whether *message*'s usage is the guard's estimate, not the provider's."""
    metadata = getattr(message, "response_metadata", None) or {}
    return bool(metadata.get(USAGE_ESTIMATED))


def _delta_text(delta: Mapping[str, Any]) -> str:
    parts = [delta.get(field) for field in _TEXT_FIELDS]
    for call in delta.get("tool_calls") or []:
        function = (call or {}).get("function") or {}
        parts.append(function.get("name"))
        parts.append(function.get("arguments"))
    return "".join(part for part in parts if isinstance(part, str))


@dataclass
class _Count:
    """What one call has received so far."""

    limit: int
    tokens: int = 0
    finished: bool = False

    @property
    def reached(self) -> bool:
        return not self.finished and self.tokens >= self.limit

    def add(self, chunk: Mapping[str, Any]) -> None:
        choices = chunk.get("choices") or (chunk.get("chunk") or {}).get(
            "choices"
        ) or []
        for choice in choices:
            self.tokens += count_text_tokens(_delta_text(choice.get("delta") or {}))
            if choice.get("finish_reason"):
                self.finished = True
        usage = chunk.get("usage") or (chunk.get("chunk") or {}).get("usage")
        completion = (usage or {}).get("completion_tokens")
        if isinstance(completion, int) and completion > self.tokens:
            # The provider counts as it goes (a usage on every chunk).
            self.tokens = completion
        if usage and not choices:
            # The last chunk of a stream: the provider's usage is complete.
            self.finished = True


_COUNT: ContextVar[Optional[_Count]] = ContextVar("output_length_count", default=None)


def _prompt_tokens(messages: Any) -> int:
    total = 0
    for message in messages or []:
        content = getattr(message, "content", "") or ""
        if isinstance(content, list):
            content = " ".join(
                str(block.get("text", "")) if isinstance(block, dict) else str(block)
                for block in content
            )
        total += count_text_tokens(str(content)) + _MESSAGE_OVERHEAD
    return total


class OutputLengthGuard:
    """Reads a ``ChatOpenAI`` stream only up to the call's output limit.

    A mixin, put before the model's class: :func:`guarded` makes the class.
    """

    def _output_limit(self, kwargs: Mapping[str, Any]) -> Optional[int]:
        for key in ("max_completion_tokens", "max_tokens"):
            value = kwargs.get(key)
            if isinstance(value, int) and value > 0:
                return value
        value = getattr(self, "max_tokens", None)
        return value if isinstance(value, int) and value > 0 else None

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        count = _COUNT.get()
        if count is not None:
            count.add(chunk)
        return super()._convert_chunk_to_generation_chunk(  # type: ignore[misc]
            chunk, default_chunk_class, base_generation_info
        )

    def _cut(self, messages: Any, count: _Count) -> ChatGenerationChunk:
        """The chunk that ends a response cut at its limit."""
        name = getattr(self, "model_name", None)
        logger.warning(
            "Output of model '%s' cut at %d tokens (limit %d): its provider "
            "does not stop at the limit",
            name,
            count.tokens,
            count.limit,
        )
        input_tokens = _prompt_tokens(messages)
        return ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                usage_metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": count.tokens,
                    "total_tokens": input_tokens + count.tokens,
                },
                response_metadata={
                    USAGE_ESTIMATED: True,
                    "finish_reason": "length",
                },
            ),
            generation_info={"finish_reason": "length", "model_name": name},
        )

    def _stream(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        limit = self._output_limit(kwargs)
        inner = super()._stream(  # type: ignore[misc]
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        if limit is None:
            yield from inner
            return
        count = _Count(limit)
        token = _COUNT.set(count)
        try:
            for generation_chunk in inner:
                yield generation_chunk
                if count.reached:
                    break
        finally:
            inner.close()
            _COUNT.reset(token)
        if count.reached:
            cut = self._cut(messages, count)
            if run_manager:
                run_manager.on_llm_new_token(cut.text, chunk=cut)
            yield cut

    async def _astream(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        limit = self._output_limit(kwargs)
        inner = super()._astream(  # type: ignore[misc]
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        if limit is None:
            async for generation_chunk in inner:
                yield generation_chunk
            return
        count = _Count(limit)
        token = _COUNT.set(count)
        try:
            async for generation_chunk in inner:
                yield generation_chunk
                if count.reached:
                    break
        finally:
            await inner.aclose()
            _COUNT.reset(token)
        if count.reached:
            cut = self._cut(messages, count)
            if run_manager:
                await run_manager.on_llm_new_token(cut.text, chunk=cut)
            yield cut

    def _generate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._output_limit(kwargs) is None:
            return super()._generate(  # type: ignore[misc]
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        # Streamed, so the response can be cut at its limit.
        return generate_from_stream(
            self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
        )

    async def _agenerate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._output_limit(kwargs) is None:
            return await super()._agenerate(  # type: ignore[misc]
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        return await agenerate_from_stream(
            self._astream(messages, stop=stop, run_manager=run_manager, **kwargs)
        )


_GUARDED: dict[type, type] = {}


def guarded(model_class: type) -> type:
    """*model_class* with :class:`OutputLengthGuard` (one class per base)."""
    if issubclass(model_class, OutputLengthGuard):
        return model_class
    found = _GUARDED.get(model_class)
    if found is None:
        found = type(
            f"{model_class.__name__}OutputLengthGuard",
            (OutputLengthGuard, model_class),
            {"__module__": __name__},
        )
        _GUARDED[model_class] = found
    return found

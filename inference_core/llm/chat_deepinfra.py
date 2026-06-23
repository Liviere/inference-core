"""DeepInfra chat model built on :class:`~langchain_openai.ChatOpenAI`.

DeepInfra exposes an OpenAI-compatible Chat Completions endpoint at
``https://api.deepinfra.com/v1/openai``.  Routing it through ``ChatOpenAI``
(instead of the deprecated ``langchain_community`` ``ChatDeepInfra``) gives us
the official OpenAI SDK streaming path — with retries and correct
``tool_call_chunks`` reassembly keyed by ``index`` — which fixes the intermittent
``ValueError: No generations found in stream`` the community adapter produced.

``ChatOpenAI`` leaves two DeepInfra-specific reasoning gaps, both handled here:

* **``reasoning_content`` field.**  Per its own docstring, ``ChatOpenAI`` targets
  the official OpenAI API only and does not extract the non-standard
  ``reasoning_content`` field some DeepInfra models return.  We lift it into
  ``additional_kwargs["reasoning_content"]``.
* **Inline ``<think>…</think>``.**  DeepInfra-served reasoning models (MiMo,
  R1-style) emit their thinking trace inline in ``content`` rather than in a
  ``reasoning_content`` field.  Left untouched it leaks into the chat bubble, so
  we extract it: incrementally during streaming via
  :class:`~inference_core.llm.think_tags.ThinkTagStreamRouter` (held in a
  per-call ``ContextVar`` so routing runs inside
  ``_convert_chunk_to_generation_chunk`` — i.e. *before* ``on_llm_new_token`` —
  keeping a shared/cached model instance safe under concurrency), and via a
  regex on the non-streaming path.

Both forms land in ``additional_kwargs["reasoning_content"]``; ``langchain-core``
then normalises that key into a ``reasoning`` content block (which requires a
non-``"openai"`` ``model_provider`` — see :meth:`_convert_chunk_to_generation_chunk`),
so the thinking text renders in the UI's thinking section instead of the chat
bubble.

``reasoning_effort`` needs no special handling: it is a native ``ChatOpenAI``
field forwarded on the Chat Completions path, so the YAML
``reasoning_config.reasoning_effort`` reaches the API unchanged.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Iterator
from contextvars import ContextVar
from typing import Any, Optional

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from .think_tags import OPEN_TAG, ThinkTagStreamRouter

logger = logging.getLogger(__name__)

DEEPINFRA_OPENAI_BASE_URL = "https://api.deepinfra.com/v1/openai"
"""DeepInfra's OpenAI-compatible Chat Completions base URL."""

# Provider label that makes langchain-core normalise ``reasoning_content`` into a
# ``reasoning`` content block.  Under ``ChatOpenAI``'s default ``"openai"`` that
# normalisation is suppressed (OpenAI carries reasoning differently).
_REASONING_PROVIDER = "deepinfra"

# Per-call ``<think>`` router.  Set in ``_stream``/``_astream`` so the per-chunk
# converter can route content before the callback fires; ``ContextVar`` keeps
# concurrent streams on a shared model instance isolated (thread- and task-local).
_THINK_ROUTER: ContextVar[Optional[ThinkTagStreamRouter]] = ContextVar(
    "deepinfra_think_router", default=None
)

_THINK_BLOCK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)


def _delta_reasoning_content(chunk: dict) -> Optional[str]:
    """Return ``delta.reasoning_content`` from a streamed chunk dict, if present.

    Mirrors how the upstream converter locates choices, including the
    ``beta.chat.completions.stream`` shape where they live under ``chunk``.
    """
    choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    return delta.get("reasoning_content")


def _append_reasoning(message: AIMessage, reasoning: str) -> None:
    """Append ``reasoning`` to ``additional_kwargs["reasoning_content"]``."""
    if not reasoning:
        return
    existing = message.additional_kwargs.get("reasoning_content") or ""
    message.additional_kwargs["reasoning_content"] = existing + reasoning


def _flush_think_router(router: ThinkTagStreamRouter) -> Optional[ChatGenerationChunk]:
    """Emit a final chunk from whatever the router still holds back, if any."""
    reasoning, content = router.flush()
    if not reasoning and not content:
        return None
    message = AIMessageChunk(content=content)
    message.response_metadata["model_provider"] = _REASONING_PROVIDER
    _append_reasoning(message, reasoning)
    return ChatGenerationChunk(message=message)


class ChatDeepInfraReasoning(ChatOpenAI):
    """``ChatOpenAI`` pointed at DeepInfra, with reasoning extraction.

    Drop-in DeepInfra chat model.  Everything except reasoning extraction is
    inherited from ``ChatOpenAI``; only the streaming/non-streaming seams are
    overridden to surface ``reasoning_content`` (server field + inline
    ``<think>``).
    """

    extract_think_tags: bool = True
    """Extract inline ``<think>…</think>`` blocks from ``content`` as reasoning.

    Enabled by default: DeepInfra-served reasoning models (MiMo, R1 family) emit
    their thinking inline in ``content`` rather than in a ``reasoning_content``
    field.  Set ``False`` to pass content through verbatim.
    """

    @property
    def _llm_type(self) -> str:
        return "deepinfra-openai"

    # -- streaming ------------------------------------------------------------

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: Optional[dict],
    ) -> Optional[ChatGenerationChunk]:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return None
        message = generation_chunk.message
        if isinstance(message, AIMessageChunk):
            # ChatOpenAI stamps ``model_provider="openai"``, under which
            # langchain-core does NOT normalise ``reasoning_content`` into a
            # ``reasoning`` content block.  Relabel as the true provider so the
            # block is emitted (and reasoning streams live to the UI).
            message.response_metadata["model_provider"] = _REASONING_PROVIDER
            server_reasoning = _delta_reasoning_content(chunk)
            if server_reasoning:
                _append_reasoning(message, server_reasoning)
            router = _THINK_ROUTER.get()
            if router is not None and isinstance(message.content, str):
                routed_reasoning, routed_content = router.feed(message.content)
                # Route content away (held back / moved to reasoning) while
                # preserving tool_call_chunks / usage_metadata on the chunk.
                message.content = routed_content
                _append_reasoning(message, routed_reasoning)
        return generation_chunk

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        if not self.extract_think_tags:
            yield from super()._stream(*args, **kwargs)
            return
        router = ThinkTagStreamRouter()
        token = _THINK_ROUTER.set(router)
        try:
            yield from super()._stream(*args, **kwargs)
        finally:
            _THINK_ROUTER.reset(token)
        final = _flush_think_router(router)
        if final is not None:
            yield final

    async def _astream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        if not self.extract_think_tags:
            async for generation_chunk in super()._astream(*args, **kwargs):
                yield generation_chunk
            return
        router = ThinkTagStreamRouter()
        token = _THINK_ROUTER.set(router)
        try:
            async for generation_chunk in super()._astream(*args, **kwargs):
                yield generation_chunk
        finally:
            _THINK_ROUTER.reset(token)
        final = _flush_think_router(router)
        if final is not None:
            yield final

    # -- non-streaming --------------------------------------------------------

    def _create_chat_result(
        self,
        response: Any,
        generation_info: Optional[dict] = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(result.generations, raw.get("choices") or []):
            message = generation.message
            if not isinstance(message, AIMessage):
                continue
            # Relabel the provider (see ``_convert_chunk_to_generation_chunk``)
            # so ``reasoning_content`` normalises into a ``reasoning`` block.
            message.response_metadata["model_provider"] = _REASONING_PROVIDER
            server_reasoning = (choice.get("message") or {}).get("reasoning_content")
            if server_reasoning:
                _append_reasoning(message, server_reasoning)
            if self.extract_think_tags and isinstance(message.content, str):
                self._extract_inline_think(message)
        return result

    @staticmethod
    def _extract_inline_think(message: AIMessage) -> None:
        """Move a leading inline ``<think>…</think>`` block into reasoning."""
        stripped = message.content.lstrip()
        if not stripped.startswith(OPEN_TAG):
            return
        match = _THINK_BLOCK_RE.match(message.content)
        if match:
            reasoning = match.group(1)
            content = message.content[match.end() :]
        else:  # unterminated block — same fallback as the streaming flush
            reasoning = stripped[len(OPEN_TAG) :]
            content = ""
        _append_reasoning(message, reasoning)
        message.content = content

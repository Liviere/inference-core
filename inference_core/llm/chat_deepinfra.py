"""ChatDeepInfra subclass with correct streaming tool-call parsing.

The upstream ``langchain_community`` adapter
:class:`~langchain_community.chat_models.deepinfra.ChatDeepInfra` mishandles
tool calls **on the streaming path**:

* :func:`_parse_tool_calling` runs ``json.loads`` on the ``arguments`` field of
  *every* streamed delta.  OpenAI-compatible servers (DeepInfra included) send
  tool-call ``arguments`` as incremental JSON *fragments* across many SSE
  deltas (``{"todos"`` … ``: [...]`` … ``}``).  Each fragment fails to parse,
  so ``args`` collapses to ``{}``.
* :func:`_convert_delta_to_message_chunk` then attaches fully-formed
  ``tool_calls=`` to the chunk instead of ``tool_call_chunks=``.  LangChain can
  only reassemble streamed argument fragments through ``tool_call_chunks``
  (merged by ``index`` in :meth:`AIMessageChunk.__add__`).  The net effect is
  that the accumulated tool call ends up with empty ``args``, so any tool that
  requires parameters fails with ``Field required`` and the model retries
  forever.

This module provides :class:`ChatDeepInfraReasoning`, a drop-in replacement
that:

* Emits proper ``tool_call_chunks`` (raw partial ``arguments`` strings, keyed by
  ``index``) so ``langchain-core`` reassembles them into complete tool calls.
* Lifts ``delta.reasoning_content`` into
  ``AIMessageChunk.additional_kwargs["reasoning_content"]`` when the server
  provides it.
* Extracts inline ``<think>…</think>`` reasoning from ``content`` (the format
  MiMo / R1-style models actually emit) into
  ``additional_kwargs["reasoning_content"]`` — incrementally during streaming
  via :class:`~inference_core.llm.think_tags.ThinkTagStreamRouter` and via a
  regex on the non-streaming path.  Downstream, langchain-core normalises that
  key into a ``reasoning`` content block, so the thinking text renders in the
  UI's thinking section instead of leaking into the chat bubble.
* Exposes ``reasoning_effort`` as a first-class constructor parameter and
  forwards it in the request body (the base class silently ignores unknown
  constructor kwargs, so ``reasoning_config.reasoning_effort`` from the YAML
  never reaches the API otherwise).

The non-streaming path is inherited unchanged: it already parses the *complete*
``arguments`` string correctly.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Optional

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
)
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_community.chat_models.deepinfra import (
    ChatDeepInfra,
    _parse_stream,
    _parse_stream_async,
)
from langchain_community.utilities.requests import Requests

from .think_tags import OPEN_TAG, ThinkTagStreamRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming-aware conversion helpers
# ---------------------------------------------------------------------------


def _convert_delta_to_message_chunk_fixed(
    _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Convert a streaming delta into a message chunk.

    Unlike the upstream helper this emits ``tool_call_chunks`` carrying the raw
    (possibly partial) ``arguments`` string keyed by ``index`` so that
    ``AIMessageChunk`` accumulation can reassemble fragmented tool-call
    arguments.  It never calls ``json.loads`` on a partial fragment.
    """
    role = _dict.get("role")
    content = _dict.get("content") or ""
    raw_tool_calls = _dict.get("tool_calls") or []

    if role == "assistant" or default_class == AIMessageChunk:
        tool_call_chunks = []
        for raw_tc in raw_tool_calls:
            func = raw_tc.get("function") or {}
            tool_call_chunks.append(
                create_tool_call_chunk(
                    name=func.get("name"),
                    args=func.get("arguments"),
                    id=raw_tc.get("id"),
                    index=raw_tc.get("index"),
                )
            )
        chunk = AIMessageChunk(content=content, tool_call_chunks=tool_call_chunks)
        reasoning_content = _dict.get("reasoning_content")
        if reasoning_content:
            chunk.additional_kwargs["reasoning_content"] = reasoning_content
        return chunk
    elif role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    else:
        return default_class(content=content)  # type: ignore[call-arg]


def _handle_sse_line_fixed(line: str) -> Optional[BaseMessageChunk]:
    """Parse a single SSE ``data:`` line into a message chunk (fixed converter)."""
    try:
        obj = json.loads(line)
        delta = obj.get("choices", [{}])[0].get("delta", {})
        return _convert_delta_to_message_chunk_fixed(delta, AIMessageChunk)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Inline <think> reasoning extraction
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)


def _route_think_tags(
    router: ThinkTagStreamRouter, chunk: BaseMessageChunk
) -> Optional[BaseMessageChunk]:
    """Route a streamed chunk's content through the ``<think>`` tag router.

    Text recognised as reasoning moves to
    ``additional_kwargs["reasoning_content"]`` (appended after any
    server-provided ``delta.reasoning_content``); tool-call chunks pass through
    untouched regardless of router state.  Returns ``None`` when the chunk has
    nothing left to emit (its text is held back pending tag disambiguation).
    """
    if not isinstance(chunk, AIMessageChunk) or not isinstance(chunk.content, str):
        return chunk
    reasoning, content = router.feed(chunk.content)
    server_reasoning = chunk.additional_kwargs.get("reasoning_content") or ""
    reasoning = server_reasoning + reasoning
    if not content and not reasoning and not chunk.tool_call_chunks:
        return None
    routed = AIMessageChunk(content=content, tool_call_chunks=chunk.tool_call_chunks)
    for key, value in chunk.additional_kwargs.items():
        if key != "reasoning_content":
            routed.additional_kwargs[key] = value
    if reasoning:
        routed.additional_kwargs["reasoning_content"] = reasoning
    return routed


def _flush_think_tags(router: ThinkTagStreamRouter) -> Optional[AIMessageChunk]:
    """Build a final chunk from text the router still holds back, if any."""
    reasoning, content = router.flush()
    if not reasoning and not content:
        return None
    chunk = AIMessageChunk(content=content)
    if reasoning:
        chunk.additional_kwargs["reasoning_content"] = reasoning
    return chunk


# ---------------------------------------------------------------------------
# ChatDeepInfraReasoning
# ---------------------------------------------------------------------------


class ChatDeepInfraReasoning(ChatDeepInfra):
    """``ChatDeepInfra`` with correct streaming tool calls and reasoning support.

    Drop-in replacement for :class:`ChatDeepInfra`.  The non-streaming path is
    inherited verbatim; only ``_stream`` / ``_astream`` are overridden to use a
    converter that emits ``tool_call_chunks`` instead of pre-parsed
    ``tool_calls``.
    """

    reasoning_effort: Optional[str] = None
    """Controls reasoning depth: ``"none"``, ``"low"``, ``"medium"`` or ``"high"``.

    Officially supported by the DeepInfra chat-completions API
    (https://docs.deepinfra.com/chat/reasoning); per the docs it has no effect
    on non-reasoning models, so sending it is always safe.  Forwarded in the
    request body.  The base ``ChatDeepInfra`` silently ignores unknown
    constructor kwargs, so declaring it here is what makes the YAML
    ``reasoning_config.reasoning_effort`` actually reach the API.
    """

    extract_think_tags: bool = True
    """Extract inline ``<think>…</think>`` blocks from ``content`` as reasoning.

    Enabled by default: DeepInfra-served reasoning models (MiMo, R1 family)
    emit their thinking inline in ``content`` rather than in a separate
    ``reasoning_content`` field.  Set to ``False`` to pass content through
    verbatim.
    """

    @property
    def _default_params(self) -> dict[str, Any]:
        params = super()._default_params
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        return params

    # -- sync streaming -------------------------------------------------------

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        router = ThinkTagStreamRouter() if self.extract_think_tags else None

        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        for line in _parse_stream(response.iter_lines()):
            chunk = _handle_sse_line_fixed(line)
            if chunk and router is not None:
                chunk = _route_think_tags(router, chunk)
            if chunk:
                cg_chunk = ChatGenerationChunk(message=chunk, generation_info=None)
                if run_manager:
                    run_manager.on_llm_new_token(str(chunk.content), chunk=cg_chunk)
                yield cg_chunk
        if router is not None:
            final = _flush_think_tags(router)
            if final is not None:
                cg_chunk = ChatGenerationChunk(message=final, generation_info=None)
                if run_manager:
                    run_manager.on_llm_new_token(str(final.content), chunk=cg_chunk)
                yield cg_chunk

    # -- async streaming ------------------------------------------------------

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {"messages": message_dicts, "stream": True, **params, **kwargs}
        router = ThinkTagStreamRouter() if self.extract_think_tags else None

        request_timeout = params.pop("request_timeout")
        request = Requests(headers=self._headers())
        async with request.apost(
            url=self._url(), data=self._body(params), timeout=request_timeout
        ) as response:
            async for line in _parse_stream_async(response.content):
                chunk = _handle_sse_line_fixed(line)
                if chunk and router is not None:
                    chunk = _route_think_tags(router, chunk)
                if chunk:
                    cg_chunk = ChatGenerationChunk(message=chunk, generation_info=None)
                    if run_manager:
                        await run_manager.on_llm_new_token(
                            str(chunk.content), chunk=cg_chunk
                        )
                    yield cg_chunk
        if router is not None:
            final = _flush_think_tags(router)
            if final is not None:
                cg_chunk = ChatGenerationChunk(message=final, generation_info=None)
                if run_manager:
                    await run_manager.on_llm_new_token(
                        str(final.content), chunk=cg_chunk
                    )
                yield cg_chunk

    # -- non-streaming --------------------------------------------------------

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        """Inherited parsing, plus inline ``<think>`` extraction for parity."""
        result = super()._create_chat_result(response)
        if not self.extract_think_tags:
            return result
        for generation in result.generations:
            message = generation.message
            if not isinstance(message, AIMessage) or not isinstance(
                message.content, str
            ):
                continue
            stripped = message.content.lstrip()
            if not stripped.startswith(OPEN_TAG):
                continue
            match = _THINK_BLOCK_RE.match(message.content)
            if match:
                reasoning = match.group(1)
                content = message.content[match.end() :]
            else:  # unterminated block — same fallback as the streaming flush
                reasoning = stripped[len(OPEN_TAG) :]
                content = ""
            existing = message.additional_kwargs.get("reasoning_content") or ""
            message.additional_kwargs["reasoning_content"] = existing + reasoning
            message.content = content
        return result

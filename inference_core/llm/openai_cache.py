"""OpenAI requests that mark the end of their system prompt for the prompt cache.

OpenAI caches the start of a prompt by itself, but from GPT-5.6 on it puts its
one automatic breakpoint at the end of the prompt. A request reads the cache
only up to a breakpoint an earlier request wrote, so a new conversation with
the same assistant, or a conversation whose history changed (a summary,
cleared tool results), reads nothing, although its instructions and tools are
the same.

From GPT-5.6 a request may mark breakpoints of its own on content blocks
(``prompt_cache_breakpoint: {mode: explicit}``) and say how its cache is
written (``prompt_cache_options``). :class:`ChatOpenAICacheBreakpoints` marks
one: the end of the system prompt. Tool definitions come before it in the
cached prefix, so the breakpoint covers the instructions and the tools.
``prompt_cache_options`` is ``{mode: implicit, ttl: 30m}``: the automatic
breakpoint at the end stays, so a conversation that goes on reads what it
read before. A cache write costs 1.25x input (30 minutes), a read 0.1x.

Earlier models refuse these fields (400), so a model entry asks for them
(``prompt_cache_breakpoints: true``), and only an entry of such a model should.
They are put on the request the model makes, never on a message, so a
fallback of another provider never sees them.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

#: The breakpoint put on the last block of the system prompt.
BREAKPOINT = {"mode": "explicit"}

#: How the requests write their cache: the automatic breakpoint stays, with
#: the one lifetime OpenAI takes for Chat Completions.
CACHE_OPTIONS = {"mode": "implicit", "ttl": "30m"}

#: The roles of the messages a system prompt is made of.
_SYSTEM_ROLES = frozenset({"system", "developer"})


def _text_block(text: str, block_type: str) -> dict[str, Any]:
    return {"type": block_type, "text": text, "prompt_cache_breakpoint": BREAKPOINT}


def _mark(message: dict[str, Any], block_type: str) -> bool:
    """Put the breakpoint on the last text block of *message*; whether it went.

    Text content becomes one block of *block_type* (``text`` for Chat
    Completions, ``input_text`` for Responses) with the same text.
    """
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return False
        message["content"] = [_text_block(content, block_type)]
        return True
    if not isinstance(content, list):
        return False
    for index in range(len(content) - 1, -1, -1):
        block = content[index]
        if (
            isinstance(block, dict)
            and block.get("type") in {"text", "input_text"}
            and block.get("text")
        ):
            content = list(content)
            content[index] = {**block, "prompt_cache_breakpoint": BREAKPOINT}
            message["content"] = content
            return True
    return False


def _system_prompt_end(items: Any, *, responses: bool) -> Optional[dict[str, Any]]:
    """The last message of the system prompt at the start of *items*."""
    if not isinstance(items, list):
        return None
    last = None
    for item in items:
        if not isinstance(item, dict):
            break
        if responses and item.get("type", "message") != "message":
            break
        if item.get("role") not in _SYSTEM_ROLES:
            break
        last = item
    return last


def mark_system_prompt(payload: dict[str, Any]) -> bool:
    """Mark the end of the system prompt of a request's *payload* in place.

    Chat Completions carries the conversation as ``messages``, the Responses
    API as ``input``. Returns whether a breakpoint went in: a request without
    a system prompt (or with one of no text) has none.
    """
    if "messages" in payload:
        message = _system_prompt_end(payload["messages"], responses=False)
        block_type = "text"
    else:
        message = _system_prompt_end(payload.get("input"), responses=True)
        block_type = "input_text"
    if message is None:
        return False
    return _mark(message, block_type)


class ChatOpenAICacheBreakpoints(ChatOpenAI):
    """``ChatOpenAI`` whose requests mark the end of the system prompt.

    With :attr:`prompt_cache_breakpoints` set, every request (streamed or
    not, Chat Completions or Responses) gets the breakpoint and, unless the
    call names its own, ``prompt_cache_options``. With it off the model is a
    plain ``ChatOpenAI``.
    """

    prompt_cache_breakpoints: bool = True
    """Whether the requests mark the end of the system prompt."""

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if not self.prompt_cache_breakpoints:
            return payload
        if mark_system_prompt(payload) and not payload.get("prompt_cache_options"):
            payload["prompt_cache_options"] = dict(CACHE_OPTIONS)
        return payload


def without_cache_breakpoints(model: Any) -> Any:
    """*model* without its system prompt's breakpoint; as it is when it has none."""
    if not getattr(model, "prompt_cache_breakpoints", False):
        return model
    try:
        return model.model_copy(update={"prompt_cache_breakpoints": False})
    except Exception:
        logger.warning(
            "Could not copy a model without its cache breakpoints; it is used as it is",
            exc_info=True,
        )
        return model

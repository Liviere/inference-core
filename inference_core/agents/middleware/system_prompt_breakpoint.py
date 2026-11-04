"""A cache breakpoint at the end of the system prompt, next to the automatic one.

A Claude model entry's ``prompt_cache`` puts Anthropic's automatic caching on
every request: a top-level ``cache_control``, for which the API puts the
breakpoint on the prompt's last block. A request reads the cache only up to a
breakpoint an earlier request wrote, so every entry of a conversation ends
where one of its requests ended. A new conversation with the same agent has
the same tools and system prompt, but no entry ends there: its first request
reads nothing and writes the whole prompt again.

:class:`SystemPromptBreakpointMiddleware` adds an explicit breakpoint on the
last block of the system prompt, with the lifetime of the automatic one. The
tools come before the system prompt in the cached prefix, so the entry it
writes covers the tools and the instructions, and the next conversation's
first request reads them. The automatic breakpoint stays, so a conversation
that goes on reads what it read before. A breakpoint costs nothing of its
own: the tokens a request writes are counted once, however many breakpoints
they are cut into.

The breakpoint is worth having only when the system prompt is the same from
one conversation to the next. Memories placed in the system prompt
(``memory_context_placement="system_prompt"``) change its last block with
every turn; with placement ``"message"`` they come in the person's message.

The middleware sits innermost, after the fallback, so it sees the model each
attempt is made on (``AgentService(system_prompt_breakpoint=True)`` puts it
last in the agent's stack and in each of its subagents'; it is never handed
on with the parent's middleware, which a subagent puts first): only a model whose requests carry the automatic
``cache_control`` (a Claude entry with ``prompt_cache``) gets the breakpoint;
a fallback of another provider, or a model built without the cache, is sent
the request as it is. Anthropic takes at most four breakpoints in a request,
the automatic one included; a request that has them all already is left as
it is.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)

#: Anthropic's limit of breakpoints in one request, the automatic one included.
MAX_BREAKPOINTS = 4


def automatic_cache_lifetime(model: Any) -> Optional[str]:
    """The lifetime of the model's automatic prompt cache, or None.

    A Claude model built with ``prompt_cache`` carries a top-level
    ``cache_control`` in its ``model_kwargs``; any other model carries none.
    """
    model_kwargs = getattr(model, "model_kwargs", None)
    if not isinstance(model_kwargs, dict):
        return None
    control = model_kwargs.get("cache_control")
    if not isinstance(control, dict):
        return None
    return str(control.get("ttl") or "5m")


def _marks_in(content: Any) -> int:
    if not isinstance(content, list):
        return 0
    return sum(
        1 for block in content if isinstance(block, dict) and "cache_control" in block
    )


def breakpoints_of(system: Optional[SystemMessage], messages: Any, tools: Any) -> int:
    """The explicit breakpoints a request carries already."""
    count = _marks_in(system.content) if system is not None else 0
    for message in messages or ():
        count += _marks_in(getattr(message, "content", None))
    for tool in tools or ():
        if isinstance(tool, dict) and "cache_control" in tool:
            count += 1
    return count


def _is_text_block(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") == "text"
        and bool(block.get("text"))
    )


def marked_system_message(system: SystemMessage, ttl: str) -> Optional[SystemMessage]:
    """*system* with a breakpoint on its last text block, or None.

    None when there is nothing to mark (no text) or the last text block has a
    breakpoint already. Text content becomes one text block of the same text;
    the text is never changed.
    """
    content = system.content
    if isinstance(content, str):
        if not content:
            return None
        blocks: list[Any] = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        blocks = [
            {"type": "text", "text": block} if isinstance(block, str) else block
            for block in content
        ]
    else:
        return None
    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index]
        if not _is_text_block(block):
            continue
        if "cache_control" in block:
            return None
        blocks = list(blocks)
        blocks[index] = {
            **block,
            "cache_control": {"type": "ephemeral", "ttl": ttl},
        }
        return system.model_copy(update={"content": blocks})
    return None


class SystemPromptBreakpointMiddleware(AgentMiddleware):
    """Mark the end of the system prompt of each call for the prompt cache."""

    def __init__(self) -> None:
        super().__init__()
        self._warned_full = False

    def _marked(self, request: Any) -> Any:
        ttl = automatic_cache_lifetime(request.model)
        system = getattr(request, "system_message", None)
        if ttl is None or system is None:
            return request
        # One slot is the automatic breakpoint's.
        present = breakpoints_of(system, request.messages, request.tools)
        if present + 2 > MAX_BREAKPOINTS:
            if not self._warned_full:
                logger.warning(
                    "No cache breakpoint on the system prompt: the request has "
                    "%d of its own",
                    present,
                )
                self._warned_full = True
            return request
        marked = marked_system_message(system, ttl)
        if marked is None:
            return request
        return request.override(system_message=marked)

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._marked(request))

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        return await handler(self._marked(request))

"""Incremental ``<think>…</think>`` extraction for streamed model output.

Some OpenAI-compatible providers (e.g. DeepInfra serving MiMo / R1-style
models) emit the model's reasoning trace inline in ``content`` wrapped in
``<think>`` tags instead of a separate ``reasoning_content`` delta field.
Without special handling the thinking text leaks into the chat bubble as
regular assistant content.

:class:`ThinkTagStreamRouter` splits a token stream into (reasoning, content)
pairs **incrementally** using a bounded lookahead buffer: it only ever holds
back characters that could still be a prefix of an open/close tag (at most
``len("</think>")`` characters), so there is no time-based delay and the
added latency is a few tokens at the start of the stream.

State machine:

* ``detecting`` — at stream start, buffer content while it (after leading
  whitespace) is still a prefix of ``<think>``.  On a full match the tag and
  leading whitespace are dropped and the router switches to ``reasoning``;
  on divergence the whole buffer is released as content.
* ``reasoning`` — text is routed to reasoning while scanning for
  ``</think>``; the longest suffix that is a prefix of the close tag is held
  back between feeds.  After the close tag, whitespace immediately following
  it is skipped (models typically emit ``</think>\\n\\n``).
* ``content`` — pass-through.

Deliberate limitations:

* The open tag is only recognised at the **start of the stream** (after
  whitespace).  A ``<think>`` appearing mid-answer (e.g. the model quoting
  the tag) is treated as literal text — no false positives.
* The R1-style quirk of omitting the opening tag and emitting only a closing
  ``</think>`` is not handled; that would require unbounded buffering.
* An unterminated ``<think>`` block routes the remainder of the stream to
  reasoning (``flush`` returns the holdback as reasoning, content stays
  empty) — the safe fallback for truncated generations.
"""

from __future__ import annotations

OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"


def _partial_suffix_len(text: str, tag: str) -> int:
    """Length of the longest suffix of *text* that is a proper prefix of *tag*."""
    max_len = min(len(text), len(tag) - 1)
    for k in range(max_len, 0, -1):
        if text[-k:] == tag[:k]:
            return k
    return 0


class ThinkTagStreamRouter:
    """Split streamed text into (reasoning, content) across chunk boundaries."""

    def __init__(self) -> None:
        self._state = "detecting"
        self._buffer = ""  # detection buffer / reasoning holdback
        self._skip_ws_after_close = False

    def feed(self, text: str) -> tuple[str, str]:
        """Process the next stream fragment, returning ``(reasoning, content)``."""
        if not text:
            return "", ""
        if self._state == "detecting":
            return self._feed_detecting(text)
        if self._state == "reasoning":
            return self._feed_reasoning(text)
        return "", self._feed_content(text)

    def flush(self) -> tuple[str, str]:
        """Release whatever is still held back at end of stream."""
        held = self._buffer
        self._buffer = ""
        if self._state == "detecting":
            self._state = "content"
            return "", held
        if self._state == "reasoning":
            return held, ""
        return "", ""

    # -- states ---------------------------------------------------------------

    def _feed_detecting(self, text: str) -> tuple[str, str]:
        self._buffer += text
        stripped = self._buffer.lstrip()
        if stripped.startswith(OPEN_TAG):
            remainder = stripped[len(OPEN_TAG) :]
            self._buffer = ""
            self._state = "reasoning"
            return self._feed_reasoning(remainder) if remainder else ("", "")
        if len(stripped) < len(OPEN_TAG) and OPEN_TAG.startswith(stripped):
            return "", ""  # still ambiguous — keep buffering
        released = self._buffer
        self._buffer = ""
        self._state = "content"
        return "", released

    def _feed_reasoning(self, text: str) -> tuple[str, str]:
        working = self._buffer + text
        self._buffer = ""
        idx = working.find(CLOSE_TAG)
        if idx >= 0:
            reasoning = working[:idx]
            rest = working[idx + len(CLOSE_TAG) :]
            self._state = "content"
            self._skip_ws_after_close = True
            return reasoning, self._feed_content(rest)
        holdback = _partial_suffix_len(working, CLOSE_TAG)
        if holdback:
            self._buffer = working[-holdback:]
            working = working[:-holdback]
        return working, ""

    def _feed_content(self, text: str) -> str:
        if self._skip_ws_after_close:
            text = text.lstrip()
            if not text:
                return ""
            self._skip_ws_after_close = False
        return text

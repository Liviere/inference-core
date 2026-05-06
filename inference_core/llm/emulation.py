"""No-cost LLM emulation primitives for test environments."""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from collections.abc import Iterator, Sequence
from typing import Any, Callable, Optional

from langchain.agents.middleware import LLMToolEmulator
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field


@dataclass
class EmulationLatencyPlan:
    """Concrete timing plan for one emulated model invocation.

    WHY: separating latency planning from the actual sleep calls makes the
    emulation easier to test and lets sync, async, and streaming paths share
    one deterministic timing contract.
    """

    total_delay_ms: int
    call_index: int
    session_seed: int | None = None
    session_scale: float = 1.0
    jitter_ms: int = 0
    pre_chunk_delays_ms: tuple[int, ...] = ()


@dataclass
class EmulationSessionOverrides:
    """Per-session override values applied on top of model-level emulation settings.

    WHY: performance tests and targeted run probes need to steer the latency
    profile of one agent session without mutating cached model instances or
    requiring a full API restart with different environment variables.
    """

    profile: str | None = None
    latency_ms: int | None = None
    latency_jitter_ms: int | None = None
    session_scale_min: float | None = None
    session_scale_max: float | None = None
    step_latency_growth: float | None = None
    stream_first_chunk_ratio: float | None = None
    error_rate: float | None = None

    def __post_init__(self) -> None:
        """Validate override ranges eagerly when a session is activated.

        WHY: invalid perf-test headers should fail at the request boundary
        rather than producing surprising timing behavior deep in the model loop.
        """

        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")
        if self.latency_jitter_ms is not None and self.latency_jitter_ms < 0:
            raise ValueError("latency_jitter_ms must be >= 0")
        if self.session_scale_min is not None and self.session_scale_min < 0:
            raise ValueError("session_scale_min must be >= 0")
        if self.session_scale_max is not None and self.session_scale_max < 0:
            raise ValueError("session_scale_max must be >= 0")
        if self.step_latency_growth is not None and self.step_latency_growth < 0:
            raise ValueError("step_latency_growth must be >= 0")
        if self.stream_first_chunk_ratio is not None and not (
            0.0 <= self.stream_first_chunk_ratio <= 1.0
        ):
            raise ValueError("stream_first_chunk_ratio must be between 0 and 1")
        if self.error_rate is not None and not (0.0 <= self.error_rate <= 1.0):
            raise ValueError("error_rate must be between 0 and 1")


@dataclass
class _EmulationSessionState:
    """Per-run state shared by all emulated model calls in one agent session.

    WHY: multi-step agent runs should vary their timing across sessions while
    staying deterministic inside one run, even when the same cached model
    instance services many invocations.
    """

    seed: int
    rng: random.Random = field(repr=False)
    call_index: int = 0
    session_scale: float | None = None
    overrides: EmulationSessionOverrides | None = None


_ACTIVE_EMULATION_SESSION: ContextVar[_EmulationSessionState | None] = ContextVar(
    "active_emulation_session",
    default=None,
)


def _coerce_session_scale_bounds(
    minimum: float,
    maximum: float,
) -> tuple[float, float]:
    """Normalize configured session multiplier bounds.

    WHY: the planner should stay resilient even if an operator swaps the min
    and max values in env configuration.
    """

    if maximum < minimum:
        return maximum, minimum
    return minimum, maximum


def _distribute_stream_delays(
    total_delay_ms: int,
    chunk_count: int,
    first_chunk_ratio: float,
) -> tuple[int, ...]:
    """Split total stream latency into per-chunk waits.

    WHY: real provider streams do not spend all latency before the first token.
    Splitting the delay lets tests approximate time-to-first-token and the tail
    of a streamed response separately.
    """

    if chunk_count <= 0:
        return ()

    if chunk_count == 1:
        return (max(0, total_delay_ms),)

    normalized_ratio = min(max(first_chunk_ratio, 0.0), 1.0)
    first_chunk_delay_ms = int(round(max(0, total_delay_ms) * normalized_ratio))
    remaining_delay_ms = max(0, total_delay_ms - first_chunk_delay_ms)
    remaining_slots = chunk_count - 1
    per_slot_delay_ms, extra_delay_ms = divmod(remaining_delay_ms, remaining_slots)

    remaining_delays = [per_slot_delay_ms] * remaining_slots
    for index in range(extra_delay_ms):
        remaining_delays[index] += 1

    return (first_chunk_delay_ms, *remaining_delays)


def _sleep_sync_ms(delay_ms: int) -> None:
    """Sleep synchronously for the requested emulated latency.

    WHY: a tiny wrapper keeps tests focused on the planned milliseconds instead
    of repeating the float conversion from milliseconds to seconds.
    """

    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)


async def _sleep_async_ms(delay_ms: int) -> None:
    """Sleep asynchronously for the requested emulated latency.

    WHY: the async emulation path should mirror the sync helper exactly while
    using asyncio-friendly waiting.
    """

    if delay_ms > 0:
        await asyncio.sleep(delay_ms / 1000.0)


@contextmanager
def activate_emulation_session(
    seed: int | None = None,
    overrides: EmulationSessionOverrides | None = None,
) -> Iterator[_EmulationSessionState]:
    """Activate deterministic per-run timing context for emulated models.

    WHY: one agent session should be able to produce a stable sequence of
    varied model-call latencies without mutating the cached model instance or
    relying on global process-wide randomness.
    """

    existing_state = _ACTIVE_EMULATION_SESSION.get()
    if existing_state is not None:
        yield existing_state
        return

    resolved_seed = (
        int(seed) if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    )
    session_state = _EmulationSessionState(
        seed=resolved_seed,
        rng=random.Random(resolved_seed),
        overrides=overrides,
    )
    token = _ACTIVE_EMULATION_SESSION.set(session_state)
    try:
        yield session_state
    finally:
        _ACTIVE_EMULATION_SESSION.reset(token)


class EmulatedChatModel(BaseChatModel):
    """Deterministic chat model used to exercise agent flows without provider calls.

    WHY: End-to-end, performance, and security tests need LangChain-compatible
    model behaviour, including streaming and tool binding, while guaranteeing
    that no network request reaches a paid provider.
    """

    model_name: str = "emulated-chat-model"
    response: str = "This is an emulated LLM response."
    profile: str = "deterministic"
    latency_ms: int = 0
    latency_jitter_ms: int = 0
    session_scale_min: float = 1.0
    session_scale_max: float = 1.0
    step_latency_growth: float = 0.0
    stream_first_chunk_ratio: float = 1.0
    error_rate: float = 0.0
    bound_tools: list[Any] = Field(default_factory=list, exclude=True)
    bound_tool_choice: Optional[str] = Field(default=None, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "inference-core-emulated-chat-model"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "profile": self.profile,
            "latency_ms": self.latency_ms,
            "latency_jitter_ms": self.latency_jitter_ms,
            "session_scale_min": self.session_scale_min,
            "session_scale_max": self.session_scale_max,
            "step_latency_growth": self.step_latency_growth,
            "stream_first_chunk_ratio": self.stream_first_chunk_ratio,
            "error_rate": self.error_rate,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        """Return a copy that remembers tool binding for agent compatibility.

        WHY: LangChain agents require chat models to support ``bind_tools``.
        The emulated model keeps this metadata so scripted profiles can later
        generate tool calls while deterministic profiles remain no-op.
        """
        return self.model_copy(
            update={
                "bound_tools": list(tools),
                "bound_tool_choice": tool_choice,
            }
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        latency_plan = self._build_latency_plan()
        self._maybe_sleep_sync(latency_plan)
        self._maybe_raise()
        message = self._build_message(messages, latency_plan=latency_plan)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _get_session_overrides(self) -> EmulationSessionOverrides | None:
        """Return active request-scoped overrides for the current session.

        WHY: one helper keeps override lookup centralized across message,
        streaming, latency, and error-planning code paths.
        """

        session_state = _ACTIVE_EMULATION_SESSION.get()
        return session_state.overrides if session_state is not None else None

    def _effective_profile(self) -> str:
        """Return the profile name visible to observers for this invocation.

        WHY: request-scoped perf overrides should appear in response metadata so
        traces and test logs can distinguish special timing profiles.
        """

        overrides = self._get_session_overrides()
        if overrides is not None and overrides.profile:
            return overrides.profile
        return self.profile

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        latency_plan = self._build_latency_plan()
        await self._maybe_sleep_async(latency_plan)
        self._maybe_raise()
        message = self._build_message(messages, latency_plan=latency_plan)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        chunks = self._build_chunks(messages)
        latency_plan = self._build_latency_plan(chunk_count=len(chunks))
        self._maybe_raise()
        for delay_ms, chunk in zip(latency_plan.pre_chunk_delays_ms, chunks):
            _sleep_sync_ms(delay_ms)
            yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ):
        chunks = self._build_chunks(messages)
        latency_plan = self._build_latency_plan(chunk_count=len(chunks))
        self._maybe_raise()
        for delay_ms, chunk in zip(latency_plan.pre_chunk_delays_ms, chunks):
            await _sleep_async_ms(delay_ms)
            yield ChatGenerationChunk(message=chunk)

    def _build_message(
        self,
        messages: list[BaseMessage],
        *,
        latency_plan: EmulationLatencyPlan | None = None,
    ) -> AIMessage:
        """Build the deterministic response with usage metadata for observability.

        WHY: Cost tracking and run logging should exercise the same code paths
        in test environments, even though the calculated provider cost is zero.
        """
        response_text = self._render_response(messages)
        input_tokens = sum(
            _estimate_tokens(str(message.content)) for message in messages
        )
        output_tokens = _estimate_tokens(response_text)
        response_metadata = {
            "model_name": self.model_name,
            "provider": "emulated",
            "profile": self._effective_profile(),
            "finish_reason": "stop",
        }
        if latency_plan is not None:
            response_metadata.update(
                {
                    "emulated_delay_ms": latency_plan.total_delay_ms,
                    "emulation_call_index": latency_plan.call_index,
                }
            )
            if latency_plan.session_seed is not None:
                response_metadata["emulation_session_seed"] = latency_plan.session_seed
        return AIMessage(
            content=response_text,
            response_metadata=response_metadata,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )

    def _build_chunks(
        self,
        messages: list[BaseMessage],
        *,
        latency_plan: EmulationLatencyPlan | None = None,
    ) -> list[AIMessageChunk]:
        """Split the response into stable stream chunks.

        WHY: Streaming endpoints and LangGraph event streams need chunked output
        during tests, but deterministic chunking keeps assertions predictable.
        """
        response_text = self._render_response(messages)
        words = response_text.split()
        if not words:
            words = [""]

        chunks: list[AIMessageChunk] = []
        for index, word in enumerate(words):
            separator = "" if index == len(words) - 1 else " "
            response_metadata = {
                "model_name": self.model_name,
                "provider": "emulated",
                "profile": self._effective_profile(),
            }
            if latency_plan is not None:
                response_metadata.update(
                    {
                        "emulated_delay_ms": latency_plan.total_delay_ms,
                        "emulation_call_index": latency_plan.call_index,
                    }
                )
                if latency_plan.session_seed is not None:
                    response_metadata["emulation_session_seed"] = (
                        latency_plan.session_seed
                    )
            chunks.append(
                AIMessageChunk(
                    content=f"{word}{separator}",
                    response_metadata=response_metadata,
                )
            )

        chunks[-1] = chunks[-1].model_copy(update={"chunk_position": "last"})
        return chunks

    def _render_response(self, messages: list[BaseMessage]) -> str:
        """Return the synthetic response text for one invocation.

        WHY: integration tests for the Agent Server need a small amount of
        prompt-following and conversational continuity even in no-cost mode.
        Keeping these rules local to the emulated model preserves the zero-cost
        guarantee without pretending to be a general-purpose LLM.
        """

        system_text = "\n".join(
            self._message_text(message)
            for message in messages
            if getattr(message, "type", "") == "system"
        )
        latest_user_text = self._latest_human_text(messages)

        override_match = re.search(
            r"reply with exactly this single token and nothing else:\s*([A-Za-z0-9_-]+)",
            system_text,
            flags=re.IGNORECASE,
        )
        if override_match:
            return override_match.group(1)

        response_text = self._rule_based_user_reply(messages, latest_user_text)

        append_match = re.search(
            r"end every reply with the literal token\s+([A-Za-z0-9_-]+)",
            system_text,
            flags=re.IGNORECASE,
        )
        if append_match:
            suffix = append_match.group(1)
            if suffix not in response_text:
                response_text = f"{response_text} {suffix}".strip()

        return response_text

    def _rule_based_user_reply(
        self,
        messages: list[BaseMessage],
        latest_user_text: str,
    ) -> str:
        """Handle the narrow scripted prompts used by contract tests.

        WHY: Agent Server tests intentionally send rigid instructions such as
        ``Say exactly: ...`` and simple memory recalls. Recognizing only these
        patterns keeps the emulator predictable while making those contracts
        meaningful in testing mode.
        """

        exact_reply_match = re.search(
            r"(?:say exactly|reply with one word)\s*:\s*(.+)",
            latest_user_text,
            flags=re.IGNORECASE,
        )
        if exact_reply_match:
            return exact_reply_match.group(1).strip().strip(". ")

        if re.search(
            r"what number did i ask you to remember\??",
            latest_user_text,
            flags=re.IGNORECASE,
        ):
            remembered_number = self._remembered_number(messages)
            if remembered_number is not None:
                return f"You asked me to remember {remembered_number}."

        return self.response

    def _remembered_number(self, messages: list[BaseMessage]) -> str | None:
        """Return the last explicitly remembered number from user history.

        WHY: thread-persistence tests only need one narrow form of continuity:
        the ability to echo back a number the user asked the agent to remember.
        """

        for message in reversed(messages[:-1]):
            if getattr(message, "type", "") != "human":
                continue
            remembered_match = re.search(
                r"remember this number:\s*([-+]?\d+(?:\.\d+)?)",
                self._message_text(message),
                flags=re.IGNORECASE,
            )
            if remembered_match:
                return remembered_match.group(1)
        return None

    def _latest_human_text(self, messages: list[BaseMessage]) -> str:
        """Return the newest human message as plain text.

        WHY: rule-based reply selection should anchor on the current user turn,
        not on earlier prompts that may still be present in the thread history.
        """

        for message in reversed(messages):
            if getattr(message, "type", "") == "human":
                return self._message_text(message)
        return ""

    def _message_text(self, message: BaseMessage) -> str:
        """Flatten message content into plain text for rule-based matching.

        WHY: LangGraph may hand the emulator either raw strings or structured
        content blocks. The scripted testing rules only need a stable text view.
        """

        content = getattr(message, "content", "") or ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = (
                        block.get("text")
                        or block.get("thinking")
                        or block.get("reasoning")
                        or ""
                    )
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(block))
            return " ".join(part.strip() for part in parts if part).strip()
        return str(content)

    def _build_latency_plan(self, *, chunk_count: int = 0) -> EmulationLatencyPlan:
        """Plan latency for one emulated model call.

        WHY: one planner keeps jitter, per-session scaling, and progressive
        call growth consistent across invoke and stream code paths.
        """

        session_state = _ACTIVE_EMULATION_SESSION.get()
        overrides = session_state.overrides if session_state is not None else None
        call_index = 1
        session_seed: int | None = None
        session_scale = 1.0
        jitter_ms = 0
        latency_ms = (
            overrides.latency_ms
            if overrides is not None and overrides.latency_ms is not None
            else self.latency_ms
        )
        latency_jitter_ms = (
            overrides.latency_jitter_ms
            if overrides is not None and overrides.latency_jitter_ms is not None
            else self.latency_jitter_ms
        )
        session_scale_min = (
            overrides.session_scale_min
            if overrides is not None and overrides.session_scale_min is not None
            else self.session_scale_min
        )
        session_scale_max = (
            overrides.session_scale_max
            if overrides is not None and overrides.session_scale_max is not None
            else self.session_scale_max
        )
        step_latency_growth = (
            overrides.step_latency_growth
            if overrides is not None and overrides.step_latency_growth is not None
            else self.step_latency_growth
        )
        stream_first_chunk_ratio = (
            overrides.stream_first_chunk_ratio
            if overrides is not None and overrides.stream_first_chunk_ratio is not None
            else self.stream_first_chunk_ratio
        )

        if session_state is not None:
            call_index = session_state.call_index + 1
            session_seed = session_state.seed
            session_min, session_max = _coerce_session_scale_bounds(
                session_scale_min,
                session_scale_max,
            )
            if session_state.session_scale is None:
                session_state.session_scale = session_state.rng.uniform(
                    session_min,
                    session_max,
                )
            session_scale = session_state.session_scale
            if latency_jitter_ms > 0:
                jitter_ms = session_state.rng.randint(
                    -latency_jitter_ms,
                    latency_jitter_ms,
                )
            session_state.call_index = call_index

        scaled_base_latency_ms = max(0, latency_ms + jitter_ms)
        growth_multiplier = 1.0 + (max(0, call_index - 1) * step_latency_growth)
        total_delay_ms = max(
            0,
            int(round(scaled_base_latency_ms * session_scale * growth_multiplier)),
        )
        pre_chunk_delays_ms = ()
        if chunk_count > 0:
            pre_chunk_delays_ms = _distribute_stream_delays(
                total_delay_ms,
                chunk_count,
                stream_first_chunk_ratio,
            )

        return EmulationLatencyPlan(
            total_delay_ms=total_delay_ms,
            call_index=call_index,
            session_seed=session_seed,
            session_scale=session_scale,
            jitter_ms=jitter_ms,
            pre_chunk_delays_ms=pre_chunk_delays_ms,
        )

    def _maybe_sleep_sync(self, latency_plan: EmulationLatencyPlan) -> None:
        _sleep_sync_ms(latency_plan.total_delay_ms)

    async def _maybe_sleep_async(self, latency_plan: EmulationLatencyPlan) -> None:
        await _sleep_async_ms(latency_plan.total_delay_ms)

    def _maybe_raise(self) -> None:
        overrides = self._get_session_overrides()
        error_rate = (
            overrides.error_rate
            if overrides is not None and overrides.error_rate is not None
            else self.error_rate
        )
        if error_rate <= 0:
            return
        session_state = _ACTIVE_EMULATION_SESSION.get()
        probability_roll = (
            session_state.rng.random() if session_state is not None else random.random()
        )
        if probability_roll < error_rate:
            raise RuntimeError("Emulated LLM failure requested by test profile")


def create_emulated_chat_model(model_name: str, **kwargs: Any) -> EmulatedChatModel:
    """Create an emulated model using global test settings plus call overrides.

    WHY: Every caller that needs a no-cost chat model should use the same
    constructor so latency, error injection, and response profiles stay aligned
    across API tests, AgentService, and Agent Server.
    """
    from inference_core.core.config import get_settings

    settings = get_settings()
    callbacks = kwargs.pop("callbacks", None)
    model = EmulatedChatModel(
        model_name=model_name,
        response=kwargs.pop("response", settings.llm_emulation_response),
        profile=kwargs.pop("profile", settings.llm_emulation_profile),
        latency_ms=kwargs.pop("latency_ms", settings.llm_emulation_latency_ms),
        latency_jitter_ms=kwargs.pop(
            "latency_jitter_ms",
            getattr(settings, "llm_emulation_latency_jitter_ms", 0),
        ),
        session_scale_min=kwargs.pop(
            "session_scale_min",
            getattr(settings, "llm_emulation_session_scale_min", 1.0),
        ),
        session_scale_max=kwargs.pop(
            "session_scale_max",
            getattr(settings, "llm_emulation_session_scale_max", 1.0),
        ),
        step_latency_growth=kwargs.pop(
            "step_latency_growth",
            getattr(settings, "llm_emulation_step_latency_growth", 0.0),
        ),
        stream_first_chunk_ratio=kwargs.pop(
            "stream_first_chunk_ratio",
            getattr(settings, "llm_emulation_stream_first_chunk_ratio", 1.0),
        ),
        error_rate=kwargs.pop("error_rate", settings.llm_emulation_error_rate),
        callbacks=callbacks,
    )
    return model


def is_llm_emulation_enabled() -> bool:
    """Return whether chat model creation should be forced into no-cost mode.

    WHY: The model factory and middleware builders need a tiny shared guard to
    avoid importing provider SDKs or falling back to real models in tests.
    """
    from inference_core.core.config import get_settings

    return bool(get_settings().llm_emulation_enabled)


def build_tool_emulation_middleware(tools: list[Any]) -> Any | None:
    """Build an LLMToolEmulator that cannot fall back to a real provider.

    WHY: LangChain's default ``LLMToolEmulator`` constructs Anthropic Claude
    when no model is supplied. Passing our emulated model preserves the no-cost
    guarantee while still letting test profiles emulate tool results.
    """
    from inference_core.core.config import get_settings

    settings = get_settings()
    if not settings.llm_emulation_enabled:
        return None
    if settings.llm_tool_emulation_mode == "off":
        return None

    selected_tools = _select_tools_for_emulation(tools, settings)
    if selected_tools == []:
        return None

    emulator_model = create_emulated_chat_model(
        "tool-emulator",
        response="Emulated tool result.",
        profile="tool-emulator",
    )
    return LLMToolEmulator(tools=selected_tools, model=emulator_model)


def _select_tools_for_emulation(tools: list[Any], settings: Any) -> list[Any] | None:
    """Select tool names or instances for LangChain's tool emulator.

    WHY: Test profiles need coarse policies such as all tools, a configured
    allowlist, or only tools that declare external side effects.
    """
    excluded = set(settings.llm_tool_emulation_exclude or [])

    if settings.llm_tool_emulation_mode == "all":
        if not excluded:
            return None
        return [tool for tool in tools if _tool_name(tool) not in excluded]

    if settings.llm_tool_emulation_mode == "configured":
        included = set(settings.llm_tool_emulation_include or [])
        return [tool for tool in tools if _tool_name(tool) in included - excluded]

    if settings.llm_tool_emulation_mode == "external":
        return [
            tool
            for tool in tools
            if _tool_name(tool) not in excluded and _tool_is_external(tool)
        ]

    return []


def _tool_name(tool: Any) -> str | None:
    return getattr(tool, "name", None)


def _tool_is_external(tool: Any) -> bool:
    return any(
        bool(getattr(tool, attr, False))
        for attr in ("requires_network", "external_side_effects", "generates_cost")
    )


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    digest_bonus = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:2], 16) % 3
    return max(1, len(text.split()) + digest_bonus)

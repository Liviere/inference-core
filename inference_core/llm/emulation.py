"""No-cost LLM emulation primitives for test environments."""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
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
        self._maybe_sleep_sync()
        self._maybe_raise()
        message = self._build_message(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        await self._maybe_sleep_async()
        self._maybe_raise()
        message = self._build_message(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self._maybe_sleep_sync()
        self._maybe_raise()
        for chunk in self._build_chunks(messages):
            yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ):
        await self._maybe_sleep_async()
        self._maybe_raise()
        for chunk in self._build_chunks(messages):
            yield ChatGenerationChunk(message=chunk)

    def _build_message(self, messages: list[BaseMessage]) -> AIMessage:
        """Build the deterministic response with usage metadata for observability.

        WHY: Cost tracking and run logging should exercise the same code paths
        in test environments, even though the calculated provider cost is zero.
        """
        input_tokens = sum(
            _estimate_tokens(str(message.content)) for message in messages
        )
        output_tokens = _estimate_tokens(self.response)
        return AIMessage(
            content=self.response,
            response_metadata={
                "model_name": self.model_name,
                "provider": "emulated",
                "profile": self.profile,
                "finish_reason": "stop",
            },
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )

    def _build_chunks(self, messages: list[BaseMessage]) -> list[AIMessageChunk]:
        """Split the response into stable stream chunks.

        WHY: Streaming endpoints and LangGraph event streams need chunked output
        during tests, but deterministic chunking keeps assertions predictable.
        """
        words = self.response.split()
        if not words:
            words = [""]

        chunks: list[AIMessageChunk] = []
        for index, word in enumerate(words):
            separator = "" if index == len(words) - 1 else " "
            chunks.append(
                AIMessageChunk(
                    content=f"{word}{separator}",
                    response_metadata={
                        "model_name": self.model_name,
                        "provider": "emulated",
                        "profile": self.profile,
                    },
                )
            )

        chunks[-1] = chunks[-1].model_copy(update={"chunk_position": "last"})
        return chunks

    def _maybe_sleep_sync(self) -> None:
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

    async def _maybe_sleep_async(self) -> None:
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)

    def _maybe_raise(self) -> None:
        if self.error_rate <= 0:
            return
        if random.random() < self.error_rate:
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

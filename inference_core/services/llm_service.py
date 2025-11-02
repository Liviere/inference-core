"""
LLM Service Module

Main service class that provides high-level interface for all LLM operations.
This is the primary entry point for the API to interact with LLM capabilities.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, cast

from fastapi import Request
from pydantic import BaseModel

try:  # Optional imports used when MCP tooling is enabled
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_community.chat_message_histories import SQLChatMessageHistory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import BaseTool
except Exception:  # pragma: no cover - optional dependency guard
    AgentExecutor = None  # type: ignore
    create_openai_tools_agent = None  # type: ignore
    ChatPromptTemplate = None  # type: ignore
    MessagesPlaceholder = None  # type: ignore
    BaseTool = None  # type: ignore
    SQLChatMessageHistory = None  # type: ignore

from inference_core.llm.callbacks import LLMUsageCallbackHandler
from inference_core.llm.chains import create_chat_chain, create_completion_chain
from inference_core.llm.config import get_llm_config
from inference_core.llm.mcp_tools import get_mcp_tool_manager
from inference_core.llm.models import get_model_factory, task_override
from inference_core.llm.prompts import ChatPrompts
from inference_core.llm.usage_logging import UsageLogger
from inference_core.services.llm_usage_service import get_llm_usage_service

logger = logging.getLogger(__name__)


class LLMMetadata(BaseModel):
    """Metadata for LLM operations"""

    model_name: str
    timestamp: str


class LLMResponse(BaseModel):
    """Response model for LLM operations"""

    result: Dict[str, Any]
    metadata: LLMMetadata


class LLMService:
    """
    Main LLM Service providing high-level interface for AI operations.
    """

    def __init__(
        self,
        *,
        default_models: Optional[Dict[str, str]] = None,
        default_model_params: Optional[Dict[str, Dict[str, Any]]] = None,
        default_prompt_names: Optional[Dict[str, str]] = None,
        default_chat_system_prompt: Optional[str] = None,
    ):
        self.config = get_llm_config()
        self.model_factory = get_model_factory()
        self.usage_logger = UsageLogger(self.config.usage_logging)
        # Customization defaults for inheritance/clone patterns
        self._default_models = default_models or {}
        self._default_model_params = default_model_params or {}
        self._default_prompt_names = default_prompt_names or {}
        self._default_chat_system_prompt = default_chat_system_prompt
        self._mcp_tool_manager = get_mcp_tool_manager()
        self._usage_stats = {
            "requests_count": 0,
            "total_tokens": 0,
            "errors_count": 0,
            "last_request": None,
        }

    # --------- MCP tooling support ---------

    @dataclass
    class _ToolingContext:
        profile_name: str
        tools: List[Any]
        instructions: str
        limits: Dict[str, Any]

    async def _get_tooling_context(
        self,
        task_type: str,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Optional["LLMService._ToolingContext"]:
        """Return tooling context when MCP is enabled for the task.

        Args:
            task_type: Effective task type (e.g. "chat")
            user_context: Optional dictionary with user flags (expects is_superuser)

        Returns:
            Tooling context or None when MCP should not be used.
        """

        mcp_cfg = self.config.mcp_config
        if not (mcp_cfg and mcp_cfg.enabled):
            return None

        task_config = self.config.task_configs.get(task_type)
        profile_name = task_config.mcp_profile if task_config else None
        if not profile_name:
            return None

        limits = mcp_cfg.get_profile(profile_name)
        if limits is None:
            logger.warning(
                "MCP profile '%s' referenced by task '%s' is not defined",
                profile_name,
                task_type,
            )
            return None

        try:
            tools = await self._mcp_tool_manager.get_tools(
                profile_name=profile_name,
                user=user_context,
            )
        except PermissionError:
            logger.warning(
                "User lacks permission for MCP profile '%s' (task '%s')",
                profile_name,
                task_type,
            )
            return None
        except Exception as exc:  # pragma: no cover - safety log
            logger.error(
                "Failed to load MCP tools for profile '%s': %s",
                profile_name,
                exc,
            )
            return None

        if not tools:
            return None

        instructions = self._build_tool_instructions(profile_name, tools, limits)
        limits_dict = self._mcp_tool_manager.get_profile_limits(profile_name)
        return LLMService._ToolingContext(
            profile_name=profile_name,
            tools=tools,
            instructions=instructions,
            limits=limits_dict,
        )

    @staticmethod
    def _build_tool_instructions(
        profile_name: str, tools: List[Any], limits: Any
    ) -> str:
        """Build instruction text for the system prompt to expose available tools."""

        header = (
            "You have access to external tools via the Model Context Protocol (MCP). "
            "Use them when needed to complete the task. Call a tool only if it helps."
        )
        lines = [header, f"Active profile: {profile_name}."]

        if getattr(limits, "max_steps", None):
            lines.append(f"Maximum tool iterations: {limits.max_steps}.")
        if getattr(limits, "max_run_seconds", None):
            lines.append(
                f"Hard timeout for tool usage: {limits.max_run_seconds} seconds."
            )

        lines.append("Available tools:")
        for tool in tools:
            name = getattr(tool, "name", "unknown-tool")
            description = getattr(tool, "description", "No description provided")
            lines.append(f"- {name}: {description}")

        lines.append(
            "If a tool returns data, summarise the outcome for the user before responding."
        )
        # Dev ergonomics: avoid closing the browser unless the user explicitly asks.
        # This helps when using headful Playwright MCP during interactive sessions.
        lines.append(
            "Do NOT call 'browser_close' unless the user explicitly instructs you to close the browser."
        )
        return "\n".join(lines)

    @staticmethod
    def _sync_connection_string() -> str:
        """Return synchronous DB URI for chat history storage."""

        from inference_core.core.config import get_settings

        settings = get_settings()
        url = settings.database_url
        if "+aiosqlite" in url:
            return url.replace("+aiosqlite", "")
        if "+asyncpg" in url:
            return url.replace("+asyncpg", "+psycopg")
        if "+aiomysql" in url:
            return url.replace("+aiomysql", "+pymysql")
        return url

    async def _chat_with_tools(
        self,
        *,
        session_id: str,
        user_input: str,
        resolved_model_name: str,
        model_params: Dict[str, Any],
        usage_session,
        callbacks,
        tooling: "LLMService._ToolingContext",
    ) -> LLMResponse:
        """Run a tool-enabled agent conversation."""

        if AgentExecutor is None or create_openai_tools_agent is None:
            raise RuntimeError(
                "LangChain agent tooling is unavailable. Ensure langchain.agents is installed."
            )

        # Prepare model with runtime params
        with task_override("chat"):
            model = self.model_factory.create_model(resolved_model_name, **model_params)
        if not model:
            raise ValueError(
                f"Failed to create model '{resolved_model_name}' for MCP chat"
            )

        try:
            base_system_prompt = ChatPrompts.CHAT.format_messages(  # type: ignore[misc]
                history=[],
                user_input="",
            )[0].content
        except (
            Exception
        ):  # pragma: no cover - fallback if format_messages signature changes
            base_system_prompt = (
                "You are a helpful assistant that provides concise and accurate answers to user questions. "
                "Always respond in a friendly and professional manner."
            )

        system_prompt = f"{base_system_prompt}\n\n{tooling.instructions}"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_openai_tools_agent(model, tooling.tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tooling.tools,
            handle_parsing_errors=True,
            max_iterations=tooling.limits.get("max_steps", 10),
            verbose=False,
        )

        # Load history synchronously, then append new messages
        if SQLChatMessageHistory is None:
            raise RuntimeError(
                "SQLChatMessageHistory is unavailable. Install langchain-community to enable chat history."
            )
        history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string=self._sync_connection_string(),
        )
        history_messages = history.messages

        config = {}
        if callbacks:
            config["callbacks"] = callbacks

        timeout = tooling.limits.get("max_run_seconds", 60)

        try:
            result = await asyncio.wait_for(
                agent_executor.ainvoke(
                    {
                        "input": user_input,
                        "chat_history": history_messages,
                    },
                    config=config,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"MCP-enabled chat exceeded timeout of {timeout} seconds"
            ) from exc

        output_text = result.get("output", "")

        # Persist history manually as we bypassed RunnableWithMessageHistory
        history.add_user_message(user_input)
        history.add_ai_message(output_text)

        return LLMResponse(
            result={"answer": output_text, "tool_profile": tooling.profile_name},
            metadata=LLMMetadata(
                model_name=resolved_model_name,
                timestamp=datetime.now(UTC).isoformat(),
            ),
        )

    # --------- Helper hooks for subclasses ---------
    def _effective_model_name(self, task: str, override: Optional[str]) -> str:
        """Resolve model name with precedence: override > default_models > config mapping."""
        if override:
            return override
        if task in self._default_models:
            return self._default_models[task]
        # Fall back to config's task mapping (supports custom task types)
        return self.config.get_task_model(task)

    def _merge_params(
        self, task: str, runtime_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge default model params with runtime params (runtime wins)."""
        base = dict(self._default_model_params.get(task, {}))
        base.update(runtime_params)
        return base

    def _build_completion_chain(
        self,
        *,
        model_name: Optional[str],
        model_params: Dict[str, Any],
        prompt_name: Optional[str],
    ):
        """Factory hook to build completion chain; subclasses can override."""
        kwargs: Dict[str, Any] = dict(model_name=model_name)
        effective_prompt_name = prompt_name or self._default_prompt_names.get(
            "completion"
        )
        if effective_prompt_name is not None:
            kwargs["prompt_name"] = effective_prompt_name
        kwargs.update(model_params)
        return create_completion_chain(**kwargs)

    def _build_chat_chain(
        self,
        *,
        model_name: Optional[str],
        model_params: Dict[str, Any],
        prompt_name: Optional[str],
        system_prompt: Optional[str],
    ):
        """Factory hook to build chat chain; subclasses can override."""
        kwargs: Dict[str, Any] = dict(model_name=model_name)
        effective_prompt_name = prompt_name or self._default_prompt_names.get("chat")
        if effective_prompt_name is not None:
            kwargs["prompt_name"] = effective_prompt_name
        effective_system_prompt = system_prompt or self._default_chat_system_prompt
        if effective_system_prompt is not None:
            kwargs["system_prompt"] = effective_system_prompt
        kwargs.update(model_params)
        return create_chat_chain(**kwargs)

    def copy_with(
        self,
        *,
        default_models: Optional[Dict[str, str]] = None,
        default_model_params: Optional[Dict[str, Dict[str, Any]]] = None,
        default_prompt_names: Optional[Dict[str, str]] = None,
        default_chat_system_prompt: Optional[str] = None,
    ) -> "LLMService":
        """Create a shallow clone with updated defaults (easy task copying)."""
        return LLMService(
            default_models=default_models or self._default_models,
            default_model_params=default_model_params or self._default_model_params,
            default_prompt_names=default_prompt_names or self._default_prompt_names,
            default_chat_system_prompt=(
                default_chat_system_prompt
                if default_chat_system_prompt is not None
                else self._default_chat_system_prompt
            ),
        )

    async def completion(
        self,
        prompt: Optional[str] = None,
        question: Optional[str] = None,
        model_name: Optional[str] = None,
        *,
        task_type: Optional[str] = "completion",
        input_vars: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        request_timeout: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        prompt_name: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a completion-style answer for a given prompt using the specified model.

        Args:
            prompt: Text prompt to generate from (preferred)
            input_vars: Optional dict of variables for the prompt template; if provided, overrides 'prompt'
            question: Legacy alias for prompt (kept for backward compatibility)
            model_name: Optional model name to override default

        Returns:
            Completion string
        """
        effective_task = task_type or "completion"
        resolved_model_name = self._effective_model_name(effective_task, model_name)

        preview = None
        if input_vars and isinstance(input_vars, dict):
            # Prefer common key for preview
            preview = (
                input_vars.get("prompt")
                if isinstance(input_vars.get("prompt"), str)
                else None
            )
            if preview is None:
                # fallback: first string value
                for v in input_vars.values():
                    if isinstance(v, str):
                        preview = v
                        break
        if preview is None:
            preview = (prompt if prompt is not None else question) or ""
        self._log_request(
            "completion",
            {"prompt": preview, "model_name": model_name, "task_type": effective_task},
        )

        # Deprecation notice for legacy 'question'
        if question is not None:
            logger.warning(
                "LLMService.completion(): parameter 'question' is deprecated; use 'prompt' or 'input_vars'"
            )

        # Start usage logging session
        model_config = self.config.models.get(resolved_model_name)
        provider = model_config.provider if model_config else "unknown"

        usage_session = self.usage_logger.start_session(
            task_type=effective_task,
            request_mode="sync",
            model_name=resolved_model_name,
            provider=provider,
            pricing_config=model_config.pricing if model_config else None,
            user_id=uuid.UUID(user_id) if user_id else None,
            request_id=request_id,
        )
        callbacks = []
        if self.config.usage_logging.enabled:
            callbacks.append(
                LLMUsageCallbackHandler(
                    usage_session=usage_session,
                    pricing_config=model_config.pricing if model_config else None,
                )
            )

        try:
            # Deprecation guard for GPT-5 family: classic sampling params removed
            if model_name and model_name.startswith("gpt-5"):
                for legacy in [
                    ("temperature", temperature),
                    ("top_p", top_p),
                    ("frequency_penalty", frequency_penalty),
                    ("presence_penalty", presence_penalty),
                ]:
                    if legacy[1] is not None:
                        raise ValueError(
                            f"Parameter '{legacy[0]}' is deprecated for {model_name}; use reasoning_effort / verbosity"
                        )
            runtime_params = {
                k: v
                for k, v in {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "timeout": request_timeout,
                    "reasoning_effort": reasoning_effort,
                    "verbosity": verbosity,
                }.items()
                if v is not None
            }
            model_params = self._merge_params(effective_task, runtime_params)
            # Use factory hook (supports prompt overrides)
            with task_override(effective_task):
                chain = self._build_completion_chain(
                    model_name=model_name,
                    model_params=model_params,
                    prompt_name=prompt_name
                    or self._default_prompt_names.get(effective_task),
                )
            # Call chain respecting legacy call shape for tests
            if input_vars is not None:
                variables = dict(input_vars)
                answer = await chain.completion(
                    input_vars=variables, callbacks=callbacks
                )
            else:
                text = (prompt if prompt is not None else question) or ""
                answer = await chain.completion(prompt=text, callbacks=callbacks)

            # Usage already accumulated by callback handler
            usage_metadata = usage_session.accumulated_usage

            result = LLMResponse(
                result={"answer": answer},
                metadata=LLMMetadata(
                    model_name=chain.model_name,
                    timestamp=datetime.now(UTC).isoformat(),
                ),
            )

            self._update_usage_stats()

            # Finalize usage logging
            await usage_session.finalize(
                success=True,
                final_usage=usage_metadata,
                streamed=False,
                partial=False,
            )

            return result
        except Exception as e:
            self._handle_error("completion", e)

            # Finalize usage logging with error
            await usage_session.finalize(
                success=False,
                error=e,
                streamed=False,
                partial=False,
            )
            raise e

    async def chat(
        self,
        session_id: str,
        user_input: str,
        model_name: Optional[str] = None,
        *,
        task_type: Optional[str] = "chat",
        input_vars: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        request_timeout: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        prompt_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> LLMResponse:
        """Engage in a multi-turn chat within a session.

        Args:
            session_id: Unique identifier for the chat session
            user_input: The user's message
            model_name: Optional model name override

        Returns:
            LLMResponse with assistant reply
        """
        effective_task = task_type or "chat"
        resolved_model_name = self._effective_model_name(effective_task, model_name)

        self._log_request(
            "chat",
            {
                "session_id": session_id,
                "user_input": (
                    (user_input[:128] if isinstance(user_input, str) else None)
                    if user_input is not None
                    else (
                        input_vars.get("user_input")[:128]
                        if isinstance(input_vars, dict)
                        and isinstance(input_vars.get("user_input"), str)
                        else None
                    )
                ),
                "model_name": model_name,
                "task_type": effective_task,
            },
        )

        # Start usage logging session
        model_config = self.config.models.get(resolved_model_name)
        provider = model_config.provider if model_config else "unknown"

        usage_session = self.usage_logger.start_session(
            task_type=effective_task,
            request_mode="sync",
            model_name=resolved_model_name,
            provider=provider,
            pricing_config=model_config.pricing if model_config else None,
            session_id=session_id,
            user_id=uuid.UUID(user_id) if user_id else None,
            request_id=request_id,
        )
        callbacks = []
        if self.config.usage_logging.enabled:
            callbacks.append(
                LLMUsageCallbackHandler(
                    usage_session=usage_session,
                    pricing_config=model_config.pricing if model_config else None,
                )
            )

        try:
            # Map request_timeout to factory's expected 'timeout'
            if model_name and model_name.startswith("gpt-5"):
                for legacy in [
                    ("temperature", temperature),
                    ("top_p", top_p),
                    ("frequency_penalty", frequency_penalty),
                    ("presence_penalty", presence_penalty),
                ]:
                    if legacy[1] is not None:
                        raise ValueError(
                            f"Parameter '{legacy[0]}' is deprecated for {model_name}; use reasoning_effort / verbosity"
                        )
            runtime_params = {
                k: v
                for k, v in {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "timeout": request_timeout,
                    "reasoning_effort": reasoning_effort,
                    "verbosity": verbosity,
                }.items()
                if v is not None
            }
            model_params = self._merge_params(effective_task, runtime_params)

            # Try MCP tooling path if available for this task/profile
            tooling_ctx = await self._get_tooling_context(
                effective_task,
                user_context=None,
            )
            if tooling_ctx is not None:
                try:
                    response = await self._chat_with_tools(
                        session_id=session_id,
                        user_input=user_input,
                        resolved_model_name=resolved_model_name,
                        model_params=model_params,
                        usage_session=usage_session,
                        callbacks=callbacks,
                        tooling=tooling_ctx,
                    )

                    usage_metadata = usage_session.accumulated_usage
                    self._update_usage_stats()
                    await usage_session.finalize(
                        success=True,
                        final_usage=usage_metadata,
                        streamed=False,
                        partial=False,
                    )
                    return response
                except Exception as tool_err:
                    logger.warning(
                        "Tool-enabled chat failed, falling back to standard chain: %s",
                        tool_err,
                    )
                    # Fall through to standard chat chain
            # Use factory hook with prompt/system overrides
            with task_override(effective_task):
                chain = self._build_chat_chain(
                    model_name=model_name,
                    model_params=model_params,
                    prompt_name=prompt_name
                    or self._default_prompt_names.get(effective_task),
                    system_prompt=system_prompt or self._default_chat_system_prompt,
                )
            if input_vars is not None:
                reply = await chain.chat(
                    session_id=session_id,
                    user_input=user_input,
                    input_vars=input_vars,
                    callbacks=callbacks,
                )
            else:
                reply = await chain.chat(
                    session_id=session_id,
                    user_input=user_input,
                    callbacks=callbacks,
                )

            usage_metadata = usage_session.accumulated_usage

            result = LLMResponse(
                result={"reply": reply, "session_id": session_id},
                metadata=LLMMetadata(
                    model_name=chain.model_name,
                    timestamp=datetime.now(UTC).isoformat(),
                ),
            )
            self._update_usage_stats()

            # Finalize usage logging
            await usage_session.finalize(
                success=True,
                final_usage=usage_metadata,
                streamed=False,
                partial=False,
            )

            return result
        except Exception as e:
            self._handle_error("chat", e)

            # Finalize usage logging with error
            await usage_session.finalize(
                success=False,
                error=e,
                streamed=False,
                partial=False,
            )
            raise e

    async def stream_chat(
        self,
        session_id: Optional[str],
        user_input: str,
        model_name: Optional[str] = None,
        request: Optional[Request] = None,
        *,
        task_type: Optional[str] = None,
        input_vars: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        request_timeout: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        prompt_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream a chat response using Server-Sent Events.

        Args:
            session_id: Chat session ID (auto-generated if None)
            user_input: User's message
            model_name: Optional model name override
            request: FastAPI request object for disconnect detection

        Returns:
            AsyncGenerator yielding SSE-formatted bytes
        """
        # Import here to avoid circular imports
        from inference_core.llm.streaming import stream_chat

        effective_task = task_type or "chat"
        self._log_request(
            "stream_chat",
            {
                "session_id": session_id,
                "user_input": user_input[:128],
                "model_name": model_name,
                "task_type": effective_task,
            },
        )

        try:
            # Build model parameters
            model_params: Dict[str, Any] = {}
            if temperature is not None:
                model_params["temperature"] = temperature
            if max_tokens is not None:
                model_params["max_tokens"] = max_tokens
            if top_p is not None:
                model_params["top_p"] = top_p
            if frequency_penalty is not None:
                model_params["frequency_penalty"] = frequency_penalty
            if presence_penalty is not None:
                model_params["presence_penalty"] = presence_penalty
            if request_timeout is not None:
                model_params["request_timeout"] = request_timeout
            if reasoning_effort is not None:
                model_params["reasoning_effort"] = reasoning_effort
            if verbosity is not None:
                model_params["verbosity"] = verbosity
            # Merge with defaults for task
            model_params = self._merge_params(effective_task, model_params)

            # Map request_timeout to factory's expected 'timeout'
            if model_name and model_name.startswith("gpt-5"):
                for legacy in [
                    ("temperature", temperature),
                    ("top_p", top_p),
                    ("frequency_penalty", frequency_penalty),
                    ("presence_penalty", presence_penalty),
                ]:
                    if legacy[1] is not None:
                        raise ValueError(
                            f"Parameter '{legacy[0]}' is deprecated for {model_name}; use reasoning_effort / verbosity"
                        )

            # Build optional extras to preserve backward-compatible call shape
            _extras: Dict[str, Any] = {}
            if input_vars is not None:
                _extras["input_vars"] = input_vars

            with task_override(effective_task):
                async for chunk in stream_chat(
                    session_id=session_id,
                    user_input=user_input,
                    model_name=model_name,
                    request=request,
                    prompt_name=prompt_name
                    or self._default_prompt_names.get(effective_task),
                    system_prompt=system_prompt or self._default_chat_system_prompt,
                    user_id=user_id,
                    request_id=request_id,
                    **_extras,
                    **model_params,
                ):
                    yield chunk

            self._update_usage_stats()
        except Exception as e:
            self._handle_error("stream_chat", e)
            raise e

    async def stream_completion(
        self,
        prompt: Optional[str] = None,
        question: Optional[str] = None,
        model_name: Optional[str] = None,
        request: Optional[Request] = None,
        *,
        task_type: Optional[str] = None,
        input_vars: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        request_timeout: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        prompt_name: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream a completion response using Server-Sent Events.

        Args:
            prompt: Text prompt to generate from (preferred)
            question: Legacy alias for prompt
            model_name: Optional model name override
            request: FastAPI request object for disconnect detection

        Returns:
            AsyncGenerator yielding SSE-formatted bytes
        """
        # Import here to avoid circular imports
        from inference_core.llm.streaming import stream_completion

        effective_task = task_type or "completion"
        self._log_request(
            "stream_completion",
            {
                "prompt": (prompt if prompt is not None else question or "")[:128],
                "model_name": model_name,
                "task_type": effective_task,
            },
        )

        try:
            # Build model parameters
            model_params: Dict[str, Any] = {}
            if temperature is not None:
                model_params["temperature"] = temperature
            if max_tokens is not None:
                model_params["max_tokens"] = max_tokens
            if top_p is not None:
                model_params["top_p"] = top_p
            if frequency_penalty is not None:
                model_params["frequency_penalty"] = frequency_penalty
            if presence_penalty is not None:
                model_params["presence_penalty"] = presence_penalty
            if request_timeout is not None:
                model_params["request_timeout"] = request_timeout
            if reasoning_effort is not None:
                model_params["reasoning_effort"] = reasoning_effort
            if verbosity is not None:
                model_params["verbosity"] = verbosity
            # Merge with defaults for task
            model_params = self._merge_params(effective_task, model_params)

            # Map request_timeout to factory's expected 'timeout'
            if model_name and model_name.startswith("gpt-5"):
                for legacy in [
                    ("temperature", temperature),
                    ("top_p", top_p),
                    ("frequency_penalty", frequency_penalty),
                    ("presence_penalty", presence_penalty),
                ]:
                    if legacy[1] is not None:
                        raise ValueError(
                            f"Parameter '{legacy[0]}' is deprecated for {model_name}; use reasoning_effort / verbosity"
                        )

            # Deprecation notice for legacy 'question'
            if question is not None:
                logger.warning(
                    "LLMService.stream_completion(): parameter 'question' is deprecated; use 'prompt' or 'input_vars'"
                )

            # Build optional extras to preserve backward-compatible call shape
            _extras: Dict[str, Any] = {}
            if input_vars is not None:
                _extras["input_vars"] = input_vars

            with task_override(effective_task):
                async for chunk in stream_completion(
                    prompt=prompt if prompt is not None else question,
                    model_name=model_name,
                    request=request,
                    prompt_name=prompt_name
                    or self._default_prompt_names.get(effective_task),
                    user_id=user_id,
                    request_id=request_id,
                    **_extras,
                    **model_params,
                ):
                    yield chunk

            self._update_usage_stats()
        except Exception as e:
            self._handle_error("stream_completion", e)
            raise e

    def get_available_models(self) -> Dict[str, bool]:
        """Get list of available models"""
        result = self.model_factory.get_available_models()
        return cast(Dict[str, bool], result)

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics including cost information"""
        # Get legacy stats for backward compatibility
        result = self._usage_stats.copy()

        # Get enhanced stats from usage logging service
        if self.config.usage_logging.enabled:
            try:
                usage_service = get_llm_usage_service()
                enhanced_stats = await usage_service.get_usage_stats()

                # Merge enhanced stats while maintaining backward compatibility
                result.update(enhanced_stats)

            except Exception as e:
                logger.error(f"Failed to get enhanced usage stats: {e}")
                # Fall back to legacy stats only

        return cast(Dict[str, Any], result)

    def _log_request(self, operation: str, params: Dict[str, Any]):
        """Log request for monitoring"""
        if self.config.enable_monitoring:
            # Remove sensitive data for logging
            safe_params = {
                k: v for k, v in params.items() if k not in ["self", "kwargs"]
            }
            logger.info(f"LLM operation: {operation}, params: {safe_params}")

    def _update_usage_stats(self, success: bool = True):
        """Update usage statistics"""
        self._usage_stats["requests_count"] += 1
        self._usage_stats["last_request"] = datetime.now(UTC).isoformat()
        if not success:
            self._usage_stats["errors_count"] += 1

    def _handle_error(self, operation: str, error: Exception):
        """Handle and log errors"""
        logger.error(f"LLM operation '{operation}' failed: {str(error)}")
        self._update_usage_stats(success=False)


# Global service instance
llm_service = LLMService()


def get_llm_service() -> LLMService:
    """Get global LLM service instance"""
    return llm_service


# No local aliases; canonical factories are imported from inference_core.llm.chains

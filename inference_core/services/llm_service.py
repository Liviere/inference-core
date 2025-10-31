"""
LLM Service Module

Main service class that provides high-level interface for all LLM operations.
This is the primary entry point for the API to interact with LLM capabilities.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, AsyncGenerator, Dict, Optional, cast

from fastapi import Request
from pydantic import BaseModel

from inference_core.llm.callbacks import LLMUsageCallbackHandler
from inference_core.llm.chains import create_chat_chain, create_completion_chain
from inference_core.llm.config import get_llm_config
from inference_core.llm.models import get_model_factory
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
        self._usage_stats = {
            "requests_count": 0,
            "total_tokens": 0,
            "errors_count": 0,
            "last_request": None,
        }

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
        effective_task = task_type or "completion"
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
        resolved_model_name = self._effective_model_name(effective_task, model_name)
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
        resolved_model_name = self._effective_model_name(effective_task, model_name)
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
            # Use factory hook with prompt/system overrides
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

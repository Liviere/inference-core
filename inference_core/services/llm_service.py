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
from inference_core.llm.chains import (
    create_conversation_chain,
    create_explanation_chain,
)
from inference_core.llm.config import llm_config
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

    def __init__(self):
        self.config = llm_config
        self.model_factory = get_model_factory()
        self.usage_logger = UsageLogger(self.config.usage_logging)
        self._usage_stats = {
            "requests_count": 0,
            "total_tokens": 0,
            "errors_count": 0,
            "last_request": None,
        }

    async def explain(
        self,
        question: str,
        model_name: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        request_timeout: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate an explanation for a given question using the specified model.

        Args:
            question: The question to explain
            model_name: Optional model name to override default

        Returns:
            Explanation string
        """
        self._log_request("explain", {"question": question, "model_name": model_name})

        # Start usage logging session
        resolved_model_name = model_name or self.config.get_task_model("explain")
        model_config = self.config.models.get(resolved_model_name)
        provider = model_config.provider if model_config else "unknown"

        usage_session = self.usage_logger.start_session(
            task_type="explain",
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
            model_params = {
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
            chain = create_explanation_chain(model_name=model_name, **model_params)
            answer = await chain.generate_story(question=question, callbacks=callbacks)

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
            self._handle_error("explain", e)

            # Finalize usage logging with error
            await usage_session.finalize(
                success=False,
                error=e,
                streamed=False,
                partial=False,
            )
            raise e

    async def converse(
        self,
        session_id: str,
        user_input: str,
        model_name: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        request_timeout: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> LLMResponse:
        """Engage in a multi-turn conversation within a session.

        Args:
            session_id: Unique identifier for the conversation session
            user_input: The user's message
            model_name: Optional model name override

        Returns:
            LLMResponse with assistant reply
        """
        self._log_request(
            "conversation",
            {
                "session_id": session_id,
                "user_input": user_input[:128],
                "model_name": model_name,
            },
        )

        # Start usage logging session
        resolved_model_name = model_name or self.config.get_task_model("conversation")
        model_config = self.config.models.get(resolved_model_name)
        provider = model_config.provider if model_config else "unknown"

        usage_session = self.usage_logger.start_session(
            task_type="conversation",
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
            model_params = {
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

            chain = create_conversation_chain(model_name=model_name, **model_params)
            reply = await chain.chat(
                session_id=session_id, user_input=user_input, callbacks=callbacks
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
            self._handle_error("conversation", e)

            # Finalize usage logging with error
            await usage_session.finalize(
                success=False,
                error=e,
                streamed=False,
                partial=False,
            )
            raise e

    async def stream_conversation(
        self,
        session_id: Optional[str],
        user_input: str,
        model_name: Optional[str] = None,
        request: Optional[Request] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        request_timeout: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream a conversation response using Server-Sent Events.

        Args:
            session_id: Conversation session ID (auto-generated if None)
            user_input: User's message
            model_name: Optional model name override
            request: FastAPI request object for disconnect detection

        Returns:
            AsyncGenerator yielding SSE-formatted bytes
        """
        # Import here to avoid circular imports
        from inference_core.llm.streaming import stream_conversation

        self._log_request(
            "stream_conversation",
            {
                "session_id": session_id,
                "user_input": user_input[:128],
                "model_name": model_name,
            },
        )

        try:
            # Build model parameters
            model_params = {}
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

            async for chunk in stream_conversation(
                session_id=session_id,
                user_input=user_input,
                model_name=model_name,
                request=request,
                user_id=user_id,
                request_id=request_id,
                **model_params,
            ):
                yield chunk

            self._update_usage_stats()
        except Exception as e:
            self._handle_error("stream_conversation", e)
            raise e

    async def stream_explanation(
        self,
        question: str,
        model_name: Optional[str] = None,
        request: Optional[Request] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        request_timeout: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream an explanation response using Server-Sent Events.

        Args:
            question: Question to explain
            model_name: Optional model name override
            request: FastAPI request object for disconnect detection

        Returns:
            AsyncGenerator yielding SSE-formatted bytes
        """
        # Import here to avoid circular imports
        from inference_core.llm.streaming import stream_explanation

        self._log_request(
            "stream_explanation",
            {
                "question": question[:128],
                "model_name": model_name,
            },
        )

        try:
            # Build model parameters
            model_params = {}
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

            async for chunk in stream_explanation(
                question=question,
                model_name=model_name,
                request=request,
                user_id=user_id,
                request_id=request_id,
                **model_params,
            ):
                yield chunk

            self._update_usage_stats()
        except Exception as e:
            self._handle_error("stream_explanation", e)
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

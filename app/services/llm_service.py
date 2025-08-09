"""
LLM Service Module

Main service class that provides high-level interface for all LLM operations.
This is the primary entry point for the API to interact with LLM capabilities.
"""

import logging
from datetime import UTC, datetime
from typing import Any, Dict, Optional, cast

from pydantic import BaseModel

from app.llm.chains import create_conversation_chain, create_explanation_chain
from app.llm.config import llm_config
from app.llm.models import get_model_factory

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
        try:
            model_params = {
                k: v
                for k, v in {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "timeout": request_timeout,
                }.items()
                if v is not None
            }
            chain = create_explanation_chain(model_name=model_name, **model_params)
            answer = await chain.generate_story(question=question)

            result = LLMResponse(
                result={"answer": answer},
                metadata=LLMMetadata(
                    model_name=chain.model_name,
                    timestamp=datetime.now(UTC).isoformat(),
                ),
            )

            self._update_usage_stats()
            return result
        except Exception as e:
            self._handle_error("explain", e)
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
        try:
            # Map request_timeout to factory's expected 'timeout'
            model_params = {
                k: v
                for k, v in {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "timeout": request_timeout,
                }.items()
                if v is not None
            }

            chain = create_conversation_chain(model_name=model_name, **model_params)
            reply = await chain.chat(session_id=session_id, user_input=user_input)

            result = LLMResponse(
                result={"reply": reply, "session_id": session_id},
                metadata=LLMMetadata(
                    model_name=chain.model_name,
                    timestamp=datetime.now(UTC).isoformat(),
                ),
            )
            self._update_usage_stats()
            return result
        except Exception as e:
            self._handle_error("conversation", e)
            raise e

    def get_available_models(self) -> Dict[str, bool]:
        """Get list of available models"""
        result = self.model_factory.get_available_models()
        return cast(Dict[str, bool], result)

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        result = self._usage_stats.copy()
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

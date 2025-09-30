"""
Celery tasks for LLM operations (explain, conversation)
"""

import asyncio
from typing import Any, Dict, Optional
from uuid import uuid4

from inference_core.celery.celery_main import celery_app
from inference_core.services.llm_service import get_llm_service


@celery_app.task(name="llm.explain")
def task_llm_explain(**kwargs) -> Dict[str, Any]:
    """Celery task: Generate explanation using LLMService"""

    async def _run(task_id: str) -> Dict[str, Any]:
        service = get_llm_service()
        question: str = kwargs.get("question", "")
        model_name: Optional[str] = kwargs.get("model_name")
        user_id: Optional[str] = kwargs.get("user_id")

        # Pass through optional model params if present
        result = await service.explain(
            question=question,
            model_name=model_name,
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            top_p=kwargs.get("top_p"),
            frequency_penalty=kwargs.get("frequency_penalty"),
            presence_penalty=kwargs.get("presence_penalty"),
            request_timeout=kwargs.get("request_timeout"),
            user_id=user_id,
            request_id=task_id,
        )
        return result.model_dump()

    # self-request id available via current task
    current = task_llm_explain.request  # type: ignore[attr-defined]
    return asyncio.run(_run(getattr(current, "id", None) or ""))


@celery_app.task(name="llm.conversation")
def task_llm_conversation(**kwargs) -> Dict[str, Any]:
    """Celery task: Conduct a conversation turn using LLMService"""

    async def _run(task_id: str) -> Dict[str, Any]:
        service = get_llm_service()
        session_id: Optional[str] = kwargs.get("session_id") or str(uuid4())
        user_input: str = kwargs.get("user_input", "")
        model_name: Optional[str] = kwargs.get("model_name")
        user_id: Optional[str] = kwargs.get("user_id")

        result = await service.converse(
            session_id=session_id,
            user_input=user_input,
            model_name=model_name,
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            top_p=kwargs.get("top_p"),
            frequency_penalty=kwargs.get("frequency_penalty"),
            presence_penalty=kwargs.get("presence_penalty"),
            request_timeout=kwargs.get("request_timeout"),
            user_id=user_id,
            request_id=task_id,
        )
        data = result.model_dump()
        # Ensure session_id is present in the result payload for clients
        data.setdefault("result", {})
        data["result"]["session_id"] = session_id
        return data

    current = task_llm_conversation.request  # type: ignore[attr-defined]
    return asyncio.run(_run(getattr(current, "id", None) or ""))

"""Celery tasks for LLM operations (completion, chat).

Refactored to use a shared persistent worker event loop via
`run_in_worker_loop` helper to avoid per-task loop creation and the
"event loop is closed" / cross-loop issues with async resources.
"""

from typing import Any, Dict, Optional
from uuid import uuid4

from inference_core.celery.async_utils import run_in_worker_loop
from inference_core.celery.celery_main import celery_app
from inference_core.services.llm_service import get_llm_service


@celery_app.task(name="llm.completion")
def task_llm_completion(**kwargs) -> Dict[str, Any]:
    """Generate completion using LLMService executed in the worker loop."""

    current = task_llm_completion.request  # type: ignore[attr-defined]
    task_id = getattr(current, "id", None) or ""

    async def _run() -> Dict[str, Any]:
        service = get_llm_service()
        prompt: str = kwargs.get("prompt") or kwargs.get("question", "")
        model_name: Optional[str] = kwargs.get("model_name")
        user_id: Optional[str] = kwargs.get("user_id")

        result = await service.completion(
            prompt=prompt,
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

    return run_in_worker_loop(_run())


@celery_app.task(name="llm.chat")
def task_llm_chat(**kwargs) -> Dict[str, Any]:
    """Run a chat turn using LLMService on the worker event loop."""

    current = task_llm_chat.request  # type: ignore[attr-defined]
    task_id = getattr(current, "id", None) or ""

    async def _run() -> Dict[str, Any]:
        service = get_llm_service()
        session_id: Optional[str] = kwargs.get("session_id") or str(uuid4())
        user_input: str = kwargs.get("user_input", "")
        model_name: Optional[str] = kwargs.get("model_name")
        user_id: Optional[str] = kwargs.get("user_id")

        result = await service.chat(
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
        data.setdefault("result", {})
        data["result"]["session_id"] = session_id
        return data

    return run_in_worker_loop(_run())

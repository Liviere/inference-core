"""
End-to-end tests: FastAPI API -> Celery task submission -> Task result retrieval.

- Uses Celery eager mode with in-memory backend to avoid external broker/worker.
- Stubs LLM chains to avoid network calls.
"""

from __future__ import annotations

import re
import types
from typing import Any, Dict, Optional

import pytest

from inference_core.celery.celery_main import celery_app
from inference_core.services.task_service import TaskService

# Local in-memory store for eager results keyed by task_id
_EAGER_RESULTS: Dict[str, Any] = {}


class _FakeCompletionChain:
    def __init__(self, model_name: Optional[str] = None, **model_params: Any):
        self.model_name = model_name or "e2e-completion"
        self.model_params = model_params

    async def generate_story(self, *, question: str) -> str:
        return f"[E2E completion:{self.model_name}] {question}"


class _FakeChatChain:
    def __init__(self, model_name: Optional[str] = None, **model_params: Any):
        self.model_name = model_name or "e2e-conv"
        self.model_params = model_params

    async def chat(self, *, session_id: str, user_input: str) -> str:
        return f"[E2E reply:{self.model_name}:{session_id}] {user_input}"


@pytest.fixture()
def celery_eager():
    """Configure Celery to run tasks eagerly and store results in-memory."""
    # Force-load tasks module so Celery registers task names
    import inference_core.celery.tasks.llm_tasks  # noqa: F401
    from inference_core.celery.celery_main import celery_app

    prev_always = celery_app.conf.task_always_eager
    prev_prop = celery_app.conf.task_eager_propagates
    prev_backend = celery_app.conf.result_backend
    prev_store = getattr(celery_app.conf, "task_store_eager_result", None)

    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    # We don't rely on Celery backend in tests (we patch TaskService), so backend can stay as-is
    celery_app.conf.task_store_eager_result = True

    try:
        yield
    finally:
        celery_app.conf.task_always_eager = prev_always
        celery_app.conf.task_eager_propagates = prev_prop
        celery_app.conf.result_backend = prev_backend
        if prev_store is not None:
            celery_app.conf.task_store_eager_result = prev_store


class TestE2ELLMApiCelry:

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_e2e_completion_task(
        self, public_access_async_client, monkeypatch, celery_eager
    ):
        # Stub chain factories in llm_service
        import inference_core.celery.tasks.llm_tasks as llm_tasks
        import inference_core.services.llm_service as llm_svc

        monkeypatch.setattr(
            llm_svc,
            "create_completion_chain",
            lambda model_name=None, **kwargs: _FakeCompletionChain(
                model_name=model_name, **kwargs
            ),
        )

        # Patch TaskService to submit via registered task (respects eager mode)
        def _completion_async(self: TaskService, **kwargs):
            task = celery_app.tasks["llm.completion"]
            res = task.apply_async(kwargs=kwargs)
            _EAGER_RESULTS[res.id] = res
            return res.id

        monkeypatch.setattr(
            TaskService, "completion_async", _completion_async, raising=True
        )

        # Patch task.run to avoid nested asyncio.run during eager execution
        def _fake_completion_run(self, *args, **kwargs):
            question = kwargs.get("question", "")
            model_name = kwargs.get("model_name") or "demo-e2e"
            return {
                "result": {"answer": f"[E2E completion:{model_name}] {question}"},
                "metadata": {
                    "model_name": model_name,
                    "timestamp": "2025-01-01T00:00:00Z",
                },
            }

        # Bind as method on the task instance so `self` is passed
        bound_explain = types.MethodType(
            _fake_completion_run, celery_app.tasks["llm.completion"]
        )
        monkeypatch.setattr(
            celery_app.tasks["llm.completion"], "run", bound_explain, raising=True
        )

        # Also patch TaskService result/status to read from eager store instead of backend
        original_get_result = TaskService.get_task_result
        original_get_status = TaskService.get_task_status

        def _get_task_result(
            self: TaskService, task_id: str, timeout: Optional[float] = None
        ):
            res = _EAGER_RESULTS.get(task_id)
            if res is not None:
                return res.get(timeout=timeout)
            return original_get_result(self, task_id, timeout=timeout)

        def _get_task_status(self: TaskService, task_id: str):
            res = _EAGER_RESULTS.get(task_id)
            if res is not None:
                ready = res.ready()
                return {
                    "task_id": task_id,
                    "status": "SUCCESS" if ready else "PENDING",
                    "result": res.result if ready else None,
                    "info": None,
                    "traceback": None,
                    "successful": res.successful() if ready else None,
                    "failed": False if ready else None,
                }
            return original_get_status(self, task_id)

        monkeypatch.setattr(
            TaskService, "get_task_result", _get_task_result, raising=True
        )
        monkeypatch.setattr(
            TaskService, "get_task_status", _get_task_status, raising=True
        )

        # Submit task via API
        resp = await public_access_async_client.post(
            "/api/v1/llm/completion",
            json={"question": "What is E2E?", "model_name": "demo-e2e"},
        )
        assert resp.status_code == 200
        task = resp.json()
        assert task["task_id"] and task["status"] == "PENDING"

        # Fetch result via API with timeout to avoid blocking on errors
        res = await public_access_async_client.get(
            f"/api/v1/tasks/{task['task_id']}/result", params={"timeout": 10.0}
        )
        assert res.status_code == 200
        payload = res.json()

        assert payload.get("success") is True
        result = payload.get("result", {})
        # Task returns a dict with keys: result, metadata
        assert "result" in result and "metadata" in result
        assert result["result"]["answer"].startswith("[E2E completion:demo-e2e]")
        meta = result["metadata"]
        assert meta["model_name"] == "demo-e2e"
        assert re.match(r"^\d{4}-\d{2}-\d{2}T", meta["timestamp"]) is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_e2e_chat_task(
        self, public_access_async_client, monkeypatch, celery_eager
    ):
        import inference_core.celery.tasks.llm_tasks as llm_tasks
        import inference_core.services.llm_service as llm_svc

        monkeypatch.setattr(
            llm_svc,
            "create_chat_chain",
            lambda model_name=None, **kwargs: _FakeChatChain(
                model_name=model_name, **kwargs
            ),
        )

        # Patch TaskService to submit via registered task (respects eager mode)
        def _chat_async(self: TaskService, **kwargs):
            task = celery_app.tasks["llm.chat"]
            res = task.apply_async(kwargs=kwargs)
            _EAGER_RESULTS[res.id] = res
            return res.id

        monkeypatch.setattr(TaskService, "chat_async", _chat_async, raising=True)

        # Patch task.run to avoid nested asyncio.run during eager execution
        def _fake_chat_run(self, *args, **kwargs):
            session_id = kwargs.get("session_id") or "gen-session"
            user_input = kwargs.get("user_input", "")
            model_name = kwargs.get("model_name") or "conv-e2e"
            return {
                "result": {
                    "reply": f"[E2E reply:{model_name}:{session_id}] {user_input}",
                    "session_id": session_id,
                },
                "metadata": {
                    "model_name": model_name,
                    "timestamp": "2025-01-01T00:00:00Z",
                },
            }

        # Bind as method on the task instance so `self` is passed
        bound_conv = types.MethodType(_fake_chat_run, celery_app.tasks["llm.chat"])
        monkeypatch.setattr(
            celery_app.tasks["llm.chat"], "run", bound_conv, raising=True
        )

        # Also patch TaskService result/status to read from eager store
        original_get_result = TaskService.get_task_result
        original_get_status = TaskService.get_task_status

        def _get_task_result(
            self: TaskService, task_id: str, timeout: Optional[float] = None
        ):
            res = _EAGER_RESULTS.get(task_id)
            if res is not None:
                return res.get(timeout=timeout)
            return original_get_result(self, task_id, timeout=timeout)

        def _get_task_status(self: TaskService, task_id: str):
            res = _EAGER_RESULTS.get(task_id)
            if res is not None:
                ready = res.ready()
                return {
                    "task_id": task_id,
                    "status": "SUCCESS" if ready else "PENDING",
                    "result": res.result if ready else None,
                    "info": None,
                    "traceback": None,
                    "successful": res.successful() if ready else None,
                    "failed": False if ready else None,
                }
            return original_get_status(self, task_id)

        monkeypatch.setattr(
            TaskService, "get_task_result", _get_task_result, raising=True
        )
        monkeypatch.setattr(
            TaskService, "get_task_status", _get_task_status, raising=True
        )

        # Submit task via API
        session_id = "e2e-session-42"
        user_input = "Hi E2E"
        resp = await public_access_async_client.post(
            "/api/v1/llm/chat",
            json={
                "session_id": session_id,
                "user_input": user_input,
                "model_name": "conv-e2e",
            },
        )
        assert resp.status_code == 200
        task = resp.json()
        assert task["task_id"] and task["status"] == "PENDING"

        # Fetch status (should be SUCCESS in eager mode)
        status_res = await public_access_async_client.get(
            f"/api/v1/tasks/{task['task_id']}/status"
        )
        assert status_res.status_code == 200
        status_payload = status_res.json()
        assert status_payload["status"] in {"SUCCESS", "PENDING", "STARTED", "RETRY"}

        # Fetch final result with timeout to avoid blocking
        res = await public_access_async_client.get(
            f"/api/v1/tasks/{task['task_id']}/result", params={"timeout": 10.0}
        )
        assert res.status_code == 200
        payload = res.json()
        assert payload.get("success") is True
        result = payload.get("result", {})

        # Result is the task return payload with reply and metadata
        assert result["result"]["reply"].startswith(
            f"[E2E reply:conv-e2e:{session_id}]"
        )
        assert result["result"]["session_id"] == session_id
        meta = result["metadata"]
        assert meta["model_name"] == "conv-e2e"
        assert re.match(r"^\d{4}-\d{2}-\d{2}T", meta["timestamp"]) is not None

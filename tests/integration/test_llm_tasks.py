import re
from typing import Any, Dict, Optional

import pytest

# Import task functions under test
from inference_core.celery.tasks.llm_tasks import task_llm_chat, task_llm_completion


class _FakeCompletionChain:
    """Minimal fake for create_completion_chain output."""

    def __init__(self, model_name: Optional[str] = None, **model_params: Any):
        self.model_name = model_name or "fake-completion-model"
        self.model_params = model_params

    async def completion(self, *, prompt: str, callbacks=None) -> str:
        return f"[completion:{self.model_name}] {prompt}"


class _FakeChatChain:
    """Minimal fake for create_chat_chain output."""

    def __init__(self, model_name: Optional[str] = None, **model_params: Any):
        self.model_name = model_name or "fake-conv-model"
        self.model_params = model_params

    async def chat(self, *, session_id: str, user_input: str, callbacks=None) -> str:
        return f"[reply:{self.model_name}:{session_id}] {user_input}"


@pytest.mark.integration
def test_task_llm_completion_basic(monkeypatch):
    """Completion task returns structured payload with answer and metadata."""

    # Patch the chain factory inside llm_service module namespace
    import inference_core.services.llm_service as llm_svc

    def _fake_create_completion_chain(model_name: Optional[str] = None, **kwargs: Any):
        return _FakeCompletionChain(model_name=model_name, **kwargs)

    monkeypatch.setattr(
        llm_svc, "create_completion_chain", _fake_create_completion_chain
    )

    question = "Why is the sky blue?"
    out: Dict[str, Any] = task_llm_completion(
        question=question, model_name="demo-model", temperature=0.2
    )

    # Validate structure
    assert isinstance(out, dict)
    assert "result" in out and "metadata" in out

    # Result assertions
    assert out["result"]["answer"] == f"[completion:demo-model] {question}"

    # Metadata assertions
    meta = out["metadata"]
    assert meta["model_name"] == "demo-model"
    assert isinstance(meta["timestamp"], str)
    # ISO timestamp basic shape
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", meta["timestamp"]) is not None


@pytest.mark.integration
def test_task_llm_chat_with_session_id(monkeypatch):
    """Chat task echoes reply and preserves provided session_id."""

    import inference_core.services.llm_service as llm_svc

    def _fake_create_chat_chain(model_name: Optional[str] = None, **kwargs: Any):
        return _FakeChatChain(model_name=model_name, **kwargs)

    monkeypatch.setattr(llm_svc, "create_chat_chain", _fake_create_chat_chain)

    session_id = "test-session-123"
    user_input = "Hello there"

    out: Dict[str, Any] = task_llm_chat(
        session_id=session_id,
        user_input=user_input,
        model_name="chat-model",
        max_tokens=64,
    )

    # Validate structure
    assert isinstance(out, dict)
    assert "result" in out and "metadata" in out

    # Result assertions
    result = out["result"]
    assert result["reply"] == f"[reply:chat-model:{session_id}] {user_input}"
    assert result["session_id"] == session_id

    # Metadata
    meta = out["metadata"]
    assert meta["model_name"] == "chat-model"
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", meta["timestamp"]) is not None


@pytest.mark.integration
def test_task_llm_chat_autogenerates_session_id(monkeypatch):
    """Chat task fills session_id when not provided."""

    import inference_core.services.llm_service as llm_svc

    def _fake_create_chat_chain(model_name: Optional[str] = None, **kwargs: Any):
        return _FakeChatChain(model_name=model_name, **kwargs)

    monkeypatch.setattr(llm_svc, "create_chat_chain", _fake_create_chat_chain)

    user_input = "How are you?"

    out: Dict[str, Any] = task_llm_chat(user_input=user_input)

    # Ensure a session_id is present
    result = out["result"]
    assert isinstance(result.get("session_id"), str)
    assert len(result["session_id"]) > 0

    # And reply format is consistent
    # We don't know the generated session_id; just check prefix
    assert result["reply"].endswith(user_input)
    assert "[reply:" in result["reply"]

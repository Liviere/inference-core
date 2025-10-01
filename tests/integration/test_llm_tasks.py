import re
from typing import Any, Dict, Optional

import pytest

# Import task functions under test
from inference_core.celery.tasks.llm_tasks import (
    task_llm_conversation,
    task_llm_explain,
)


class _FakeExplanationChain:
    """Minimal fake for create_explanation_chain output."""

    def __init__(self, model_name: Optional[str] = None, **model_params: Any):
        self.model_name = model_name or "fake-explain-model"
        self.model_params = model_params

    async def generate_story(self, *, question: str, callbacks=None) -> str:
        return f"[explained:{self.model_name}] {question}"


class _FakeConversationChain:
    """Minimal fake for create_conversation_chain output."""

    def __init__(self, model_name: Optional[str] = None, **model_params: Any):
        self.model_name = model_name or "fake-conv-model"
        self.model_params = model_params

    async def chat(self, *, session_id: str, user_input: str, callbacks=None) -> str:
        return f"[reply:{self.model_name}:{session_id}] {user_input}"


@pytest.mark.integration
def test_task_llm_explain_basic(monkeypatch):
    """Explain task returns structured payload with answer and metadata."""

    # Patch the chain factory inside llm_service module namespace
    import inference_core.services.llm_service as llm_svc

    def _fake_create_explanation_chain(model_name: Optional[str] = None, **kwargs: Any):
        return _FakeExplanationChain(model_name=model_name, **kwargs)

    monkeypatch.setattr(
        llm_svc, "create_explanation_chain", _fake_create_explanation_chain
    )

    question = "Why is the sky blue?"
    out: Dict[str, Any] = task_llm_explain(
        question=question, model_name="demo-model", temperature=0.2
    )

    # Validate structure
    assert isinstance(out, dict)
    assert "result" in out and "metadata" in out

    # Result assertions
    assert out["result"]["answer"] == f"[explained:demo-model] {question}"

    # Metadata assertions
    meta = out["metadata"]
    assert meta["model_name"] == "demo-model"
    assert isinstance(meta["timestamp"], str)
    # ISO timestamp basic shape
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", meta["timestamp"]) is not None


@pytest.mark.integration
def test_task_llm_conversation_with_session_id(monkeypatch):
    """Conversation task echoes reply and preserves provided session_id."""

    import inference_core.services.llm_service as llm_svc

    def _fake_create_conversation_chain(
        model_name: Optional[str] = None, **kwargs: Any
    ):
        return _FakeConversationChain(model_name=model_name, **kwargs)

    monkeypatch.setattr(
        llm_svc, "create_conversation_chain", _fake_create_conversation_chain
    )

    session_id = "test-session-123"
    user_input = "Hello there"

    out: Dict[str, Any] = task_llm_conversation(
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
def test_task_llm_conversation_autogenerates_session_id(monkeypatch):
    """Conversation task fills session_id when not provided."""

    import inference_core.services.llm_service as llm_svc

    def _fake_create_conversation_chain(
        model_name: Optional[str] = None, **kwargs: Any
    ):
        return _FakeConversationChain(model_name=model_name, **kwargs)

    monkeypatch.setattr(
        llm_svc, "create_conversation_chain", _fake_create_conversation_chain
    )

    user_input = "How are you?"

    out: Dict[str, Any] = task_llm_conversation(user_input=user_input)

    # Ensure a session_id is present
    result = out["result"]
    assert isinstance(result.get("session_id"), str)
    assert len(result["session_id"]) > 0

    # And reply format is consistent
    # We don't know the generated session_id; just check prefix
    assert result["reply"].endswith(user_input)
    assert "[reply:" in result["reply"]

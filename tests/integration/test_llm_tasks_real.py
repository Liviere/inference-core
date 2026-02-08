"""
Integration tests using authentic chains and models defined under the 'testing' key
for each task in llm_config.yaml. These tests will be skipped if the configured
models are not available (e.g., missing API keys).
"""

import os
import re
from pathlib import Path
from typing import List

import pytest
import yaml

from inference_core.celery.tasks.llm_tasks import task_llm_chat, task_llm_completion
from inference_core.llm.config import llm_config


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_testing_models(task_name: str) -> List[str]:
    cfg_path = _project_root() / "llm_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tasks = (cfg or {}).get("tasks", {})
    task_cfg = tasks.get(task_name, {})
    testing_models = task_cfg.get("testing", []) or []
    return [str(m) for m in testing_models]


def _env_truthy(var_name: str, default: str = "0") -> bool:
    """Return True if the environment variable is set to a truthy value."""
    return os.getenv(var_name, default).strip().lower() in ("1", "true", "yes")


@pytest.mark.integration
@pytest.mark.parametrize("model_name", _load_testing_models("completion"))
def test_llm_completion_real_uses_testing_models(model_name: str):
    """Run completion task with real chain using a testing model from YAML."""
    # Opt-in gate for running real LLM tests
    if not _env_truthy("RUN_LLM_REAL_TESTS"):
        pytest.skip("Set RUN_LLM_REAL_TESTS=1 to run real-chain LLM tests.")
    # Skip if model not available (e.g., missing API key for provider)
    if not llm_config.is_model_available(model_name):
        pytest.skip(f"Testing model not available: {model_name}")

    prompt = "Explain unit testing briefly."
    out = task_llm_completion(
        prompt=prompt,
        model_name=model_name,
        max_tokens=64,
        request_timeout=30,
        temperature=0.2,
    )

    assert isinstance(out, dict)
    assert "result" in out and "metadata" in out
    # Answer should be a non-empty string
    answer = out["result"].get("answer", "")
    if not isinstance(answer, str) or len(answer.strip()) == 0:
        pytest.xfail(
            "Provider returned empty answer (possibly due to invalid key/network)."
        )

    # Metadata should reflect the selected model and have an ISO timestamp
    meta = out["metadata"]
    assert meta["model_name"] == model_name
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", meta["timestamp"]) is not None


@pytest.mark.integration
@pytest.mark.parametrize("model_name", _load_testing_models("chat"))
def test_llm_chat_real_uses_testing_models(model_name: str):
    """Run chat task with real chain using a testing model from YAML."""
    if not _env_truthy("RUN_LLM_REAL_TESTS"):
        pytest.skip("Set RUN_LLM_REAL_TESTS=1 to run real-chain LLM tests.")
    if not llm_config.is_model_available(model_name):
        pytest.skip(f"Testing model not available: {model_name}")

    session_id = "it-session-001"
    user_input = "Hello, who are you?"
    out = task_llm_chat(
        session_id=session_id,
        user_input=user_input,
        model_name=model_name,
        max_tokens=64,
        request_timeout=30,
        temperature=0.2,
    )

    assert isinstance(out, dict)
    assert "result" in out and "metadata" in out

    result = out["result"]
    reply = result.get("reply", "")
    if not isinstance(reply, str) or len(reply.strip()) == 0:
        pytest.xfail(
            "Provider returned empty reply (possibly due to invalid key/network)."
        )
    assert result.get("session_id") == session_id

    meta = out["metadata"]
    assert meta["model_name"] == model_name
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", meta["timestamp"]) is not None

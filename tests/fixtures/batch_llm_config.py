"""
A small, self-contained LLM configuration for batch API tests.

WHY: batch job creation validates provider, model, batch enablement and mode
against the loaded LLM configuration. Reading the environment's
``llm_config.yaml`` makes the tests depend on whatever models a developer
happens to have configured (and fail in CI, where the file does not exist),
so batch tests load this YAML instead.
"""

import textwrap

import pytest

from inference_core.llm.config import LLMConfig

# Batch-enabled OpenAI chat model.
BATCH_CHAT_MODEL = "test-batch-chat"
# OpenAI model that exists but is not configured for batch.
NON_BATCH_MODEL = "test-chat-no-batch"
# Model that belongs to Gemini, used to request it under the wrong provider.
GEMINI_MODEL = "test-gemini-chat"
# Model of a provider that has no batch section at all.
DEEPINFRA_MODEL = "test-deepinfra-chat"

_BATCH_LLM_CONFIG_YAML = f"""
providers:
  openai:
    name: OpenAI
    api_key_env: OPENAI_API_KEY
    requires_api_key: true
  gemini:
    name: Google Gemini
    api_key_env: GOOGLE_API_KEY
    requires_api_key: true
  deepinfra:
    name: DeepInfra
    api_key_env: DEEPINFRA_API_TOKEN
    requires_api_key: true
models:
  {BATCH_CHAT_MODEL}:
    provider: openai
  {NON_BATCH_MODEL}:
    provider: openai
  {GEMINI_MODEL}:
    provider: gemini
  {DEEPINFRA_MODEL}:
    provider: deepinfra
agents: {{}}
batch:
  enabled: true
  providers:
    openai:
      enabled: true
      models:
        - name: {BATCH_CHAT_MODEL}
          mode: chat
          max_prompts_per_batch: 20
"""


@pytest.fixture
def batch_llm_config(tmp_path, monkeypatch) -> LLMConfig:
    """Load the batch test configuration and make it the active one.

    WHY: the batch route binds ``llm_config`` at import time while Celery
    batch tasks call ``get_llm_config()``, so both references are replaced
    for the duration of the test.
    """
    from inference_core.api.v1.routes import batch as batch_routes
    from inference_core.llm import config as llm_config_module

    path = tmp_path / "llm_config.yaml"
    path.write_text(textwrap.dedent(_BATCH_LLM_CONFIG_YAML), encoding="utf-8")
    monkeypatch.setenv("LLM_CONFIG_PATH", str(path))
    config = LLMConfig()

    monkeypatch.setattr(llm_config_module, "llm_config", config)
    monkeypatch.setattr(batch_routes, "llm_config", config)
    return config

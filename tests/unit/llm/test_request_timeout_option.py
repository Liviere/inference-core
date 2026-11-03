"""The request timeout a provider's policy drops, handed to its client on request.

Gemini's policy drops ``request_timeout``. ``give_request_timeout`` hands it to
the client under the name the client takes; without the call nothing changes.
"""

import asyncio
import socket
import threading
import time

import pytest

from inference_core.llm import param_policy
from inference_core.llm.config import ModelConfig, ModelProvider
from inference_core.llm.models import LLMModelFactory
from inference_core.llm.param_policy import (
    give_request_timeout,
    normalize_params,
    takes_request_timeout,
    withdraw_request_timeout,
)


@pytest.fixture(autouse=True)
def _no_timeout_given():
    withdraw_request_timeout(ModelProvider.GEMINI)
    yield
    withdraw_request_timeout(ModelProvider.GEMINI)


@pytest.fixture(autouse=True)
def _disable_llm_emulation(monkeypatch):
    monkeypatch.setattr(
        "inference_core.llm.models.is_llm_emulation_enabled", lambda: False
    )


def _gemini_config(**fields) -> ModelConfig:
    return ModelConfig(
        name="gemini-test-flash",
        provider=ModelProvider.GEMINI,
        api_key="test-key",
        **fields,
    )


def _factory() -> LLMModelFactory:
    from unittest.mock import MagicMock

    config = MagicMock()
    config.enable_caching = False
    return LLMModelFactory(config)


class TestByDefault:
    def test_gemini_drops_the_timeout_as_it_always_did(self):
        assert normalize_params(ModelProvider.GEMINI, {"request_timeout": 30}) == {}
        assert not takes_request_timeout(ModelProvider.GEMINI)

    def test_the_providers_that_pass_it_are_said_to_take_it(self):
        for provider in (
            ModelProvider.OPENAI,
            ModelProvider.CLAUDE,
            ModelProvider.MISTRAL,
            ModelProvider.DEEPINFRA,
        ):
            assert takes_request_timeout(provider), provider

    def test_a_gemini_model_is_built_without_one(self):
        model = _factory()._create_model_instance(_gemini_config(timeout=7))
        assert model is not None
        assert model.timeout is None


class TestGiven:
    def test_the_timeout_reaches_gemini_as_its_client_names_it(self):
        give_request_timeout(ModelProvider.GEMINI)
        assert normalize_params(ModelProvider.GEMINI, {"request_timeout": 30}) == {
            "timeout": 30
        }
        assert takes_request_timeout(ModelProvider.GEMINI)

    def test_the_rest_of_the_policy_stays(self):
        give_request_timeout(ModelProvider.GEMINI)
        params = normalize_params(
            ModelProvider.GEMINI,
            {"max_tokens": 100, "frequency_penalty": 0.1, "request_timeout": 5},
        )
        assert params == {"max_output_tokens": 100, "timeout": 5}
        # The built-in policy itself is not changed.
        assert "request_timeout" in param_policy.POLICIES[ModelProvider.GEMINI].dropped

    def test_a_gemini_model_is_built_with_its_entrys_timeout(self):
        give_request_timeout(ModelProvider.GEMINI)
        model = _factory()._create_model_instance(_gemini_config(timeout=7))
        assert model is not None
        assert model.timeout == 7

    def test_withdrawn_it_is_dropped_again(self):
        give_request_timeout(ModelProvider.GEMINI)
        withdraw_request_timeout(ModelProvider.GEMINI)
        assert not takes_request_timeout(ModelProvider.GEMINI)

    def test_a_provider_that_passes_it_already_needs_nothing(self):
        give_request_timeout(ModelProvider.CLAUDE)
        assert normalize_params(ModelProvider.CLAUDE, {"request_timeout": 30}) == {
            "timeout": 30
        }

    def test_a_model_policy_resolved_before_is_resolved_again(self, monkeypatch):
        monkeypatch.setattr(param_policy, "_DYNAMIC_LOADED", True)
        monkeypatch.setattr(param_policy, "_EFFECTIVE_MODEL_POLICIES", {})
        placeholder = param_policy.ProviderParamPolicy()
        placeholder._raw_override = {"patch": {"allowed": ["thinking_level"]}}
        placeholder._provider = ModelProvider.GEMINI
        param_policy._EFFECTIVE_MODEL_POLICIES["gemini-test-flash"] = placeholder

        assert not takes_request_timeout(ModelProvider.GEMINI, "gemini-test-flash")
        give_request_timeout(ModelProvider.GEMINI)
        assert takes_request_timeout(ModelProvider.GEMINI, "gemini-test-flash")
        withdraw_request_timeout(ModelProvider.GEMINI)
        assert not takes_request_timeout(ModelProvider.GEMINI, "gemini-test-flash")

    def test_a_model_override_that_drops_every_timeout_is_seen(self, monkeypatch):
        monkeypatch.setattr(param_policy, "_DYNAMIC_LOADED", True)
        monkeypatch.setattr(param_policy, "_EFFECTIVE_MODEL_POLICIES", {})
        placeholder = param_policy.ProviderParamPolicy()
        placeholder._raw_override = {
            "replace": {"allowed": ["temperature"], "renamed": {}}
        }
        placeholder._provider = ModelProvider.GEMINI
        param_policy._EFFECTIVE_MODEL_POLICIES["gemini-test-flash"] = placeholder
        give_request_timeout(ModelProvider.GEMINI)

        assert takes_request_timeout(ModelProvider.GEMINI)
        assert not takes_request_timeout(ModelProvider.GEMINI, "gemini-test-flash")


class _HangingServer:
    """Accepts connections and never answers."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.sock.settimeout(0.2)
        self.port = self.sock.getsockname()[1]
        self.accepted = 0
        self._held = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                continue
            self.accepted += 1
            self._held.append(conn)

    def __enter__(self) -> "_HangingServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        for conn in self._held:
            conn.close()
        self.sock.close()


@pytest.fixture
def no_proxy(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


def _gemini_on(server: _HangingServer, max_retries: int):
    from langchain_google_genai import ChatGoogleGenerativeAI

    give_request_timeout(ModelProvider.GEMINI)
    params = normalize_params(ModelProvider.GEMINI, {"request_timeout": 1})
    return ChatGoogleGenerativeAI(
        model="gemini-test-flash",
        api_key="test-key",
        base_url=f"http://127.0.0.1:{server.port}",
        max_retries=max_retries,
        **params,
    )


class TestAgainstAServerThatNeverAnswers:
    def test_a_call_ends_at_the_timeout(self, no_proxy):
        with _HangingServer() as server:
            model = _gemini_on(server, max_retries=1)
            started = time.monotonic()
            with pytest.raises(Exception):
                model.invoke("hello")
            elapsed = time.monotonic() - started
        assert 0.9 <= elapsed < 5
        assert server.accepted == 1

    def test_an_async_call_ends_at_the_timeout(self, no_proxy):
        with _HangingServer() as server:
            model = _gemini_on(server, max_retries=1)
            started = time.monotonic()
            with pytest.raises(Exception):
                asyncio.run(model.ainvoke("hello"))
            elapsed = time.monotonic() - started
        assert 0.9 <= elapsed < 5
        assert server.accepted == 1

    def test_the_client_tries_again_up_to_its_retries(self, no_proxy):
        with _HangingServer() as server:
            model = _gemini_on(server, max_retries=2)
            started = time.monotonic()
            with pytest.raises(Exception):
                model.invoke("hello")
            elapsed = time.monotonic() - started
        # Two timeouts of a second and a backoff of about one between them.
        assert server.accepted == 2
        assert 1.9 <= elapsed < 10

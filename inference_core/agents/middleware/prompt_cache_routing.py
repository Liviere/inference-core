"""Requests of one conversation sent to the server that cached their prompt.

OpenAI, Mistral, Fireworks, DeepInfra and xAI cache the start of a prompt on
the server that answered it, by themselves. A provider runs many servers, so
the next request of the same conversation reads that cache only when it goes
to the same one. A key on the request asks for that: ``prompt_cache_key`` in
the body, or xAI's ``x-grok-conv-id`` header for Chat Completions. Without it,
xAI and Fireworks seldom read their cache (measured: 128 tokens of 5,137 for
xAI, 123 of 5,212 for Fireworks; with it, all but the last block).

A model entry asks for it with ``prompt_cache_routing: true`` (off by
default). :class:`PromptCacheRoutingMiddleware` puts the key on each model
call of an agent whose model asks for it. It sits inside the fallback
middleware, next to cost tracking, so it sees the model each attempt is made
on: a fallback of another provider gets its own key or none.

The key is a hash of the agent's thread and the model's name: the same for
every request of a conversation on one model, and nothing of the
conversation, the person or the account can be read from it.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Dict, Optional

from langchain.agents.middleware import AgentMiddleware

from inference_core.llm.model_identity import get_model_identity, provider_name

logger = logging.getLogger(__name__)

#: The providers whose requests take a key, and how: a field of the body or
#: an HTTP header.
BODY_KEY = "prompt_cache_key"
XAI_HEADER = "x-grok-conv-id"
ROUTED_PROVIDERS: Dict[str, tuple[str, str]] = {
    "openai": ("body", BODY_KEY),
    "mistral": ("body", BODY_KEY),
    "fireworks": ("body", BODY_KEY),
    "deepinfra": ("body", BODY_KEY),
    "xai": ("header", XAI_HEADER),
}


def routing_key(thread_id: str, model_name: str) -> str:
    """The key the requests of one thread on one model are sent with."""
    digest = hashlib.sha256(
        f"prompt-cache:{thread_id}:{model_name}".encode("utf-8")
    ).hexdigest()
    return digest[:32]


def routing_settings(provider: Optional[str], key: str) -> Dict[str, Any]:
    """What a request of *provider* carries the key in; empty for no provider."""
    how = ROUTED_PROVIDERS.get(provider or "")
    if how is None:
        return {}
    kind, name = how
    if kind == "header":
        return {"extra_headers": {name: key}}
    return {name: key}


def _thread_id() -> Optional[str]:
    """The thread of the run the model call belongs to, if there is one."""
    try:
        from langgraph.config import get_config

        configurable = (get_config() or {}).get("configurable") or {}
    except Exception:
        return None
    thread_id = configurable.get("thread_id")
    return str(thread_id) if thread_id else None


class PromptCacheRoutingMiddleware(AgentMiddleware):
    """Puts the routing key on the model calls whose model asks for it.

    *config* is the ``LLMConfig`` the agent's models were built from; a model
    is looked up there by the name it is tagged with
    (:func:`~inference_core.llm.model_identity.tag_model_identity`). A model
    without a name, without ``prompt_cache_routing``, of a provider that takes
    no key, or a call outside a thread, goes as it is.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._config = config
        self._warned: set[str] = set()

    @staticmethod
    def wanted(config: Any) -> bool:
        """Whether any model of *config* asks for the routing."""
        models = getattr(config, "models", None) or {}
        return any(
            getattr(model, "prompt_cache_routing", False) for model in models.values()
        )

    def _settings(self, request: Any) -> Dict[str, Any]:
        identity = get_model_identity(getattr(request, "model", None))
        if identity is None or not identity.name:
            return {}
        models = getattr(self._config, "models", None) or {}
        entry = models.get(identity.name)
        if entry is None or not getattr(entry, "prompt_cache_routing", False):
            return {}
        provider = provider_name(getattr(entry, "provider", None))
        if provider not in ROUTED_PROVIDERS:
            if identity.name not in self._warned:
                self._warned.add(identity.name)
                logger.warning(
                    "prompt_cache_routing is ignored for model '%s': its "
                    "provider takes no routing key",
                    identity.name,
                )
            return {}
        thread_id = _thread_id()
        if not thread_id:
            return {}
        return routing_settings(provider, routing_key(thread_id, identity.name))

    def _routed(self, request: Any) -> Any:
        settings = self._settings(request)
        if not settings:
            return request
        model_settings = dict(getattr(request, "model_settings", None) or {})
        headers = settings.pop("extra_headers", None)
        if headers:
            model_settings["extra_headers"] = {
                **(model_settings.get("extra_headers") or {}),
                **headers,
            }
        model_settings.update(settings)
        return request.override(model_settings=model_settings)

    def wrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        return handler(self._routed(request))

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        return await handler(self._routed(request))

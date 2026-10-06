"""Chat model providers registered by the host application.

The built-in providers are an enum with a fixed dispatch in the model factory.
An application that serves completions some other way (an in-house gateway, a
runtime on the user's own device, a model reached over a message bus)
registers a provider name here together with the function that builds its
chat model. From then on the name works wherever a built-in provider does: in
``llm_config.yaml``, in a ``ModelConfig`` added at runtime, as an agent's
primary model or as a fallback.

Register before the first model of that provider is configured: a model entry
naming an unknown provider is rejected, in YAML and at runtime alike.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.language_models.chat_models import BaseChatModel

    from .config import ModelConfig
    from .param_policy import ProviderParamPolicy

ChatModelBuilder = Callable[["ModelConfig", Dict[str, Any]], "BaseChatModel"]
"""Builds a chat model from its config and the normalized call parameters."""


@dataclass(frozen=True)
class RegisteredProvider:
    """A provider the application registered, with how to build its models."""

    name: str
    builder: ChatModelBuilder
    param_policy: Optional["ProviderParamPolicy"] = None


_REGISTRY: Dict[str, RegisteredProvider] = {}


def register_chat_model_provider(
    name: str,
    builder: ChatModelBuilder,
    *,
    param_policy: Optional["ProviderParamPolicy"] = None,
) -> None:
    """Make ``name`` a provider whose models ``builder`` creates.

    ``builder(config, params)`` receives the model's :class:`ModelConfig` and
    the parameters left after normalization, and returns the chat model.
    ``param_policy`` says which parameters reach it; without one only the
    common sampling parameters and the request timeout do.

    Registering the same name again replaces the earlier entry, so a module
    that registers at import time stays safe to import twice. A built-in
    provider name cannot be taken over.
    """
    from .config import ModelProvider  # local: config imports this module

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Provider name must be a non-empty string")
    name = name.strip()
    if name in {provider.value for provider in ModelProvider}:
        raise ValueError(f"'{name}' is a built-in provider and cannot be replaced")
    if not callable(builder):
        raise TypeError("Provider builder must be callable")
    _REGISTRY[name] = RegisteredProvider(
        name=name, builder=builder, param_policy=param_policy
    )


def unregister_chat_model_provider(name: str) -> None:
    """Forget a registered provider (a no-op for an unknown name)."""
    _REGISTRY.pop(name, None)


def get_registered_provider(name: Any) -> Optional[RegisteredProvider]:
    """The registration for ``name``, or None for a built-in or unknown one."""
    if not isinstance(name, str):
        return None
    return _REGISTRY.get(str(name))


def is_registered_provider(name: Any) -> bool:
    return get_registered_provider(name) is not None

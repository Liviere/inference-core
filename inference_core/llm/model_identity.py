"""Which configured model a chat-model object is.

A model is configured under a name of its own in the LLM config, and that name
is what pricing, provider keys and the logs go by. A provider's response does
not have to repeat it: it may name a dated snapshot, an alias or a served
version. And a chat-model class keeps the name it was built with under an
attribute of its own choosing. So the factory writes down, on the object it
builds, which configured model it is and whose provider serves it, and whoever
sees that object make a call can read it back.

The note lives in the model's ``metadata``, a field every chat model declares:
it travels with ``bind_tools`` and copies of the model, and it is not part of
the request sent to the provider.

Whoever builds a model may leave a second note on it: details to record with
the usage of every call made on it (:func:`tag_usage_details`). The usage logs
copy them as they are and read nothing into them.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional

logger = logging.getLogger(__name__)

_NAME_KEY = "inference_core.model_name"
_PROVIDER_KEY = "inference_core.model_provider"
_USAGE_DETAILS_KEY = "inference_core.usage_details"

# A model bound to tools is a binding around the model; this many layers are
# looked through before giving up.
_MAX_BINDING_DEPTH = 4


@dataclass(frozen=True)
class ModelIdentity:
    """The configured name of a model and the provider that serves it."""

    name: str
    provider: Optional[str] = None


def provider_name(provider: Any) -> Optional[str]:
    """A provider as its plain name, whether it came as an enum or a string."""
    if provider is None:
        return None
    value = getattr(provider, "value", provider)
    return str(value) if value else None


def tag_model_identity(model: Any, *, name: str, provider: Any = None) -> None:
    """Write on *model* which configured model it is.

    Never raises: a model that cannot carry the note is used without one, and
    whoever reads it falls back to what it knows otherwise.
    """
    if model is None or not name:
        return
    try:
        existing = getattr(model, "metadata", None)
        tagged = dict(existing) if isinstance(existing, dict) else {}
        tagged[_NAME_KEY] = str(name)
        resolved = provider_name(provider)
        if resolved:
            tagged[_PROVIDER_KEY] = resolved
        else:
            tagged.pop(_PROVIDER_KEY, None)
        model.metadata = tagged
    except Exception:
        logger.debug("Could not tag model %r with its identity", name, exc_info=True)


def _metadata_along_bindings(model: Any) -> Iterator[Dict[str, Any]]:
    """The ``metadata`` of *model* and of what it is a binding around.

    A model bound to tools is a binding around the model; the notes are on the
    model underneath. Never raises.
    """
    current = model
    for _ in range(_MAX_BINDING_DEPTH):
        if current is None:
            return
        try:
            metadata = getattr(current, "metadata", None)
        except Exception:
            return
        if isinstance(metadata, dict):
            yield metadata
        try:
            current = getattr(current, "bound", None)
        except Exception:
            return


def get_model_identity(model: Any) -> Optional[ModelIdentity]:
    """Read back what :func:`tag_model_identity` wrote, or ``None``.

    Looks through tool bindings to the model underneath. Never raises.
    """
    for metadata in _metadata_along_bindings(model):
        name = metadata.get(_NAME_KEY)
        if isinstance(name, str) and name:
            provider = metadata.get(_PROVIDER_KEY)
            return ModelIdentity(
                name=name,
                provider=provider if isinstance(provider, str) and provider else None,
            )
    return None


def tag_usage_details(model: Any, details: Optional[Mapping[str, Any]]) -> None:
    """Write on *model* details to record with the usage of every call on it.

    For whoever builds a model and knows something about it that its calls
    should carry into the usage log: what the model is used for, say, or on
    whose account it runs. Only plain values are kept (text, numbers, flags):
    the details end up in a JSON column and in the metadata of a traced run.

    Replaces details written before. Never raises.
    """
    if model is None:
        return
    try:
        plain = {
            str(key): value
            for key, value in (details or {}).items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
        existing = getattr(model, "metadata", None)
        tagged = dict(existing) if isinstance(existing, dict) else {}
        if plain:
            tagged[_USAGE_DETAILS_KEY] = plain
        else:
            tagged.pop(_USAGE_DETAILS_KEY, None)
        model.metadata = tagged
    except Exception:
        logger.debug("Could not tag a model with usage details", exc_info=True)


def get_usage_details(model: Any) -> Dict[str, Any]:
    """What :func:`tag_usage_details` wrote, as a dict of its own; else empty.

    Looks through tool bindings to the model underneath. Never raises.
    """
    for metadata in _metadata_along_bindings(model):
        details = metadata.get(_USAGE_DETAILS_KEY)
        if isinstance(details, dict):
            return dict(details)
    return {}


__all__ = [
    "ModelIdentity",
    "get_model_identity",
    "get_usage_details",
    "provider_name",
    "tag_model_identity",
    "tag_usage_details",
]

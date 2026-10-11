"""What a provider says about one of its models, in one shape for all of them.

Providers describe their models very differently: one returns little more than
an id, another a retirement date, a third a price list. A lister turns each
answer into a :class:`DiscoveredModel`. The fields the catalog acts on are
typed; everything that is only shown to an operator lives in ``attributes``.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

MAX_MODEL_ID_LENGTH = 255
"""Longest model id the catalog keeps; a longer one is not a model id."""


class ModelKind(str, Enum):
    """What a model is for, as far as its provider says."""

    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE = "image"
    AUDIO = "audio"
    OTHER = "other"
    UNKNOWN = "unknown"


class ModelLifecycle(str, Enum):
    """Where a model is in its life.

    ``UNKNOWN`` is for a provider whose listing carries no lifecycle data at
    all. It is not a warning: such a model is treated like an active one.
    """

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DiscoveredModel:
    """One model as its provider lists it.

    ``attributes`` may hold ``display_name``, ``created_at`` (ISO 8601),
    ``context_window``, ``max_output_tokens``, ``capabilities`` (``vision``,
    ``tools``, ``reasoning``; a key is absent when the provider does not say)
    and ``pricing`` (``input``, ``output``, ``cache_read`` in USD per one
    million tokens). All of it is the provider's own text and numbers.
    """

    provider: str
    model_id: str
    kind: ModelKind = ModelKind.UNKNOWN
    lifecycle: ModelLifecycle = ModelLifecycle.UNKNOWN
    deprecated_at: Optional[datetime] = None
    retires_at: Optional[datetime] = None
    replacement: Optional[str] = None
    aliases: Tuple[str, ...] = ()
    attributes: Dict[str, Any] = field(default_factory=dict)

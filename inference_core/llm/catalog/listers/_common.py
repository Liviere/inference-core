"""Helpers shared by the listers.

Everything a provider sends is read defensively: a field of the wrong type is
dropped, not trusted, and text is cut to a length an operator can read. A
listing whose outer shape is wrong is an error, because a listing must be
whole or not there at all.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

from ..http import CatalogResponseError
from ..types import MAX_MODEL_ID_LENGTH

MAX_TEXT_LENGTH = 200
MAX_ALIASES = 20

# Unix timestamps outside these years are some other kind of number.
_EARLIEST_TS = datetime(2015, 1, 1, tzinfo=timezone.utc).timestamp()
_LATEST_TS = datetime(2100, 1, 1, tzinfo=timezone.utc).timestamp()


def base_url(runtime: Dict[str, Any], default: str) -> str:
    """The provider's configured ``base_url``, or ``default``, without a trailing slash."""
    configured = runtime.get("base_url")
    url = configured if isinstance(configured, str) and configured.strip() else default
    return url.strip().rstrip("/")


def bearer_headers(runtime: Dict[str, Any]) -> Dict[str, str]:
    """``Authorization`` header for the provider's key, or none without a key."""
    api_key = runtime.get("api_key")
    if isinstance(api_key, str) and api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def path_segment(model_id: str) -> str:
    """``model_id`` as one URL path segment."""
    return quote(model_id, safe="")


def clean_model_id(value: Any) -> Optional[str]:
    """``value`` as a model id, or ``None`` when it cannot be one."""
    if not isinstance(value, str):
        return None
    model_id = value.strip()
    if not model_id or len(model_id) > MAX_MODEL_ID_LENGTH:
        return None
    if any(ch.isspace() or not ch.isprintable() for ch in model_id):
        return None
    return model_id


def clean_text(value: Any, max_length: int = MAX_TEXT_LENGTH) -> Optional[str]:
    """``value`` as one line of printable text, cut to ``max_length``."""
    if not isinstance(value, str):
        return None
    text = " ".join("".join(ch for ch in value if ch.isprintable()).split())
    return text[:max_length] or None


def clean_aliases(value: Any) -> Tuple[str, ...]:
    """The model ids in ``value``, when it is a list of them."""
    if not isinstance(value, list):
        return ()
    aliases = [alias for alias in map(clean_model_id, value[:MAX_ALIASES]) if alias]
    return tuple(dict.fromkeys(aliases))


def positive_int(value: Any) -> Optional[int]:
    """``value`` as a positive whole number, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value > 0 else None


def price(value: Any, per_million_factor: float) -> Optional[float]:
    """A provider's price as USD per one million tokens, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None
    return round(float(value) * per_million_factor, 6)


def from_unix(value: Any) -> Optional[datetime]:
    """A Unix timestamp in seconds as an aware datetime, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not _EARLIEST_TS <= value <= _LATEST_TS:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc)


def from_iso(value: Any) -> Optional[datetime]:
    """An ISO 8601 date or date-time as an aware datetime, or ``None``."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso(value: Optional[datetime]) -> Optional[str]:
    """``value`` as ISO 8601 text for ``attributes``."""
    return value.isoformat() if value else None


def compact(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """``attributes`` without the keys the provider said nothing about."""
    return {
        key: value
        for key, value in attributes.items()
        if value is not None and value != {} and value != []
    }


def require_list(
    payload: Any, key: Optional[str], *, may_be_absent: bool = False
) -> List[Any]:
    """The list of models in ``payload``; anything else is not a listing.

    ``may_be_absent`` is for APIs that leave an empty list out of the object.
    """
    if key is None:
        items = payload
    elif not isinstance(payload, dict):
        items = None
    elif may_be_absent and key not in payload:
        items = []
    else:
        items = payload.get(key)
    if not isinstance(items, list):
        raise CatalogResponseError("unexpected listing shape", status_code=200)
    return items


def optional_object(payload: Any) -> Optional[Dict[str, Any]]:
    """One model's object, ``None`` for "no such model"; anything else is an error."""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise CatalogResponseError("unexpected model shape", status_code=200)
    return payload


def dicts(items: Iterable[Any]) -> Iterable[Dict[str, Any]]:
    """The entries of a listing that are objects."""
    return (item for item in items if isinstance(item, dict))

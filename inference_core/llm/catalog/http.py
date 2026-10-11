"""Bounded HTTP for model listings.

A listing that comes back short must never be mistaken for a provider that
dropped models, so everything here fails loudly: a status other than 200, a
body over the size limit, a body that is not JSON. Errors carry the provider
and the status code and nothing else. They are stored and shown to operators,
and neither a URL nor a response body belongs there.
"""

import json
from typing import Any, Dict, Optional

import httpx

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
"""Largest listing page that is read. The biggest seen so far is about 0.5 MiB."""

MAX_PAGES = 50
"""Most pages one listing may have before it is taken for a loop."""

CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_READ_TIMEOUT_SECONDS = 20.0


class CatalogError(Exception):
    """A provider's model listing could not be read."""

    def __init__(self, message: str, *, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class CatalogHTTPError(CatalogError):
    """The provider answered with a status other than 200."""


class CatalogResponseError(CatalogError):
    """The provider answered 200 with something that is not a model listing."""


def build_client(
    read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
) -> httpx.AsyncClient:
    """Client for listing requests.

    Redirects are not followed: a request carries the provider's API key, and
    a listing endpoint has no reason to send it elsewhere. Proxy settings come
    from the environment, as for every other httpx client.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(read_timeout, connect=CONNECT_TIMEOUT_SECONDS),
        follow_redirects=False,
    )


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> Any:
    """GET ``url`` and return its JSON body.

    Raises :class:`CatalogHTTPError` for a status other than 200 and
    :class:`CatalogResponseError` for a body that is too large or not JSON.
    """
    async with client.stream("GET", url, headers=headers, params=params) as response:
        if response.status_code != 200:
            raise CatalogHTTPError(
                f"HTTP {response.status_code}", status_code=response.status_code
            )
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > max_bytes:
                raise CatalogResponseError("response too large", status_code=200)
    try:
        return json.loads(bytes(body))
    except ValueError as exc:
        raise CatalogResponseError("response is not JSON", status_code=200) from exc


async def get_json_or_none(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Like :func:`get_json`, with ``None`` for a 404.

    For looking one model up: "no such model" is an answer there, while any
    other failure still says nothing about the model.
    """
    try:
        return await get_json(client, url, headers=headers, params=params)
    except CatalogHTTPError as exc:
        if exc.status_code == 404:
            return None
        raise

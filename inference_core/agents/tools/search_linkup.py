"""Linkup as the provider behind ``internet_search``.

Takes the tool's arguments and answers in the shape Tavily's client does, with
``results`` of ``title``, ``url`` and ``content``, so what reads a result does
not depend on the provider.

The query is written by a model and may carry what the model has read, so it
never reaches a log or an error message of this module.

Environment:
  LINKUP_API_KEY   required.
  LINKUP_BASE_URL  default ``https://api.linkup.so/v1``.
  LINKUP_DEPTH     ``flash``, ``fast`` (default) or ``standard``. ``deep`` costs
                   ten times as much per query and is not offered.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import ToolException

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.linkup.so/v1"
DEFAULT_DEPTH = "fast"
DEPTHS = ("flash", "fast", "standard")

# Linkup's ``content`` can be far longer than a snippet. One result must not
# take over the model's context.
MAX_CONTENT_CHARS = 4000

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_TIME_RANGE_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


def _client() -> httpx.Client:
    """A client that reads the proxy settings of the environment."""
    return httpx.Client(trust_env=True, timeout=_TIMEOUT)


def _depth(depth: Optional[str]) -> str:
    chosen = (depth or os.getenv("LINKUP_DEPTH") or DEFAULT_DEPTH).strip().lower()
    if chosen in DEPTHS:
        return chosen
    logger.warning("LINKUP_DEPTH is not one of %s: using %s", DEPTHS, DEFAULT_DEPTH)
    return DEFAULT_DEPTH


def _request_body(
    query: str,
    max_results: int,
    time_range: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    include_domains: Optional[List[str]],
    exclude_domains: Optional[List[str]],
    depth: str,
    today: Optional[date] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "q": query,
        "depth": depth,
        "outputType": "searchResults",
        "maxResults": max(1, max_results),
    }
    from_date = start_date
    if not from_date and time_range in _TIME_RANGE_DAYS:
        days = timedelta(days=_TIME_RANGE_DAYS[time_range])
        from_date = ((today or date.today()) - days).isoformat()
    if from_date:
        body["fromDate"] = from_date
    if end_date:
        body["toDate"] = end_date
    if include_domains:
        body["includeDomains"] = include_domains
    if exclude_domains:
        body["excludeDomains"] = exclude_domains
    return body


def _failure(response: httpx.Response) -> ToolException:
    status = response.status_code
    if status == 400:
        logger.info("Linkup refused a search's parameters: status 400")
        return ToolException(
            "The web search service refused the search's parameters. Check the "
            "dates (YYYY-MM-DD, the start before the end) and the domains."
        )
    if status in (401, 402, 403):
        logger.error("Linkup refused the platform's key: status %s", status)
        return ToolException(
            "The web search service is not available right now because of its "
            "configuration. Do not search again: tell the user the search failed."
        )
    if status == 429:
        logger.error("Linkup's rate limit or credit balance ran out: status 429")
        return ToolException(
            "The web search service has reached its limit for now. Do not search "
            "again right away: work with what you know and tell the user."
        )
    logger.warning("Linkup could not answer a search: status %s", status)
    return ToolException(
        "The web search service could not answer. Try once more later, or tell "
        "the user the search failed."
    )


def _results(payload: Any) -> List[Dict[str, Any]]:
    items = payload.get("results") if isinstance(payload, dict) else None
    results: List[Dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or item.get("type", "text") != "text":
            continue
        results.append(
            {
                "title": item.get("name") or "",
                "url": item.get("url") or "",
                "content": str(item.get("content") or "")[:MAX_CONTENT_CHARS],
            }
        )
    return results


def linkup_search(
    query: str,
    max_results: int = 5,
    topic: str = "general",
    include_raw_content: Optional[str] = None,
    time_range: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    country: Optional[str] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    depth: Optional[str] = None,
) -> Dict[str, Any]:
    """One search. Raises ``ToolException``, in words for the model, on failure.

    ``topic``, ``country`` and ``include_raw_content`` have no counterpart.
    Those a call did set are named in ``ignored``, so the model does not take
    the results for filtered ones.
    """
    api_key = os.getenv("LINKUP_API_KEY")
    if not api_key:
        logger.error("LINKUP_API_KEY is not set: the search was not made")
        raise ToolException(
            "The web search service is not available right now because of its "
            "configuration. Do not search again: tell the user the search failed."
        )

    chosen_depth = _depth(depth)
    base_url = (os.getenv("LINKUP_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    body = _request_body(
        query,
        max_results,
        time_range,
        start_date,
        end_date,
        include_domains,
        exclude_domains,
        chosen_depth,
    )
    try:
        with _client() as client:
            response = client.post(
                f"{base_url}/search",
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.HTTPError as exc:
        logger.warning("Could not reach Linkup: %s", type(exc).__name__)
        raise ToolException(
            "The web search service could not be reached. Try once more later, "
            "or tell the user the search failed."
        ) from None
    if response.status_code != 200:
        raise _failure(response)
    try:
        payload = response.json()
    except ValueError:
        logger.warning("Linkup's answer to a search was not JSON")
        raise ToolException(
            "The web search service could not answer. Try once more later, or "
            "tell the user the search failed."
        ) from None

    results = _results(payload)
    logger.info("Linkup search: depth=%s results=%s", chosen_depth, len(results))
    answer: Dict[str, Any] = {"query": query, "results": results}
    ignored = [
        name
        for name, was_set in (
            ("topic", topic != "general"),
            ("country", bool(country)),
            ("include_raw_content", bool(include_raw_content)),
        )
        if was_set
    ]
    if ignored:
        answer["ignored"] = ignored
    return answer


__all__ = ["DEPTHS", "linkup_search"]

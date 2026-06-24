"""Synchronous internet search tool for LangChain agents.

This module provides a BaseTool implementation that wraps Tavily's search API
and exposes a sync `_run` path so it can be used in sync agent executions. The
async `_arun` is implemented via a thread executor for parity.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Literal, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from tavily import TavilyClient

API_KEY = os.getenv("TAVILY_API_KEY")

_tavily_client_instance: Optional[TavilyClient] = None

# Maps common ISO codes and localized/alias names to the full lowercase English
# country name Tavily expects. Deliberately small — only the countries agents
# realistically pass. Unrecognized full names are passed through untouched;
# unrecognized short codes are dropped (see `_normalize_country`).
_COUNTRY_ALIASES: Dict[str, str] = {
    # Poland
    "pl": "poland",
    "pol": "poland",
    "polska": "poland",
    # United States
    "us": "united states",
    "usa": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "united states of america": "united states",
    "stany zjednoczone": "united states",
    # United Kingdom
    "uk": "united kingdom",
    "gb": "united kingdom",
    "gbr": "united kingdom",
    "great britain": "united kingdom",
    "england": "united kingdom",
    # Germany
    "de": "germany",
    "deu": "germany",
    "niemcy": "germany",
    "deutschland": "germany",
    # France
    "fr": "france",
    "fra": "france",
    "francja": "france",
    # Spain
    "es": "spain",
    "esp": "spain",
    "hiszpania": "spain",
    "espana": "spain",
    "españa": "spain",
    # Italy
    "it": "italy",
    "ita": "italy",
    "wlochy": "italy",
    "włochy": "italy",
    # Netherlands
    "nl": "netherlands",
    "nld": "netherlands",
    "holandia": "netherlands",
    # Czech Republic
    "cz": "czech republic",
    "cze": "czech republic",
    "czechy": "czech republic",
    "czechia": "czech republic",
    # Ukraine
    "ua": "ukraine",
    "ukr": "ukraine",
    "ukraina": "ukraine",
    # A few more frequently used
    "ca": "canada",
    "can": "canada",
    "au": "australia",
    "aus": "australia",
    "jp": "japan",
    "jpn": "japan",
    "cn": "china",
    "chn": "china",
    "in": "india",
    "ind": "india",
    "br": "brazil",
    "bra": "brazil",
    "ru": "russia",
    "rus": "russia",
}


def _normalize_country(country: Optional[str], topic: str) -> Optional[str]:
    """Coerce an agent-supplied ``country`` into a value Tavily accepts.

    Tavily requires the full English country name in lowercase (e.g. ``"poland"``)
    and only honours ``country`` when ``topic == "general"``. Agents frequently
    pass ISO codes (``"PL"``), localized names (``"Polska"``) or set it for the
    wrong topic, which makes the API raise ``BadRequestError``. This guard maps
    the obvious cases and otherwise drops a value that would certainly fail,
    rather than letting the whole search blow up.

    Returns the normalized country name, or ``None`` to omit the parameter.
    """
    if not country or not country.strip():
        return None

    if topic != "general":
        logging.warning(
            "Dropping Tavily 'country=%s': only supported with topic='general' (got topic=%s)",
            country,
            topic,
        )
        return None

    normalized = country.strip().lower()

    alias = _COUNTRY_ALIASES.get(normalized)
    if alias is not None:
        if alias != normalized:
            logging.info("Normalized Tavily country '%s' -> '%s'", country, alias)
        return alias

    # Looks like a bare ISO-ish code we don't recognize -> safer to omit than to
    # send something Tavily will reject.
    if normalized.replace(".", "").isalpha() and len(normalized.replace(".", "")) <= 3:
        logging.warning(
            "Dropping unrecognized Tavily country code '%s' (use a full lowercase country name)",
            country,
        )
        return None

    # Assume it's a real full country name; pass through (Tavily accepts names
    # beyond our small alias table).
    return normalized


def get_tavily_client() -> TavilyClient:
    global _tavily_client_instance
    if _tavily_client_instance is None:
        if not API_KEY:
            raise ValueError("TAVILY_API_KEY environment variable is not set.")
        _tavily_client_instance = TavilyClient(api_key=API_KEY)
    return _tavily_client_instance


class InternetSearchInput(BaseModel):
    """Argument schema for the Tavily-backed internet search tool.

    Defined explicitly so each parameter — especially ``country`` — carries a
    description the model can see, instead of relying on the bare ``_run``
    signature.
    """

    query: str = Field(description="The search query to look up on the web.")
    max_results: int = Field(
        default=5, description="Maximum number of search results to return."
    )
    topic: Literal["general", "news", "finance"] = Field(
        default="general",
        description=(
            "Search category. Use 'general' for broad web search, 'news' for "
            "recent news, 'finance' for financial topics."
        ),
    )
    include_raw_content: Optional[Literal["text", "markdown"]] = Field(
        default=None,
        description="Include the full page content as 'text' or 'markdown' (omit for snippets only).",
    )
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        default=None,
        description="Restrict results to content from the last day/week/month/year.",
    )
    start_date: Optional[str] = Field(
        default=None, description="Earliest publish date, ISO format YYYY-MM-DD."
    )
    end_date: Optional[str] = Field(
        default=None, description="Latest publish date, ISO format YYYY-MM-DD."
    )
    country: Optional[str] = Field(
        default=None,
        description=(
            "Boost results from a specific country. Must be the FULL English "
            "country name in lowercase, e.g. 'poland', 'united states', "
            "'germany'. Do NOT use ISO codes ('PL', 'US') or localized names "
            "('Polska'). Only takes effect with topic='general' — omit it for "
            "other topics."
        ),
    )
    include_domains: Optional[list[str]] = Field(
        default=None, description="Only include results from these domains."
    )
    exclude_domains: Optional[list[str]] = Field(
        default=None, description="Exclude results from these domains."
    )


class InternetSearchTool(BaseTool):
    """Sync-capable Tavily search tool for agents.

    Exists to ensure sync LangGraph tool execution succeeds when tools are
    invoked via sync pathways. It also exposes an async path for parity.
    """

    name: str = "internet_search"
    description: str = (
        "Perform an internet search using Tavily and return the raw response. "
        "Use when you need up-to-date web results. To bias results toward a "
        "country, pass `country` as the full English country name in lowercase "
        "(e.g. 'poland'), only with topic='general'."
    )
    args_schema: type[BaseModel] = InternetSearchInput

    max_workers: int = Field(default=1, exclude=True)

    def _run(
        self,
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: Optional[Literal["text", "markdown"]] = None,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        country: Optional[str] = None,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        client = get_tavily_client()
        country = _normalize_country(country, topic)
        logging.info(
            self._format_log(
                query,
                max_results,
                topic,
                include_raw_content,
                time_range,
                start_date,
                end_date,
                country,
                include_domains,
                exclude_domains,
            )
        )
        return client.search(
            query=query,
            max_results=max_results,
            topic=topic,
            include_raw_content=include_raw_content,
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            country=country,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: Optional[Literal["text", "markdown"]] = None,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        country: Optional[str] = None,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
    ) -> Dict[str, Any]:

        return await asyncio.to_thread(
            self._run,
            query,
            max_results,
            topic,
            include_raw_content,
            time_range,
            start_date,
            end_date,
            country,
            include_domains,
            exclude_domains,
        )

    @staticmethod
    def _format_log(
        query: str,
        max_results: int,
        topic: str,
        include_raw_content: Optional[str],
        time_range: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        country: Optional[str],
        include_domains: Optional[list[str]],
        exclude_domains: Optional[list[str]],
    ) -> str:
        parts = [
            f"Performing internet search with query: {query}",
            f"max_results: {max_results}",
            f"topic: {topic}",
        ]
        if include_raw_content:
            parts.append(f"include_raw_content: {include_raw_content}")
        if time_range:
            parts.append(f"time_range: {time_range}")
        if start_date:
            parts.append(f"start_date: {start_date}")
        if end_date:
            parts.append(f"end_date: {end_date}")
        if country:
            parts.append(f"country: {country}")
        if include_domains:
            parts.append(f"include_domains: {', '.join(include_domains)}")
        if exclude_domains:
            parts.append(f"exclude_domains: {', '.join(exclude_domains)}")
        return ", ".join(parts)


def get_search_tools() -> list[BaseTool]:
    """Factory returning the internet search tool with sync support."""

    return [InternetSearchTool()]


__all__ = ["InternetSearchTool", "get_search_tools"]

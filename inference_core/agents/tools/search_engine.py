"""Synchronous internet search tool for LangChain agents.

This module provides a BaseTool implementation that wraps Tavily's search API
and exposes a sync `_run` path so it can be used in sync agent executions. The
async `_arun` is implemented via a thread executor for parity.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from typing import Any, Dict, Literal, Optional

from langchain_core.tools import BaseTool
from pydantic import Field
from tavily import TavilyClient

API_KEY = os.getenv("TAVILY_API_KEY")

_tavily_client_instance: Optional[TavilyClient] = None


def get_tavily_client() -> TavilyClient:
    global _tavily_client_instance
    if _tavily_client_instance is None:
        if not API_KEY:
            raise ValueError("TAVILY_API_KEY environment variable is not set.")
        _tavily_client_instance = TavilyClient(api_key=API_KEY)
    return _tavily_client_instance


class InternetSearchTool(BaseTool):
    """Sync-capable Tavily search tool for agents.

    Exists to ensure sync LangGraph tool execution succeeds when tools are
    invoked via sync pathways. It also exposes an async path for parity.
    """

    name: str = "internet_search"
    description: str = (
        "Perform an internet search using Tavily and return the raw response. "
        "Use when you need up-to-date web results."
    )

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
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            return await loop.run_in_executor(
                executor,
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

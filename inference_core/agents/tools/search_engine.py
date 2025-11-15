import logging
import os
from typing import Literal, Optional

from langchain.tools import tool
from tavily import TavilyClient

API_KEY = os.getenv("TAVILY_API_KEY")

_tavily_client_instance = None


def get_tavily_client() -> TavilyClient:
    global _tavily_client_instance
    if _tavily_client_instance is None:
        if not API_KEY:
            raise ValueError("TAVILY_API_KEY environment variable is not set.")
        _tavily_client_instance = TavilyClient(api_key=API_KEY)
    return _tavily_client_instance


# Web search tool
@tool
def internet_search(
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
) -> dict:
    """
    Perform an internet search using the configured Tavily client and return the raw response.

    This function is a thin wrapper around the underlying Tavily client search API. It forwards
    the provided search parameters to the client and returns whatever dictionary the client
    produces. It does not perform additional processing of results.

    Parameters
    ----------
    query : str
        The search query string.
    max_results : int, optional
        Maximum number of results to return (default: 5).
    topic : {'general', 'news', 'finance'}, optional
        Restrict search to a specific topic domain. Defaults to 'general'.
    include_raw_content : {'text', 'markdown'} or None, optional
        If provided, include raw content in the chosen format for each result. If None, raw
        content is omitted.
    time_range : {'day', 'week', 'month', 'year'} or None, optional
        Predefined relative time range to filter results (e.g., 'day' for last 24 hours).
    start_date : str or None, optional
        Explicit start date for the search window in ISO 8601 format (YYYY-MM-DD). If both
        start_date and end_date are provided, they define an explicit range; otherwise
        time_range may be used.
    end_date : str or None, optional
        Explicit end date for the search window in ISO 8601 format (YYYY-MM-DD).
    country : str or None, optional
        lowerstring country name (e.g., 'united states') to localize or filter results, depending on the client capabilities.
    include_domains : list[str] or None, optional
        If provided, only include results from these domains (exact hostnames).
    exclude_domains : list[str] or None, optional
        If provided, exclude results from these domains.

    Returns
    -------
    dict
        The raw response dictionary returned by the Tavily client search method. Structure and
        fields depend on the client implementation and requested options.

    Notes
    -----
    - Date format expectations and exact behavior for `country`, `include_domains` and
      `exclude_domains` depend on the configured Tavily client.
    - This function does not validate combinations of parameters (e.g., overlapping
      start_date/time_range); validation is delegated to the client.
    """
    client = get_tavily_client()

    msg = f"Performing internet search with query: {query}"
    if topic != "general":
        msg += f", topic: {topic}"
    msg += f", max_results: {max_results}"
    if include_raw_content:
        msg += f", include_raw_content: {include_raw_content}"
    if time_range:
        msg += f", time_range: {time_range}"
    if start_date or end_date:
        parts = []
        if start_date:
            parts.append(f"start_date: {start_date}")
        if end_date:
            parts.append(f"end_date: {end_date}")
        msg += ", " + ", ".join(parts)
    if country:
        msg += f", country: {country}"
    if include_domains:
        msg += f", include_domains: {', '.join(include_domains)}"
    if exclude_domains:
        msg += f", exclude_domains: {', '.join(exclude_domains)}"

    logging.info(msg)

    response = client.search(
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
    return response

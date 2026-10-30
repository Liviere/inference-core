"""What the internet search tool's log lines may quote.

The model writes the tool's text arguments, so they can hold anything it has
read. ``internet_search_log_query`` off keeps them out of the log.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from inference_core.agents.tools.search_engine import (
    InternetSearchTool,
    _normalize_country,
)
from inference_core.core.config import Settings

QUERY = "invoice 4711 of Jan Kowalski"
FREE_TEXT = {
    "start_date": "start-secret",
    "end_date": "end-secret",
    "country": "country secret",
    "include_domains": ["included-secret.example", "other-secret.example"],
    "exclude_domains": ["excluded-secret.example"],
}
SECRETS = (
    QUERY,
    "start-secret",
    "end-secret",
    "country secret",
    "included-secret",
    "other-secret",
    "excluded-secret",
)


@pytest.fixture
def log_query(request):
    """Sets the option for what the tool reads; a parameter gives its value."""
    settings = MagicMock(
        internet_search_provider="tavily", internet_search_log_query=request.param
    )
    with patch(
        "inference_core.agents.tools.search_engine.get_settings",
        return_value=settings,
    ):
        yield request.param


@pytest.fixture
def tavily():
    client = MagicMock()
    client.search.return_value = {"results": []}
    with patch(
        "inference_core.agents.tools.search_engine.get_tavily_client",
        return_value=client,
    ):
        yield client


def _search(**kwargs):
    InternetSearchTool(provider="tavily").invoke({"query": QUERY, **kwargs})


@pytest.mark.parametrize("log_query", [False], indirect=True)
def test_without_the_option_no_text_of_the_model_is_logged(log_query, tavily, caplog):
    with caplog.at_level(logging.INFO):
        _search(max_results=3, topic="general", time_range="week", **FREE_TEXT)

    for secret in SECRETS:
        assert secret not in caplog.text
    assert f"query length: {len(QUERY)}" in caplog.text
    assert "max_results: 3" in caplog.text
    assert "topic: general" in caplog.text
    assert "time_range: week" in caplog.text
    assert "start_date: set" in caplog.text
    assert "end_date: set" in caplog.text
    assert "country: set" in caplog.text
    assert "include_domains: 2" in caplog.text
    assert "exclude_domains: 1" in caplog.text


@pytest.mark.parametrize("log_query", [False], indirect=True)
def test_without_the_option_the_search_itself_is_the_same(log_query, tavily):
    _search(**FREE_TEXT)

    call = tavily.search.call_args.kwargs
    assert call["query"] == QUERY
    assert call["include_domains"] == FREE_TEXT["include_domains"]
    assert call["start_date"] == "start-secret"


@pytest.mark.parametrize("log_query", [False], indirect=True)
@pytest.mark.parametrize(
    "country, topic",
    [("country secret", "news"), ("zzq", "general")],
    ids=["other topic", "unknown code"],
)
def test_without_the_option_a_dropped_country_is_not_quoted(
    log_query, caplog, country, topic
):
    with caplog.at_level(logging.INFO):
        assert _normalize_country(country, topic) is None

    assert "Dropping" in caplog.text
    assert country not in caplog.text
    assert f"<{len(country)} chars>" in caplog.text


@pytest.mark.parametrize("log_query", [True], indirect=True)
def test_with_the_option_the_line_stays_as_it_was(log_query, tavily, caplog):
    with caplog.at_level(logging.INFO):
        _search(max_results=3, topic="general", time_range="week", **FREE_TEXT)

    assert (
        f"Performing internet search with query: {QUERY}, max_results: 3, "
        "topic: general, time_range: week, start_date: start-secret, "
        "end_date: end-secret, country: country secret, "
        "include_domains: included-secret.example, other-secret.example, "
        "exclude_domains: excluded-secret.example"
    ) in caplog.text


@pytest.mark.parametrize("log_query", [True], indirect=True)
def test_with_the_option_a_dropped_country_is_quoted(log_query, caplog):
    with caplog.at_level(logging.INFO):
        _normalize_country("zzq", "general")

    assert "'zzq'" in caplog.text


@pytest.mark.parametrize("log_query", [False], indirect=True)
@pytest.mark.parametrize("argument", ["topic", "time_range", "include_raw_content"])
def test_the_values_the_line_keeps_are_limited_by_the_schema(
    log_query, tavily, argument
):
    """What keeps them out of the model's free text; the line relies on it."""
    with pytest.raises(ValidationError):
        _search(**{argument: "free text"})


def test_the_setting_keeps_todays_log_unless_turned_off(monkeypatch):
    monkeypatch.delenv("INTERNET_SEARCH_LOG_QUERY", raising=False)
    assert Settings.model_fields["internet_search_log_query"].default is True

    monkeypatch.setenv("INTERNET_SEARCH_LOG_QUERY", "false")
    assert Settings().internet_search_log_query is False

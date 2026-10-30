"""The internet search tool with Linkup as its provider, on a fake HTTP server."""

import json
import logging
from datetime import date
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain_core.tools import ToolException

from inference_core.agents.tools import search_linkup
from inference_core.agents.tools.search_engine import (
    InternetSearchTool,
    resolve_search_provider,
    search_provider_label,
)
from inference_core.agents.tools.search_linkup import (
    MAX_CONTENT_CHARS,
    _request_body,
    linkup_search,
)

SECRET_QUERY = "Anna Kowalska 44051401458"


class FakeLinkup:
    """Answers each POST the same way and keeps the requests."""

    def __init__(self, status=200, payload=None, raises=None):
        self.status = status
        self.payload = {"results": []} if payload is None else payload
        self.raises = raises
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if self.raises:
            raise self.raises
        if isinstance(self.payload, str):
            return httpx.Response(self.status, text=self.payload)
        return httpx.Response(self.status, json=self.payload)

    @property
    def body(self):
        return json.loads(self.requests[-1].content)


@pytest.fixture
def linkup(monkeypatch):
    monkeypatch.setenv("LINKUP_API_KEY", "linkup-test-key")
    monkeypatch.delenv("LINKUP_BASE_URL", raising=False)
    monkeypatch.delenv("LINKUP_DEPTH", raising=False)

    def install(**kwargs):
        server = FakeLinkup(**kwargs)
        monkeypatch.setattr(
            search_linkup,
            "_client",
            lambda: httpx.Client(transport=httpx.MockTransport(server)),
        )
        return server

    return install


class TestChoosingTheProvider:
    def test_tavily_is_the_default(self):
        assert resolve_search_provider() == "tavily"
        assert InternetSearchTool().provider == "tavily"
        assert search_provider_label() == "Tavily"

    def test_the_setting_chooses(self):
        settings = MagicMock(internet_search_provider="linkup")
        with patch(
            "inference_core.agents.tools.search_engine.get_settings",
            return_value=settings,
        ):
            assert InternetSearchTool().provider == "linkup"

    def test_a_name_given_wins_over_the_setting(self):
        settings = MagicMock(internet_search_provider="linkup")
        with patch(
            "inference_core.agents.tools.search_engine.get_settings",
            return_value=settings,
        ):
            assert InternetSearchTool(provider="tavily").provider == "tavily"

    def test_an_unknown_name_is_refused_and_does_not_become_tavily(self):
        with pytest.raises(ValueError, match="bing"):
            InternetSearchTool(provider="bing")
        with pytest.raises(ValueError, match="bing"):
            search_provider_label("bing")

    def test_name_and_arguments_do_not_depend_on_the_provider(self):
        tavily, linkup_tool = InternetSearchTool(), InternetSearchTool(provider="linkup")
        assert tavily.name == linkup_tool.name == "internet_search"
        assert tavily.args == linkup_tool.args

    def test_tavilys_tool_is_as_it_was(self):
        tool = InternetSearchTool()
        assert "using Tavily" in tool.description
        assert not tool.handle_tool_error

    def test_the_model_is_told_what_has_no_effect(self):
        description = InternetSearchTool(provider="linkup").description
        assert "Tavily" not in description
        assert "`country`, `topic` and `include_raw_content` have no effect" in description

    def test_tavily_is_not_called_with_linkup_chosen(self, linkup):
        linkup()
        with patch(
            "inference_core.agents.tools.search_engine.get_tavily_client"
        ) as tavily:
            InternetSearchTool(provider="linkup")._run(query="q")
        tavily.assert_not_called()


class TestRequest:
    def test_arguments_are_mapped(self, linkup):
        server = linkup()
        InternetSearchTool(provider="linkup")._run(
            query="pytanie",
            max_results=3,
            start_date="2026-01-01",
            end_date="2026-02-01",
            include_domains=["gov.pl"],
            exclude_domains=["spam.example"],
        )
        request = server.requests[-1]
        assert str(request.url) == "https://api.linkup.so/v1/search"
        assert request.headers["authorization"] == "Bearer linkup-test-key"
        assert server.body == {
            "q": "pytanie",
            "depth": "fast",
            "outputType": "searchResults",
            "maxResults": 3,
            "fromDate": "2026-01-01",
            "toDate": "2026-02-01",
            "includeDomains": ["gov.pl"],
            "excludeDomains": ["spam.example"],
        }

    def test_what_is_not_set_is_not_sent(self, linkup):
        server = linkup()
        InternetSearchTool(provider="linkup")._run(query="q", include_domains=[])
        assert set(server.body) == {"q", "depth", "outputType", "maxResults"}

    def test_the_address_comes_from_the_setting(self, linkup, monkeypatch):
        server = linkup()
        monkeypatch.setenv("LINKUP_BASE_URL", "https://eu.linkup.example/v1/")
        linkup_search(query="q")
        assert str(server.requests[-1].url) == "https://eu.linkup.example/v1/search"

    @pytest.mark.parametrize(
        "time_range, expected",
        [("day", "2026-10-29"), ("week", "2026-10-23"), ("month", "2026-09-30")],
    )
    def test_a_time_range_becomes_a_start_date(self, time_range, expected):
        body = _request_body(
            "q", 5, time_range, None, None, None, None, "fast", today=date(2026, 10, 30)
        )
        assert body["fromDate"] == expected

    def test_a_start_date_given_wins_over_the_time_range(self):
        body = _request_body("q", 5, "day", "2026-01-01", None, None, None, "fast")
        assert body["fromDate"] == "2026-01-01"

    @pytest.mark.parametrize("depth", ["flash", "fast", "standard"])
    def test_the_depth_comes_from_the_setting(self, linkup, monkeypatch, depth):
        server = linkup()
        monkeypatch.setenv("LINKUP_DEPTH", depth)
        linkup_search(query="q")
        assert server.body["depth"] == depth

    def test_deep_is_not_offered(self, linkup, monkeypatch):
        server = linkup()
        monkeypatch.setenv("LINKUP_DEPTH", "deep")
        linkup_search(query="q")
        assert server.body["depth"] == "fast"

    def test_the_client_reads_the_proxy_settings(self):
        with search_linkup._client() as client:
            assert client.trust_env is True
            assert client.follow_redirects is False


class TestAnswer:
    def test_it_has_the_shape_of_tavilys(self, linkup):
        linkup(
            payload={
                "results": [
                    {
                        "type": "text",
                        "name": "Tytuł",
                        "url": "https://example.pl/a",
                        "content": "Treść",
                        "favicon": "",
                    },
                    {"type": "image", "name": "Obraz", "url": "https://example.pl/i"},
                ]
            }
        )
        answer = InternetSearchTool(provider="linkup")._run(query="pytanie")
        assert answer == {
            "query": "pytanie",
            "results": [
                {"title": "Tytuł", "url": "https://example.pl/a", "content": "Treść"}
            ],
        }

    def test_a_long_content_is_cut(self, linkup):
        linkup(payload={"results": [{"type": "text", "content": "x" * 50000}]})
        answer = linkup_search(query="q")
        assert len(answer["results"][0]["content"]) == MAX_CONTENT_CHARS

    @pytest.mark.parametrize("payload", [{}, {"results": None}, {"results": ["x"]}, []])
    def test_an_odd_answer_has_no_results(self, linkup, payload):
        linkup(payload=payload)
        assert linkup_search(query="q")["results"] == []

    def test_what_had_no_effect_is_named(self, linkup):
        server = linkup()
        answer = InternetSearchTool(provider="linkup")._run(
            query="q", topic="news", country="poland", include_raw_content="text"
        )
        assert answer["ignored"] == ["topic", "country", "include_raw_content"]
        assert not {"topic", "country", "include_raw_content"} & set(server.body)

    def test_nothing_is_named_when_nothing_was_set(self, linkup):
        linkup()
        assert "ignored" not in linkup_search(query="q")


class TestFailures:
    @pytest.mark.parametrize("status", [400, 401, 402, 403, 429, 500, 503])
    def test_a_status_is_a_failure_the_model_reads(self, linkup, status, caplog):
        linkup(status=status, payload={"error": SECRET_QUERY})
        tool = InternetSearchTool(provider="linkup")
        with caplog.at_level(logging.DEBUG, logger=search_linkup.__name__):
            with pytest.raises(ToolException) as failure:
                tool._run(query=SECRET_QUERY)
        assert SECRET_QUERY not in str(failure.value)
        assert SECRET_QUERY not in caplog.text
        assert str(status) in caplog.text

    @pytest.mark.parametrize("status", [401, 402, 429])
    def test_what_an_operator_must_fix_is_logged_as_an_error(self, linkup, status, caplog):
        linkup(status=status)
        with caplog.at_level(logging.ERROR, logger=search_linkup.__name__):
            with pytest.raises(ToolException):
                linkup_search(query="q")
        assert caplog.records

    @pytest.mark.parametrize(
        "error", [httpx.ConnectError("no"), httpx.ReadTimeout("slow")]
    )
    def test_no_connection_is_a_failure_the_model_reads(self, linkup, error):
        linkup(raises=error)
        with pytest.raises(ToolException, match="could not be reached"):
            linkup_search(query="q")

    def test_an_answer_that_is_not_json(self, linkup):
        linkup(payload="<html>")
        with pytest.raises(ToolException, match="could not answer"):
            linkup_search(query="q")

    def test_no_key_makes_no_request(self, linkup, monkeypatch):
        server = linkup()
        monkeypatch.delenv("LINKUP_API_KEY")
        with pytest.raises(ToolException, match="configuration"):
            linkup_search(query="q")
        assert server.requests == []

    def test_there_is_one_request_and_no_retry(self, linkup):
        server = linkup(status=503)
        with pytest.raises(ToolException):
            linkup_search(query="q")
        assert len(server.requests) == 1

    def test_the_tool_answers_the_model_instead_of_raising(self, linkup):
        linkup(status=429)
        tool = InternetSearchTool(provider="linkup")
        result = tool.invoke({"query": "q"})
        assert "reached its limit" in result

    def test_a_success_logs_no_query(self, linkup, caplog):
        linkup()
        with caplog.at_level(logging.DEBUG, logger=search_linkup.__name__):
            linkup_search(query=SECRET_QUERY)
        assert caplog.records
        assert SECRET_QUERY not in caplog.text

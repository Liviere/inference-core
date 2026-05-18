"""Tests for the OpenWeather-backed weather tool."""

from unittest.mock import patch

from inference_core.agents.tools.weather_provider import check_weather
from inference_core.schemas.weather import OpenWeatherForecastResponse
from tests.fixtures.weather_response import load_sample_openweather_payload


class _MockResponse:
    """Minimal requests.Response stand-in for focused tool tests."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        """Mirror a successful HTTP response."""

    def json(self):
        """Return the prepared JSON payload."""

        return self._payload


@patch("inference_core.agents.tools.weather_provider._OPEN_WEATHER_API_KEY", "test-key")
@patch("inference_core.agents.tools.weather_provider.requests.get")
def test_check_weather_returns_validated_forecast_model(mock_get) -> None:
    mock_get.side_effect = [
        _MockResponse([{"lat": "51.1073", "lon": "17.0385"}]),
        _MockResponse(load_sample_openweather_payload()),
    ]

    payload = check_weather.func(country="Poland", city="Wroclaw")

    assert isinstance(payload, OpenWeatherForecastResponse)
    assert payload.cod == "200"
    assert payload.cnt == 40
    assert payload.city.name == "Wrocław"
    assert payload.forecasts[5].rain is not None
    assert payload.forecasts[5].rain.volume_last_3h == 0.22


@patch("inference_core.agents.tools.weather_provider._OPEN_WEATHER_API_KEY", "test-key")
@patch("inference_core.agents.tools.weather_provider.requests.get")
def test_check_weather_returns_string_error_for_invalid_payload(mock_get) -> None:
    mock_get.side_effect = [
        _MockResponse([{"lat": "51.1073", "lon": "17.0385"}]),
        _MockResponse({"cod": "200"}),
    ]

    payload = check_weather.func(country="Poland", city="Wroclaw")

    assert payload == "Weather API returned an unexpected payload."

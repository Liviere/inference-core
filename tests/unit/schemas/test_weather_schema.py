"""Tests for the OpenWeather structured-output schema."""

import json

from inference_core.schemas.weather import (
    OpenWeatherForecastResponse,
    get_openweather_response_schema,
)
from tests.fixtures.weather_response import load_sample_openweather_payload


def test_validates_openweather_forecast_payload() -> None:
    payload = load_sample_openweather_payload()

    response = OpenWeatherForecastResponse.model_validate(payload)

    assert response.cod == "200"
    assert response.cnt == 40
    assert response.city.name == "Wrocław"
    assert len(response.forecasts) == 40
    assert response.forecasts[5].rain is not None
    assert response.forecasts[5].rain.volume_last_3h == 0.22


def test_dump_and_schema_keep_openweather_aliases() -> None:
    payload = load_sample_openweather_payload()

    response = OpenWeatherForecastResponse.model_validate(payload)
    dumped = response.model_dump(by_alias=True)
    schema = get_openweather_response_schema()

    assert "list" in dumped
    assert dumped["list"][5]["rain"]["3h"] == 0.22
    assert "list" in schema["properties"]
    assert '"3h"' in json.dumps(schema)

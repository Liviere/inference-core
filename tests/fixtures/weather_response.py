"""Shared OpenWeather payload for weather-related unit tests.

WHY: weather tool and schema tests need one stable, realistic forecast payload
that exercises aliases and nested precipitation fields without depending on an
external API response file.
"""

from datetime import UTC, datetime, timedelta
from typing import Any


def load_sample_openweather_payload() -> dict[str, Any]:
    """Provide one canonical forecast payload across weather tests."""

    base_forecast_time = datetime(2024, 5, 20, 9, 0, tzinfo=UTC)
    forecasts: list[dict[str, Any]] = []

    for index in range(40):
        forecast_time = base_forecast_time + timedelta(hours=3 * index)
        temperature = round(14.5 + (index % 6) * 0.7, 2)
        forecast: dict[str, Any] = {
            "dt": int(forecast_time.timestamp()),
            "main": {
                "temp": temperature,
                "feels_like": round(temperature - 0.8, 2),
                "temp_min": round(temperature - 0.6, 2),
                "temp_max": round(temperature + 0.4, 2),
                "pressure": 1012 + (index % 4),
                "humidity": 58 + (index % 10),
                "temp_kf": 0.0,
            },
            "weather": [
                {
                    "id": 803 if index % 5 else 500,
                    "main": "Rain" if index % 5 == 0 else "Clouds",
                    "description": (
                        "light rain" if index % 5 == 0 else "broken clouds"
                    ),
                    "icon": "10d" if index % 5 == 0 else "04d",
                }
            ],
            "clouds": {"all": 62 + (index % 20)},
            "wind": {
                "speed": round(3.2 + (index % 5) * 0.4, 2),
                "deg": 210 + index,
            },
            "visibility": 10000,
            "pop": round(0.05 * (index % 4), 2),
            "sys": {"pod": "d" if 6 <= forecast_time.hour < 18 else "n"},
            "dt_txt": forecast_time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if index == 5:
            forecast["rain"] = {"3h": 0.22}
            forecast["weather"] = [
                {
                    "id": 500,
                    "main": "Rain",
                    "description": "light rain",
                    "icon": "10n",
                }
            ]
            forecast["pop"] = 0.48

        forecasts.append(forecast)

    return {
        "cod": "200",
        "message": 0,
        "cnt": 40,
        "list": forecasts,
        "city": {
            "id": 3081368,
            "name": "Wroc\u0142aw",
            "coord": {"lat": 51.1079, "lon": 17.0385},
            "country": "PL",
            "population": 672929,
            "timezone": 7200,
            "sunrise": 1716176195,
            "sunset": 1716233554,
        },
    }


__all__ = ["load_sample_openweather_payload"]

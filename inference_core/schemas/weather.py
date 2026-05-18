"""OpenWeather response schemas for structured output.

WHY: weather-related agents need a typed representation of the OpenWeather
5-day / 3-hour forecast payload that can also be converted into JSON Schema
for ``response_format`` or model ``with_structured_output(...)`` usage.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenWeatherMainMetrics(BaseModel):
    """Capture the main temperature and pressure metrics for one forecast step."""

    temp: float = Field(description="Forecast temperature in Celsius")
    feels_like: float = Field(description="Human-perceived temperature in Celsius")
    temp_min: float = Field(description="Minimum temperature in Celsius")
    temp_max: float = Field(description="Maximum temperature in Celsius")
    pressure: int = Field(description="Atmospheric pressure in hPa")
    sea_level: int | None = Field(
        default=None,
        description="Sea level atmospheric pressure in hPa when available",
    )
    grnd_level: int | None = Field(
        default=None,
        description="Ground level atmospheric pressure in hPa when available",
    )
    humidity: int = Field(description="Humidity percentage")
    temp_kf: float = Field(description="Internal OpenWeather temperature offset")


class OpenWeatherCondition(BaseModel):
    """Describe the forecasted weather condition for one forecast step."""

    id: int = Field(description="OpenWeather condition identifier")
    main: str = Field(description="Short weather group, for example Rain or Clouds")
    description: str = Field(description="Human-readable weather description")
    icon: str = Field(description="OpenWeather icon code")


class OpenWeatherCloudCoverage(BaseModel):
    """Represent cloud coverage percentage for one forecast step."""

    all: int = Field(description="Cloud coverage percentage")


class OpenWeatherWind(BaseModel):
    """Represent wind metrics for one forecast step."""

    speed: float = Field(description="Wind speed in meters per second")
    deg: int = Field(description="Wind direction in degrees")
    gust: float | None = Field(
        default=None,
        description="Wind gust speed in meters per second when available",
    )


class OpenWeatherPrecipitation(BaseModel):
    """Normalize precipitation volume fields that use numeric JSON keys."""

    model_config = ConfigDict(populate_by_name=True)

    volume_last_3h: float = Field(
        alias="3h",
        description="Precipitation volume for the last 3 hours in millimeters",
    )


class OpenWeatherForecastSystem(BaseModel):
    """Keep the day or night flag emitted for a forecast step."""

    pod: Literal["d", "n"] = Field(description="Forecast period: day or night")


class OpenWeatherForecastEntry(BaseModel):
    """Model one item from the OpenWeather forecast list."""

    dt: int = Field(description="Unix timestamp for the forecast step")
    main: OpenWeatherMainMetrics = Field(
        description="Core temperature and pressure metrics"
    )
    weather: list[OpenWeatherCondition] = Field(
        description="Weather conditions associated with this forecast step"
    )
    clouds: OpenWeatherCloudCoverage = Field(
        description="Cloud coverage for this forecast step"
    )
    wind: OpenWeatherWind = Field(description="Wind metrics for this forecast step")
    visibility: int | None = Field(
        default=None,
        description="Visibility in meters when available",
    )
    pop: float | None = Field(
        default=None,
        description="Probability of precipitation between 0 and 1",
    )
    rain: OpenWeatherPrecipitation | None = Field(
        default=None,
        description="Rain volume for the forecast window when available",
    )
    snow: OpenWeatherPrecipitation | None = Field(
        default=None,
        description="Snow volume for the forecast window when available",
    )
    sys: OpenWeatherForecastSystem = Field(description="OpenWeather forecast metadata")
    dt_txt: str = Field(description="Forecast datetime in OpenWeather text format")


class OpenWeatherCoordinates(BaseModel):
    """Represent latitude and longitude returned for the forecast city."""

    lat: float = Field(description="Latitude")
    lon: float = Field(description="Longitude")


class OpenWeatherCity(BaseModel):
    """Describe the city metadata bundled with the forecast payload."""

    id: int = Field(description="OpenWeather city identifier")
    name: str = Field(description="City name")
    coord: OpenWeatherCoordinates = Field(description="City coordinates")
    country: str = Field(description="ISO country code")
    population: int | None = Field(
        default=None,
        description="City population when available",
    )
    timezone: int = Field(description="Timezone offset from UTC in seconds")
    sunrise: int = Field(description="Sunrise Unix timestamp")
    sunset: int = Field(description="Sunset Unix timestamp")


class OpenWeatherForecastResponse(BaseModel):
    """Represent the full OpenWeather 5-day / 3-hour forecast response."""

    model_config = ConfigDict(populate_by_name=True)

    cod: str = Field(description="Status code returned by OpenWeather")
    message: float | int = Field(description="OpenWeather response message value")
    cnt: int = Field(description="Number of forecast entries in the payload")
    forecasts: list[OpenWeatherForecastEntry] = Field(
        alias="list",
        description="Forecast entries emitted by OpenWeather",
    )
    city: OpenWeatherCity = Field(description="City metadata for the forecast")


def get_openweather_response_schema() -> dict[str, Any]:
    """Expose a JSON Schema that keeps the original OpenWeather field names."""

    return OpenWeatherForecastResponse.model_json_schema(by_alias=True)


__all__ = [
    "OpenWeatherMainMetrics",
    "OpenWeatherCondition",
    "OpenWeatherCloudCoverage",
    "OpenWeatherWind",
    "OpenWeatherPrecipitation",
    "OpenWeatherForecastSystem",
    "OpenWeatherForecastEntry",
    "OpenWeatherCoordinates",
    "OpenWeatherCity",
    "OpenWeatherForecastResponse",
    "get_openweather_response_schema",
]

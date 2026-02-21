---
name: check_weather
description: Use this skill for complex requests requiring detailed weather analysis across multiple locations or time periods.
---

# Weather Planning Skill

This skill provides a structured approach to weather-related information gathering using specialized subagents. It is particularly useful for planning activities, trips, or events influenced by the weather.

## When to Use This Skill

Use this skill when you need to:

- Get detailed weather information for multiple cities
- Perform comparative analysis of weather conditions
- Provide specific recommendations based on weather (e.g., "should I take an umbrella tomorrow morning in Krakow?")

## Process

### Step 1: Analyze the Request

Break down the user's weather query into specific locations and timeframes.

### Step 2: Delegate to Weather Subagents

For each specific location use the `task` tool to spawn a `weather_agent`.

**Example Usage of `task` tool:**

```json
{
	"task_name": "weather_agent",
	"instructions": "Check the weather in Krakow for tomorrow morning and tell me if it will rain."
}
```

### Step 3: Synthesize and Advise

Combine the findings from all subagents to provide a comprehensive response.

**Guidelines:**

- If the user asks for advice (e.g., "should I take an umbrella?"), interpret the weather data (precipitation probability, intensity) to give a clear "Yes" or "No" with reasoning.
- Cite the findings from the subagents clearly.

## Available Tools

You have access to:

- **task**: Spawn subagents. Currently available: `weather_agent`.
- **write_file**: Save your final analysis if requested.

## Best Practices

- **Specific Instructions**: Give the `weather_agent` clear instructions including the exact city and time period.
- **Contextual synthesis**: If multiple locations are involved, compare them if relevant to the user's goal.

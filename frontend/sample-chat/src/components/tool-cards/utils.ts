/**
 * Shared types and small helpers for tool-calling cards.
 *
 * WHY: Keeps card files free of duplicate type imports. The generic args
 * type lives here because we deliberately avoid the ``ToolCallFromTool``
 * helper from ``@langchain/react`` — our tools live in Python, so there is
 * no TypeScript schema to feed the helper. Callers provide the expected
 * args shape via a plain generic parameter on each card.
 */

import type { ToolMessage } from 'langchain';
import type { ToolCallWithResult as BaseToolCallWithResult } from '@langchain/react';

export type ToolCallState = 'pending' | 'completed' | 'error';

export type ToolCall = BaseToolCallWithResult;

/**
 * Safely parse a possibly-JSON string. If parsing fails the original
 * value is returned unchanged; callers then fall back to a string view.
 */
export function tryParseJSON(value: unknown): unknown {
	if (typeof value === 'string') {
		try {
			return JSON.parse(value);
		} catch {
			return value;
		}
	}
	return value;
}

// -------------------------------------------------------------------------
// Shared payload shapes for the playground-style tool cards.
// -------------------------------------------------------------------------

/** Payload expected by ``WeatherToolCard``. */
export interface WeatherData {
	city: string;
	temperature: number;
	condition: string;
	unit: string;
	humidity?: number;
	wind_speed?: number;
}

/** Payload expected by ``SearchToolCard``. */
export interface SearchResult {
	title: string;
	url: string;
	snippet: string;
}

// -------------------------------------------------------------------------
// Adapters from real Python-backed tool outputs to the card payload shapes.
// WHY: The cards were designed for the playground mock tools; these adapters
// let us reuse them for the real ``check_weather`` (OpenWeatherMap) and
// ``internet_search`` (Tavily) backends without touching card internals.
// -------------------------------------------------------------------------

/** Minimal slice of the OpenWeatherMap 5-day / 3-hour forecast response. */
interface OpenWeatherForecast {
	city?: { name?: string };
	list?: Array<{
		main?: { temp?: number; humidity?: number };
		weather?: Array<{ main?: string; description?: string }>;
		wind?: { speed?: number };
	}>;
	error?: string;
}

/**
 * Convert OpenWeatherMap's 5-day forecast into the card's flat "current
 * conditions" shape by taking the first 3-hour slot as a stand-in for
 * "now".  Returns ``null`` when the payload is missing / an error envelope,
 * so the dispatcher can fall back to ``GenericToolCard``.
 */
export function adaptOpenWeatherForecast(raw: unknown): WeatherData | null {
	if (!raw || typeof raw !== 'object') return null;
	const data = raw as OpenWeatherForecast;
	if (data.error) return null;

	const slot = data.list?.[0];
	const cityName = data.city?.name;
	const temp = slot?.main?.temp;
	const condition = slot?.weather?.[0]?.main ?? slot?.weather?.[0]?.description;

	if (!cityName || typeof temp !== 'number' || !condition) return null;

	return {
		city: cityName,
		temperature: Math.round(temp),
		condition,
		unit: 'celsius',
		humidity: slot?.main?.humidity,
		wind_speed: slot?.wind?.speed,
	};
}

/** Minimal slice of the Tavily search response. */
interface TavilyResponse {
	results?: Array<{
		title?: string;
		url?: string;
		content?: string;
		snippet?: string;
	}>;
	error?: string;
}

/**
 * Map Tavily's ``results[].content`` field onto the card's ``snippet``
 * field.  Returns ``null`` when the payload has no usable results.
 */
export function adaptTavilyResults(
	raw: unknown
): { results: SearchResult[] } | null {
	if (!raw || typeof raw !== 'object') return null;
	const data = raw as TavilyResponse;
	if (data.error || !Array.isArray(data.results)) return null;

	const results: SearchResult[] = data.results
		.filter((r) => r && r.url && r.title)
		.map((r) => ({
			title: r.title as string,
			url: r.url as string,
			snippet: r.snippet ?? r.content ?? '',
		}));

	if (results.length === 0) return null;
	return { results };
}

export type { ToolMessage };

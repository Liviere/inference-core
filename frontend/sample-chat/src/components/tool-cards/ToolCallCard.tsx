import type { ToolCallWithResult } from '@langchain/react';
import type { ToolMessage } from 'langchain';
import { WeatherToolCard } from './WeatherToolCard';
import { CalculatorToolCard } from './CalculatorToolCard';
import { SearchToolCard } from './SearchToolCard';
import { GenericToolCard } from './GenericToolCard';
import {
	adaptOpenWeatherForecast,
	adaptTavilyResults,
	tryParseJSON,
} from './utils';

/**
 * Dispatcher that picks the right tool card by ``call.name``.
 *
 * WHY: Keeps ChatView agnostic to the tool catalogue — adding a new
 * specialised card means editing this single switch, not the chat code.
 *
 * For Python-backed real tools (``check_weather``, ``internet_search``)
 * we reshape the server payload here so the playground-style cards can
 * render them unchanged.
 */

/**
 * Replace ``result.content`` with a card-friendly JSON payload.
 *
 * WHY: The cards call ``tryParseJSON(result?.content)`` which only
 * handles strings or already-parsed objects.  We JSON-stringify the
 * adapted shape so the existing parsing path keeps working.
 */
function withAdaptedContent(
	result: ToolMessage | undefined,
	adapted: unknown
): ToolMessage | undefined {
	if (!result) return result;
	const payload = JSON.stringify(adapted);
	// Clone the original ToolMessage so downstream consumers still see
	// a real LangChain message class (preserves tool_call_id, status, ...).
	return Object.assign(Object.create(Object.getPrototypeOf(result)), result, {
		content: payload,
	}) as ToolMessage;
}

export function ToolCallCard({ toolCall }: { toolCall: ToolCallWithResult }) {
	const { call, result, state } = toolCall;
	const args = (call.args ?? {}) as Record<string, unknown>;

	// --- Playground mock tools (get_weather / calculate / search_web) ----
	if (call.name === 'get_weather') {
		return (
			<WeatherToolCard
				call={{ name: call.name, args: args as { city?: string } }}
				result={result}
				state={state}
			/>
		);
	}

	if (call.name === 'calculate') {
		return (
			<CalculatorToolCard
				call={{ name: call.name, args: args as { expression?: string } }}
				result={result}
				state={state}
			/>
		);
	}

	if (call.name === 'search_web') {
		return (
			<SearchToolCard
				call={{ name: call.name, args: args as { query?: string } }}
				result={result}
				state={state}
			/>
		);
	}

	// --- Real Python-backed tools ----------------------------------------
	if (call.name === 'check_weather') {
		// Pass-through while the call is still pending so the card can
		// render its loading skeleton from ``args.city``.
		if (state === 'pending') {
			return (
				<WeatherToolCard
					call={{ name: call.name, args: args as { city?: string } }}
					result={undefined}
					state={state}
				/>
			);
		}
		const adapted = adaptOpenWeatherForecast(tryParseJSON(result?.content));
		if (adapted) {
			return (
				<WeatherToolCard
					call={{ name: call.name, args: args as { city?: string } }}
					result={withAdaptedContent(result, adapted)}
					state={state}
				/>
			);
		}
		// Adapter rejected the payload (e.g. ``{"error": "..."}``) — fall
		// through to GenericToolCard so the user sees the raw error.
	}

	if (call.name === 'internet_search') {
		if (state === 'pending') {
			return (
				<SearchToolCard
					call={{ name: call.name, args: args as { query?: string } }}
					result={undefined}
					state={state}
				/>
			);
		}
		const adapted = adaptTavilyResults(tryParseJSON(result?.content));
		if (adapted) {
			return (
				<SearchToolCard
					call={{ name: call.name, args: args as { query?: string } }}
					result={withAdaptedContent(result, adapted)}
					state={state}
				/>
			);
		}
	}

	return (
		<GenericToolCard
			call={{ name: call.name, args }}
			result={result}
			state={state}
		/>
	);
}

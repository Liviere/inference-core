import type { ToolCallWithResult } from '@langchain/react';
import { WeatherToolCard } from './WeatherToolCard';
import { CalculatorToolCard } from './CalculatorToolCard';
import { SearchToolCard } from './SearchToolCard';
import { GenericToolCard } from './GenericToolCard';

/**
 * Dispatcher that picks the right tool card by ``call.name``.
 *
 * WHY: Keeps ChatView agnostic to the tool catalogue — adding a new
 * specialised card means editing this single switch, not the chat code.
 */
export function ToolCallCard({ toolCall }: { toolCall: ToolCallWithResult }) {
	const { call, result, state } = toolCall;
	const args = (call.args ?? {}) as Record<string, unknown>;

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

	return (
		<GenericToolCard
			call={{ name: call.name, args }}
			result={result}
			state={state}
		/>
	);
}

import type { ToolCallState, ToolMessage } from './utils';
import { tryParseJSON } from './utils';

type CalcArgs = { expression?: string } & Record<string, unknown>;

/**
 * Calculator card — shows the original expression and the computed
 * ``result``, or an error banner if the expression was rejected.
 */
export function CalculatorToolCard({
	call,
	result,
	state,
}: {
	call: { name: string; args: CalcArgs };
	result?: ToolMessage;
	state: ToolCallState;
}) {
	const expression = call.args.expression ?? '';

	if (state === 'pending') {
		return (
			<div className="rounded-xl border border-[color:var(--color-accent-gold,#d5c3f7)] bg-surface p-4">
				<div className="flex items-center gap-3">
					<svg
						className="w-5 h-5 text-text-secondary animate-spin"
						viewBox="0 0 24 24"
						fill="none"
					>
						<circle
							className="opacity-25"
							cx="12"
							cy="12"
							r="10"
							stroke="currentColor"
							strokeWidth={3}
						/>
						<path
							className="opacity-75"
							fill="currentColor"
							d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
						/>
					</svg>
					<div className="text-sm font-medium text-text-secondary font-mono">
						Computing {expression}…
					</div>
				</div>
			</div>
		);
	}

	const raw =
		typeof result?.content === 'string'
			? result.content
			: String(result?.content ?? '');
	const parsed = tryParseJSON(raw);
	const computed =
		parsed && typeof parsed === 'object' && 'result' in parsed
			? String((parsed as { result: unknown }).result)
			: parsed && typeof parsed === 'object' && 'error' in parsed
				? `Error: ${(parsed as { error: unknown }).error}`
				: raw;
	const isError = computed.toLowerCase().startsWith('error');

	return (
		<div className="rounded-xl border border-[color:var(--color-accent-gold,#d5c3f7)] bg-surface p-4">
			<div className="flex items-center gap-2 mb-3">
				<div className="w-6 h-6 rounded-md bg-[color:var(--color-accent-gold,#d5c3f7)]/30 flex items-center justify-center">
					<span className="text-text-secondary text-xs font-bold">fx</span>
				</div>
				<span className="text-[11px] font-semibold text-text-tertiary uppercase tracking-wider">
					Calculator
				</span>
			</div>
			{isError ? (
				<div className="text-sm text-[color:var(--color-error)] font-mono">
					{computed}
				</div>
			) : (
				<div className="font-mono">
					<div className="text-sm text-text-tertiary">{expression}</div>
					<div className="text-2xl font-bold text-text tabular-nums mt-1">
						= {computed}
					</div>
				</div>
			)}
		</div>
	);
}

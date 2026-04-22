import { useState } from 'react';
import { ChevronIcon } from '../icons';
import type { ToolCallState, ToolMessage } from './utils';
import { tryParseJSON } from './utils';

/**
 * Fallback card for any unknown tool name.
 *
 * WHY: Keeps the conversation usable even if the agent invokes a tool
 * we haven't built a specialised card for — shows the raw args and
 * (collapsible) result so the user can still follow along.
 */
export function GenericToolCard({
	call,
	result,
	state,
}: {
	call: { name: string; args: Record<string, unknown> };
	result?: ToolMessage;
	state: ToolCallState;
}) {
	const [open, setOpen] = useState(false);
	const isPending = state === 'pending';

	return (
		<div className="rounded-xl border border-border bg-surface-secondary p-4">
			<div className="flex items-center gap-2.5">
				<svg
					className="w-4 h-4 text-primary"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					strokeWidth={2}
					strokeLinecap="round"
					strokeLinejoin="round"
				>
					<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
				</svg>
				<span className="font-semibold text-sm text-text font-mono flex-1">
					{call.name}
				</span>
				{isPending && (
					<svg
						className="w-4 h-4 text-primary animate-spin"
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
				)}
			</div>

			<div className="mt-2 rounded-md bg-surface-tertiary px-3 py-2 font-mono text-xs text-text-secondary overflow-x-auto">
				<pre className="whitespace-pre-wrap">
					{JSON.stringify(call.args, null, 2)}
				</pre>
			</div>

			{state === 'completed' && result !== undefined && (
				<div className="mt-2">
					<button
						type="button"
						onClick={() => setOpen((o) => !o)}
						className="flex items-center gap-1.5 text-xs font-medium text-text-tertiary hover:text-text-secondary transition-colors"
					>
						<ChevronIcon open={open} />
						Result
					</button>
					<div
						className={`overflow-hidden transition-all duration-200 ${
							open ? 'max-h-60 opacity-100 mt-2' : 'max-h-0 opacity-0'
						}`}
					>
						<ResultBlock content={result.content} />
					</div>
				</div>
			)}
		</div>
	);
}

function ResultBlock({ content }: { content: unknown }) {
	const parsed = tryParseJSON(content);
	const pretty =
		typeof parsed === 'string' ? parsed : JSON.stringify(parsed, null, 2);
	return (
		<div className="rounded-md bg-surface-tertiary px-3 py-2 font-mono text-xs text-text-secondary overflow-x-auto max-h-56 overflow-y-auto">
			<pre className="whitespace-pre-wrap">{pretty}</pre>
		</div>
	);
}

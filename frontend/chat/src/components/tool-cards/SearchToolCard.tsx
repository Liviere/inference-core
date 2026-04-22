import type { ToolCallState, ToolMessage } from './utils';
import { tryParseJSON } from './utils';

type SearchArgs = { query?: string } & Record<string, unknown>;

interface SearchResult {
	title: string;
	url: string;
	snippet: string;
}

/**
 * Renders the results of a web search tool call as a simple list of
 * title/url/snippet triples. Accepts both ``{results: [...]}`` payloads
 * and bare arrays.
 */
export function SearchToolCard({
	call,
	result,
	state,
}: {
	call: { name: string; args: SearchArgs };
	result?: ToolMessage;
	state: ToolCallState;
}) {
	if (state === 'pending') {
		return (
			<div className="rounded-xl border border-border bg-surface p-4">
				<div className="flex items-center gap-3">
					<svg
						className="w-5 h-5 text-primary animate-spin"
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
					<div className="text-sm font-medium text-text-secondary">
						Searching for &ldquo;{call.args.query ?? ''}&rdquo;…
					</div>
				</div>
			</div>
		);
	}

	const parsed = tryParseJSON(result?.content);
	let results: SearchResult[] = [];
	if (Array.isArray(parsed)) {
		results = parsed as SearchResult[];
	} else if (
		parsed &&
		typeof parsed === 'object' &&
		'results' in (parsed as Record<string, unknown>)
	) {
		results = (parsed as { results: SearchResult[] }).results;
	}

	return (
		<div className="rounded-xl border border-border bg-surface p-4">
			<div className="flex items-center gap-2 mb-3">
				<svg
					className="w-4 h-4 text-primary"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					strokeWidth={2}
					strokeLinecap="round"
					strokeLinejoin="round"
				>
					<circle cx="11" cy="11" r="8" />
					<line x1="21" y1="21" x2="16.65" y2="16.65" />
				</svg>
				<span className="text-[11px] font-semibold text-text-tertiary uppercase tracking-wider">
					Search Results
				</span>
			</div>
			<div className="space-y-3">
				{results.map((r, i) => (
					<div key={i} className={i > 0 ? 'pt-3 border-t border-border' : ''}>
						<div className="text-sm font-medium text-text">{r.title}</div>
						<a
							href={r.url}
							target="_blank"
							rel="noopener noreferrer"
							className="text-[11px] text-primary truncate mt-0.5 block"
						>
							{r.url}
						</a>
						<div className="text-xs text-text-tertiary mt-0.5 leading-relaxed">
							{r.snippet}
						</div>
					</div>
				))}
			</div>
		</div>
	);
}

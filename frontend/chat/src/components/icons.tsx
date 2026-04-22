/**
 * Inline SVG icons ported from the LangChain AI-Elements playground.
 *
 * WHY inline: no extra dependency, trivially themeable via currentColor,
 * and these icons are re-used across every pattern (human/ai avatars,
 * reasoning bubble, future tool cards).
 */

export function BotIcon() {
	return (
		<svg
			className="w-4 h-4"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth={2}
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<rect x="3" y="11" width="18" height="10" rx="2" />
			<circle cx="12" cy="5" r="2" />
			<path d="M12 7v4" />
			<line x1="8" y1="16" x2="8" y2="16" />
			<line x1="16" y1="16" x2="16" y2="16" />
		</svg>
	);
}

export function UserIcon() {
	return (
		<svg
			className="w-4 h-4"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth={2}
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
			<circle cx="12" cy="7" r="4" />
		</svg>
	);
}

export function SparklesIcon({
	className = 'w-3.5 h-3.5',
}: {
	className?: string;
}) {
	return (
		<svg
			className={className}
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth={2}
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d="M12 3l1.912 5.813a2 2 0 0 0 1.275 1.275L21 12l-5.813 1.912a2 2 0 0 0-1.275 1.275L12 21l-1.912-5.813a2 2 0 0 0-1.275-1.275L3 12l5.813-1.912a2 2 0 0 0 1.275-1.275L12 3z" />
		</svg>
	);
}

/**
 * Rotating chevron used by collapsibles (e.g. GenericToolCard's result
 * panel). ``open`` toggles a CSS rotate transform.
 */
export function ChevronIcon({ open = false }: { open?: boolean }) {
	return (
		<svg
			className={`w-3.5 h-3.5 transition-transform duration-150 ${open ? 'rotate-90' : ''}`}
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth={2}
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<polyline points="9 18 15 12 9 6" />
		</svg>
	);
}

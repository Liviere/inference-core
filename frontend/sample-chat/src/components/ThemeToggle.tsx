import { useTheme } from '../hooks/useTheme';

/**
 * Inline SVGs kept local to avoid a second icon file; size/stroke match
 * the existing icons.tsx set.
 */
function SunIcon() {
	return (
		<svg
			width="16"
			height="16"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
			aria-hidden="true"
		>
			<circle cx="12" cy="12" r="4" />
			<path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
		</svg>
	);
}

function MoonIcon() {
	return (
		<svg
			width="16"
			height="16"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
			aria-hidden="true"
		>
			<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
		</svg>
	);
}

/**
 * Compact theme toggle. Mirrors the rest of the header buttons visually
 * (same border/bg/hover) so it slots in without looking bolted on.
 */
export function ThemeToggle({ className = '' }: { className?: string }) {
	const { theme, toggle } = useTheme();
	const isDark = theme === 'dark';
	return (
		<button
			onClick={toggle}
			aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
			title={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
			className={`inline-flex items-center justify-center rounded-lg border border-border bg-surface-secondary p-1.5 text-text-secondary hover:border-primary hover:text-primary transition-colors ${className}`}
		>
			{isDark ? <SunIcon /> : <MoonIcon />}
		</button>
	);
}

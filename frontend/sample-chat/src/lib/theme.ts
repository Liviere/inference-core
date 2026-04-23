/**
 * Theme (light/dark) management.
 *
 * WHY: We want the UI to honour OS preference by default but still let the
 * user override it across sessions. Keeping this outside React (module-level
 * init + tiny hook) ensures the theme is applied before the first paint,
 * so there is no flash of the wrong palette.
 */

type Theme = 'light' | 'dark';

const STORAGE_KEY = 'chat.theme';

function readStored(): Theme | null {
	if (typeof localStorage === 'undefined') return null;
	const v = localStorage.getItem(STORAGE_KEY);
	return v === 'light' || v === 'dark' ? v : null;
}

function systemPrefersDark(): boolean {
	if (typeof window === 'undefined' || !window.matchMedia) return false;
	return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function apply(theme: Theme): void {
	if (typeof document === 'undefined') return;
	const root = document.documentElement;
	root.classList.toggle('dark', theme === 'dark');
	root.style.colorScheme = theme;
}

export function getInitialTheme(): Theme {
	return readStored() ?? (systemPrefersDark() ? 'dark' : 'light');
}

export function setTheme(theme: Theme): void {
	apply(theme);
	try {
		localStorage.setItem(STORAGE_KEY, theme);
	} catch {
		/* storage disabled — in-memory only */
	}
}

/**
 * Apply the initial theme as early as possible (called from main.tsx before
 * React mounts). Keeping this a separate call avoids FOUC.
 */
export function bootstrapTheme(): void {
	apply(getInitialTheme());
}

export type { Theme };

import { useCallback, useEffect, useState } from 'react';
import {
	getInitialTheme,
	setTheme as persistTheme,
	type Theme,
} from '../lib/theme';

/**
 * Hook exposing the current theme and a toggle.
 *
 * WHY: A dedicated hook keeps callers (toggle button, header) decoupled from
 * the persistence layer. We also listen to ``prefers-color-scheme`` changes
 * so that users who never made an explicit choice follow their OS.
 */
export function useTheme(): {
	theme: Theme;
	toggle: () => void;
	setTheme: (t: Theme) => void;
} {
	const [theme, setThemeState] = useState<Theme>(() => getInitialTheme());

	useEffect(() => {
		if (typeof window === 'undefined' || !window.matchMedia) return;
		const mq = window.matchMedia('(prefers-color-scheme: dark)');
		const onChange = () => {
			// Only follow the OS when the user has not explicitly chosen yet.
			if (localStorage.getItem('chat.theme') == null) {
				const next: Theme = mq.matches ? 'dark' : 'light';
				setThemeState(next);
				persistTheme(next);
				// Overwrite the just-written storage entry so we keep "follow OS" mode.
				localStorage.removeItem('chat.theme');
			}
		};
		mq.addEventListener('change', onChange);
		return () => mq.removeEventListener('change', onChange);
	}, []);

	const setTheme = useCallback((t: Theme) => {
		setThemeState(t);
		persistTheme(t);
	}, []);

	const toggle = useCallback(() => {
		setThemeState((prev) => {
			const next: Theme = prev === 'dark' ? 'light' : 'dark';
			persistTheme(next);
			return next;
		});
	}, []);

	return { theme, toggle, setTheme };
}

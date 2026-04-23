/**
 * Token storage backed by localStorage.
 *
 * WHY: Phase 3 MVP keeps state simple — JWT lives in localStorage so it
 * survives page reloads.  Production hardening (httpOnly cookie, refresh
 * rotation in the browser) is intentionally deferred.
 */

const TOKEN_KEY = 'ic_access_token';

export function getToken(): string | null {
	return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
	localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
	localStorage.removeItem(TOKEN_KEY);
}

import { useState, type FormEvent } from 'react';
import { ApiError, login } from '../lib/api';
import { setToken } from '../lib/auth';
import { ThemeToggle } from './ThemeToggle';

interface Props {
	onAuthenticated: () => void;
}

/**
 * Username/password form that posts to /api/v1/auth/login.
 *
 * On success the JWT is persisted via setToken() and the parent is
 * notified to swap views.  Error handling is intentionally minimal —
 * surface the backend's 401/422 detail and let the user retry.
 */
export function LoginForm({ onAuthenticated }: Props) {
	const [username, setUsername] = useState('');
	const [password, setPassword] = useState('');
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);

	async function onSubmit(e: FormEvent) {
		e.preventDefault();
		setError(null);
		setLoading(true);
		try {
			const res = await login(username, password);
			setToken(res.access_token);
			onAuthenticated();
		} catch (err) {
			if (err instanceof ApiError) {
				const detail =
					(err.body as { detail?: string } | undefined)?.detail ?? err.message;
				setError(detail);
			} else {
				setError(String(err));
			}
		} finally {
			setLoading(false);
		}
	}

	return (
		<div className="flex min-h-screen items-center justify-center bg-surface px-4">
			<form
				onSubmit={onSubmit}
				className="w-full max-w-sm space-y-4 rounded-2xl border border-border bg-surface-secondary p-6 shadow-lg"
			>
				<header className="flex items-start justify-between gap-2">
					<div>
						<h1 className="text-xl font-semibold text-text">
							Inference Core — Chat
						</h1>
						<p className="text-sm text-text-secondary">Sign in to continue</p>
					</div>
					<ThemeToggle />
				</header>

				<label className="block text-sm">
					<span className="mb-1 block text-text-secondary">
						Username or email
					</span>
					<input
						className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-text placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
						value={username}
						onChange={(e) => setUsername(e.target.value)}
						autoFocus
						required
						autoComplete="username"
					/>
				</label>

				<label className="block text-sm">
					<span className="mb-1 block text-text-secondary">Password</span>
					<input
						type="password"
						className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-text placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
						value={password}
						onChange={(e) => setPassword(e.target.value)}
						required
						autoComplete="current-password"
					/>
				</label>

				{error && (
					<div className="rounded-lg border border-[color:var(--color-error)]/40 bg-[color:var(--color-error)]/10 px-3 py-2 text-sm text-[color:var(--color-error)]">
						{error}
					</div>
				)}

				<button
					type="submit"
					disabled={loading}
					className="w-full rounded-lg bg-primary-dark px-3 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-40 transition-opacity"
				>
					{loading ? 'Signing in…' : 'Sign in'}
				</button>
			</form>
		</div>
	);
}

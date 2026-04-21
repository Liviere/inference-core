import { useState, type FormEvent } from 'react';
import { ApiError, login } from '../lib/api';
import { setToken } from '../lib/auth';

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
		<div className="flex min-h-screen items-center justify-center px-4">
			<form
				onSubmit={onSubmit}
				className="w-full max-w-sm space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-xl"
			>
				<header>
					<h1 className="text-xl font-semibold">Inference Core — Chat</h1>
					<p className="text-sm text-slate-400">Sign in to continue</p>
				</header>

				<label className="block text-sm">
					<span className="mb-1 block text-slate-300">Username or email</span>
					<input
						className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 outline-none focus:border-sky-500"
						value={username}
						onChange={(e) => setUsername(e.target.value)}
						autoFocus
						required
						autoComplete="username"
					/>
				</label>

				<label className="block text-sm">
					<span className="mb-1 block text-slate-300">Password</span>
					<input
						type="password"
						className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 outline-none focus:border-sky-500"
						value={password}
						onChange={(e) => setPassword(e.target.value)}
						required
						autoComplete="current-password"
					/>
				</label>

				{error && (
					<div className="rounded-lg border border-rose-700/50 bg-rose-900/30 px-3 py-2 text-sm text-rose-200">
						{error}
					</div>
				)}

				<button
					type="submit"
					disabled={loading}
					className="w-full rounded-lg bg-sky-500 px-3 py-2 font-medium text-slate-950 transition hover:bg-sky-400 disabled:opacity-50"
				>
					{loading ? 'Signing in…' : 'Sign in'}
				</button>
			</form>
		</div>
	);
}

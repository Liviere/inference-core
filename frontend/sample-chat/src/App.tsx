import { useCallback, useEffect, useState } from 'react';
import { LoginForm } from './components/LoginForm';
import { InstanceSelector } from './components/InstanceSelector';
import { ChatView } from './components/ChatView';
import { clearToken, getToken } from './lib/auth';
import { getAccessMode, type AccessMode, type AgentInstance } from './lib/api';

type View =
	| { kind: 'bootstrap' }
	| { kind: 'bootstrapError'; message: string }
	| { kind: 'login' }
	| { kind: 'select' }
	| { kind: 'chat'; instance: AgentInstance };

/**
 * Tiny stateful router.
 *
 * WHY: The MVP has only a handful of screens and no deep links — pulling in
 * react-router would be overkill. The initial view depends on two things:
 *   1. the backend's ``LLM_API_ACCESS_MODE`` (public skips login entirely)
 *   2. whether a JWT is already persisted in localStorage
 */
export function App() {
	const [mode, setMode] = useState<AccessMode | null>(null);
	const [view, setView] = useState<View>({ kind: 'bootstrap' });

	// Resolve access mode on first mount so we know whether to render the
	// login screen at all. Failure is surfaced rather than silently
	// defaulting — an unreachable backend should be visible to the user.
	useEffect(() => {
		let cancelled = false;
		getAccessMode()
			.then(({ mode: m }) => {
				if (cancelled) return;
				setMode(m);
				if (m === 'public') {
					// Public mode = no login screen. Token (if any) is ignored
					// by the backend for anon mapping, but we leave it in place
					// so authenticated users keep their identity on refresh.
					setView({ kind: 'select' });
				} else {
					setView(getToken() ? { kind: 'select' } : { kind: 'login' });
				}
			})
			.catch((err) => {
				if (cancelled) return;
				setView({
					kind: 'bootstrapError',
					message: err instanceof Error ? err.message : String(err),
				});
			});
		return () => {
			cancelled = true;
		};
	}, []);

	const logout = useCallback(() => {
		clearToken();
		// In public mode logout cannot take the user back to a login screen
		// (there is none) — just reset to the instance picker.
		setView(mode === 'public' ? { kind: 'select' } : { kind: 'login' });
	}, [mode]);

	switch (view.kind) {
		case 'bootstrap':
			return (
				<div className="flex min-h-screen items-center justify-center bg-surface">
					<p className="text-sm text-text-secondary">Connecting to backend…</p>
				</div>
			);
		case 'bootstrapError':
			return (
				<div className="flex min-h-screen items-center justify-center bg-surface px-4">
					<div className="max-w-md rounded-xl border border-[color:var(--color-error)]/40 bg-[color:var(--color-error)]/10 p-4 text-sm text-[color:var(--color-error)]">
						<p className="mb-1 font-semibold">
							Could not reach the backend.
						</p>
						<p className="opacity-80">{view.message}</p>
					</div>
				</div>
			);
		case 'login':
			return <LoginForm onAuthenticated={() => setView({ kind: 'select' })} />;
		case 'select':
			return (
				<InstanceSelector
					onPick={(instance) => setView({ kind: 'chat', instance })}
					onLogout={logout}
					mode={mode ?? 'user'}
				/>
			);
		case 'chat':
			return (
				<ChatView
					instance={view.instance}
					onBack={() => setView({ kind: 'select' })}
					onLogout={logout}
					mode={mode ?? 'user'}
				/>
			);
	}
}

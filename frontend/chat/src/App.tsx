import { useCallback, useState } from 'react';
import { LoginForm } from './components/LoginForm';
import { InstanceSelector } from './components/InstanceSelector';
import { ChatView } from './components/ChatView';
import { clearToken, getToken } from './lib/auth';
import type { AgentInstance } from './lib/api';

type View =
	| { kind: 'login' }
	| { kind: 'select' }
	| { kind: 'chat'; instance: AgentInstance };

/**
 * Tiny stateful router.
 *
 * WHY: The MVP has only three screens and no deep links — pulling in
 * react-router would be overkill.  Token presence drives the initial view.
 */
export function App() {
	const [view, setView] = useState<View>(() =>
		getToken() ? { kind: 'select' } : { kind: 'login' }
	);

	const logout = useCallback(() => {
		clearToken();
		setView({ kind: 'login' });
	}, []);

	switch (view.kind) {
		case 'login':
			return <LoginForm onAuthenticated={() => setView({ kind: 'select' })} />;
		case 'select':
			return (
				<InstanceSelector
					onPick={(instance) => setView({ kind: 'chat', instance })}
					onLogout={logout}
				/>
			);
		case 'chat':
			return (
				<ChatView
					instance={view.instance}
					onBack={() => setView({ kind: 'select' })}
					onLogout={logout}
				/>
			);
	}
}

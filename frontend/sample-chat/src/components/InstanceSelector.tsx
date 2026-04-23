import { useEffect, useState } from 'react';
import {
	ApiError,
	listAgentInstances,
	type AccessMode,
	type AgentInstance,
} from '../lib/api';
import { ThemeToggle } from './ThemeToggle';

interface Props {
	onPick: (instance: AgentInstance) => void;
	onLogout: () => void;
	mode: AccessMode;
}

/**
 * Renders the user's agent instances and lets them pick one to chat with.
 *
 * WHY: Chat resolution depends on the chosen instance — its DB overrides
 * (model, prompts, subagents) are baked into the run-bundle that the
 * subsequent ChatView consumes.  Picking is a single click, no detail view.
 */
export function InstanceSelector({ onPick, onLogout, mode }: Props) {
	const [instances, setInstances] = useState<AgentInstance[] | null>(null);
	const [error, setError] = useState<string | null>(null);
	const publicMode = mode === 'public';

	useEffect(() => {
		let cancelled = false;
		listAgentInstances()
			.then((res) => {
				if (cancelled) return;
				// Default first, then alphabetical — keeps the picker predictable.
				const sorted = [...res.instances].sort((a, b) => {
					if (a.is_default !== b.is_default) return a.is_default ? -1 : 1;
					return a.display_name.localeCompare(b.display_name);
				});
				setInstances(sorted);
			})
			.catch((err) => {
				if (cancelled) return;
				// In public mode there is no login screen to bounce back to,
				// so even a 401 is reported as a plain error.
				if (
					err instanceof ApiError &&
					err.status === 401 &&
					!publicMode
				) {
					onLogout();
					return;
				}
				setError(err instanceof Error ? err.message : String(err));
			});
		return () => {
			cancelled = true;
		};
	}, [onLogout, publicMode]);

	return (
		<div className="min-h-screen bg-surface">
			<div className="mx-auto max-w-3xl px-4 py-8">
				<header className="mb-6 flex items-center justify-between">
					<div>
						<h1 className="text-2xl font-semibold text-text">
							Choose an agent
						</h1>
						<p className="text-sm text-text-secondary">
							{publicMode
								? 'Public access mode — all anonymous visitors share this workspace.'
								: 'Pick one of your configured instances to start a chat.'}
						</p>
					</div>
					<div className="flex gap-2">
						<ThemeToggle />
						{!publicMode && (
							<button
								onClick={onLogout}
								className="rounded-lg border border-border bg-surface-secondary px-3 py-1.5 text-sm text-text-secondary hover:border-primary hover:text-primary transition-colors"
							>
								Sign out
							</button>
						)}
					</div>
				</header>

				{error && (
					<div className="mb-4 rounded-lg border border-[color:var(--color-error)]/40 bg-[color:var(--color-error)]/10 px-3 py-2 text-sm text-[color:var(--color-error)]">
						{error}
					</div>
				)}

				{instances === null && !error && (
					<p className="text-sm text-text-secondary">Loading instances…</p>
				)}

				{instances && instances.length === 0 && (
					<div className="rounded-xl border border-border bg-surface-secondary p-6 text-center text-text-secondary">
						No agent instances yet. Create one via the backend API (
						<code className="text-text">POST /api/v1/agent-instances</code>).
					</div>
				)}

				<ul className="grid gap-3 sm:grid-cols-2">
					{instances?.map((inst) => (
						<li key={inst.id}>
							<button
								onClick={() => onPick(inst)}
								className="group flex w-full flex-col items-start gap-1 rounded-xl border border-border bg-surface-secondary p-4 text-left transition hover:border-primary hover:bg-surface-tertiary"
							>
								<div className="flex w-full items-center justify-between">
									<span className="font-semibold text-text group-hover:text-primary">
										{inst.display_name}
									</span>
									{inst.is_default && (
										<span className="rounded bg-primary/15 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-primary">
											default
										</span>
									)}
								</div>
								<span className="text-xs text-text-tertiary">
									{inst.base_agent_name}
									{inst.is_deepagent && ' · deep'}
									{inst.primary_model && ` · ${inst.primary_model}`}
								</span>
								{inst.description && (
									<span className="mt-1 line-clamp-2 text-xs text-text-secondary">
										{inst.description}
									</span>
								)}
							</button>
						</li>
					))}
				</ul>
			</div>
		</div>
	);
}

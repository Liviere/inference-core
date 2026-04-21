import { useEffect, useState } from 'react';
import { ApiError, listAgentInstances, type AgentInstance } from '../lib/api';

interface Props {
	onPick: (instance: AgentInstance) => void;
	onLogout: () => void;
}

/**
 * Renders the user's agent instances and lets them pick one to chat with.
 *
 * WHY: Chat resolution depends on the chosen instance — its DB overrides
 * (model, prompts, subagents) are baked into the run-bundle that the
 * subsequent ChatView consumes.  Picking is a single click, no detail view.
 */
export function InstanceSelector({ onPick, onLogout }: Props) {
	const [instances, setInstances] = useState<AgentInstance[] | null>(null);
	const [error, setError] = useState<string | null>(null);

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
				if (err instanceof ApiError && err.status === 401) {
					onLogout();
					return;
				}
				setError(err instanceof Error ? err.message : String(err));
			});
		return () => {
			cancelled = true;
		};
	}, [onLogout]);

	return (
		<div className="mx-auto max-w-3xl px-4 py-8">
			<header className="mb-6 flex items-center justify-between">
				<div>
					<h1 className="text-2xl font-semibold">Choose an agent</h1>
					<p className="text-sm text-slate-400">
						Pick one of your configured instances to start a chat.
					</p>
				</div>
				<button
					onClick={onLogout}
					className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm hover:bg-slate-800"
				>
					Sign out
				</button>
			</header>

			{error && (
				<div className="mb-4 rounded-lg border border-rose-700/50 bg-rose-900/30 px-3 py-2 text-sm text-rose-200">
					{error}
				</div>
			)}

			{instances === null && !error && (
				<p className="text-sm text-slate-400">Loading instances…</p>
			)}

			{instances && instances.length === 0 && (
				<div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 text-center text-slate-400">
					No agent instances yet. Create one via the backend API (
					<code className="text-slate-300">POST /api/v1/agent-instances</code>).
				</div>
			)}

			<ul className="grid gap-3 sm:grid-cols-2">
				{instances?.map((inst) => (
					<li key={inst.id}>
						<button
							onClick={() => onPick(inst)}
							className="group flex w-full flex-col items-start gap-1 rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-left transition hover:border-sky-600 hover:bg-slate-900"
						>
							<div className="flex w-full items-center justify-between">
								<span className="font-semibold group-hover:text-sky-300">
									{inst.display_name}
								</span>
								{inst.is_default && (
									<span className="rounded bg-sky-500/20 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-sky-300">
										default
									</span>
								)}
							</div>
							<span className="text-xs text-slate-500">
								{inst.base_agent_name}
								{inst.is_deepagent && ' · deep'}
								{inst.primary_model && ` · ${inst.primary_model}`}
							</span>
							{inst.description && (
								<span className="mt-1 line-clamp-2 text-xs text-slate-400">
									{inst.description}
								</span>
							)}
						</button>
					</li>
				))}
			</ul>
		</div>
	);
}

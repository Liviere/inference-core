import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { useStream } from '@langchain/langgraph-sdk/react';
import {
	ApiError,
	getRunBundle,
	type AgentInstance,
	type RunBundleResponse,
} from '../lib/api';

interface Props {
	instance: AgentInstance;
	onBack: () => void;
	onLogout: () => void;
}

/**
 * Live chat backed by LangGraph's ``useStream`` directly against the
 * Agent Server.
 *
 * Flow:
 *   1. Fetch the run-bundle from the backend (auth, configurable, URL).
 *   2. Mount ``useStream`` with that bundle.
 *   3. Render messages + a single textarea/input that calls ``thread.submit``
 *      with the bundle's ``configurable`` so middleware (InstanceConfig,
 *      Memory, SubagentConfig…) sees the right per-user overrides.
 *
 * Stream mode "messages-tuple" gives us partial token streaming for the AI
 * response — sufficient for an MVP without custom UI for tool/reasoning.
 */
export function ChatView({ instance, onBack, onLogout }: Props) {
	const [bundle, setBundle] = useState<RunBundleResponse | null>(null);
	const [bundleError, setBundleError] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		setBundle(null);
		setBundleError(null);
		getRunBundle(instance.id)
			.then((b) => {
				if (!cancelled) setBundle(b);
			})
			.catch((err) => {
				if (cancelled) return;
				if (err instanceof ApiError && err.status === 401) {
					onLogout();
					return;
				}
				setBundleError(err instanceof Error ? err.message : String(err));
			});
		return () => {
			cancelled = true;
		};
	}, [instance.id, onLogout]);

	if (bundleError) {
		return (
			<ChatShell instance={instance} onBack={onBack} onLogout={onLogout}>
				<div className="rounded-lg border border-rose-700/50 bg-rose-900/30 p-4 text-sm text-rose-200">
					Failed to load run bundle: {bundleError}
				</div>
			</ChatShell>
		);
	}

	if (!bundle) {
		return (
			<ChatShell instance={instance} onBack={onBack} onLogout={onLogout}>
				<p className="text-sm text-slate-400">
					Connecting to the Agent Server…
				</p>
			</ChatShell>
		);
	}

	return (
		<ChatShell instance={instance} onBack={onBack} onLogout={onLogout}>
			<ChatStream bundle={bundle} />
		</ChatShell>
	);
}

// --------------------------------------------------------------------------
// Layout shell — separated so the stream component can remount on bundle
// changes without disturbing surrounding state (back/logout buttons).
// --------------------------------------------------------------------------

function ChatShell({
	instance,
	onBack,
	onLogout,
	children,
}: {
	instance: AgentInstance;
	onBack: () => void;
	onLogout: () => void;
	children: React.ReactNode;
}) {
	return (
		<div className="mx-auto flex h-screen max-w-3xl flex-col px-4 py-4">
			<header className="mb-3 flex items-center justify-between">
				<div>
					<h1 className="text-lg font-semibold">{instance.display_name}</h1>
					<p className="text-xs text-slate-500">
						{instance.base_agent_name}
						{instance.primary_model && ` · ${instance.primary_model}`}
					</p>
				</div>
				<div className="flex gap-2">
					<button
						onClick={onBack}
						className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm hover:bg-slate-800"
					>
						← Agents
					</button>
					<button
						onClick={onLogout}
						className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm hover:bg-slate-800"
					>
						Sign out
					</button>
				</div>
			</header>
			<div className="flex-1 min-h-0">{children}</div>
		</div>
	);
}

// --------------------------------------------------------------------------
// Stream container — keeps useStream isolated so it only initialises after
// the bundle is loaded (useStream cannot be conditionally called).
// --------------------------------------------------------------------------

interface StreamProps {
	bundle: RunBundleResponse;
}

interface MessageLike {
	id?: string;
	type?: string;
	role?: string;
	content?: unknown;
}

function ChatStream({ bundle }: StreamProps) {
	const [input, setInput] = useState('');
	const scrollRef = useRef<HTMLDivElement>(null);

	const defaultHeaders = useMemo<Record<string, string>>(() => {
		const h: Record<string, string> = {};
		if (bundle.access_token) h.Authorization = `Bearer ${bundle.access_token}`;
		return h;
	}, [bundle.access_token]);

	const thread = useStream<{ messages: MessageLike[] }>({
		apiUrl: bundle.agent_server_url,
		assistantId: bundle.assistant_id,
		messagesKey: 'messages',
		defaultHeaders,
	});

	const messages = (thread.messages ?? []) as MessageLike[];

	// Auto-scroll on new messages or while streaming.
	useEffect(() => {
		const el = scrollRef.current;
		if (el) el.scrollTop = el.scrollHeight;
	}, [messages.length, thread.isLoading]);

	function onSubmit(e: FormEvent) {
		e.preventDefault();
		const text = input.trim();
		if (!text || thread.isLoading) return;
		setInput('');
		thread.submit(
			{ messages: [{ type: 'human', content: text }] },
			{ config: bundle.config }
		);
	}

	return (
		<div className="flex h-full flex-col rounded-2xl border border-slate-800 bg-slate-900/40">
			<div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto p-4">
				{messages.length === 0 && (
					<p className="text-sm text-slate-500">
						Say hello — the agent is ready.
					</p>
				)}
				{messages.map((m, idx) => (
					<MessageBubble key={m.id ?? idx} msg={m} />
				))}
				{thread.isLoading && (
					<p className="text-xs text-slate-500 italic">Streaming…</p>
				)}
				{thread.error != null && (
					<div className="rounded-lg border border-rose-700/50 bg-rose-900/30 px-3 py-2 text-sm text-rose-200">
						{formatThreadError(thread.error)}
					</div>
				)}
			</div>

			<form
				onSubmit={onSubmit}
				className="flex gap-2 border-t border-slate-800 p-3"
			>
				<input
					value={input}
					onChange={(e) => setInput(e.target.value)}
					placeholder="Type a message…"
					disabled={thread.isLoading}
					className="flex-1 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 outline-none focus:border-sky-500 disabled:opacity-50"
				/>
				<button
					type="submit"
					disabled={thread.isLoading || !input.trim()}
					className="rounded-lg bg-sky-500 px-4 py-2 font-medium text-slate-950 transition hover:bg-sky-400 disabled:opacity-50"
				>
					Send
				</button>
				{thread.isLoading && (
					<button
						type="button"
						onClick={() => thread.stop()}
						className="rounded-lg border border-slate-700 px-3 py-2 text-sm hover:bg-slate-800"
					>
						Stop
					</button>
				)}
			</form>
		</div>
	);
}

function MessageBubble({ msg }: { msg: MessageLike }) {
	const role = msg.type ?? msg.role ?? 'ai';
	const isUser = role === 'human' || role === 'user';
	const text = renderContent(msg.content);

	return (
		<div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
			<div
				className={`max-w-[80%] whitespace-pre-wrap break-words rounded-2xl px-3 py-2 text-sm ${
					isUser ? 'bg-sky-600 text-slate-950' : 'bg-slate-800 text-slate-100'
				}`}
			>
				{text || <span className="opacity-50">…</span>}
			</div>
		</div>
	);
}

/**
 * LangChain message ``content`` is either a string or a list of typed blocks
 * (``text``, ``tool_use``, ``thinking`` …).  For the MVP we render only the
 * text payloads and drop the rest — keeps the surface tiny.
 */
function renderContent(content: unknown): string {
	if (typeof content === 'string') return content;
	if (Array.isArray(content)) {
		return content
			.map((block) => {
				if (typeof block === 'string') return block;
				if (block && typeof block === 'object') {
					const b = block as { type?: string; text?: string };
					if (b.type === 'text' && typeof b.text === 'string') return b.text;
				}
				return '';
			})
			.filter(Boolean)
			.join('');
	}
	return '';
}

function formatThreadError(err: unknown): string {
	if (!err) return 'Unknown error';
	if (err instanceof Error) return err.message;
	if (typeof err === 'string') return err;
	try {
		return JSON.stringify(err);
	} catch {
		return String(err);
	}
}

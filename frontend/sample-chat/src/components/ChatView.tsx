import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useStream } from '@langchain/react';
import type { ToolCallWithResult } from '@langchain/react';
import { HumanMessage, AIMessage } from 'langchain';

import {
	ApiError,
	getRunBundle,
	type AccessMode,
	type AgentInstance,
	type RunBundleResponse,
} from '../lib/api';
import { ChatContainer } from './chat/ChatContainer';
import { AIBubble, HumanBubble, ThinkingBubble } from './chat/Bubble';
import { ChatInput } from './chat/ChatInput';
import { TypingIndicator } from './chat/TypingIndicator';
import { PresetPrompts } from './chat/PresetPrompts';
import { HistorySidebar } from './chat/HistorySidebar';
import { Markdown } from './Markdown';
import { ThemeToggle } from './ThemeToggle';
import { ToolCallCard } from './tool-cards/ToolCallCard';
import { getPresetPrompts } from '../lib/presetPrompts';
import { useThreadHistory } from '../hooks/useThreadHistory';

interface Props {
	instance: AgentInstance;
	onBack: () => void;
	onLogout: () => void;
	mode: AccessMode;
}

/**
 * Live chat backed by @langchain/react's ``useStream`` against the
 * LangGraph Agent Server, layered on top of our run-bundle flow.
 *
 * Flow:
 *   1. Fetch the run-bundle from FastAPI (auth, configurable, URL).
 *   2. Mount ``useStream`` with the bundle's apiUrl + assistantId and
 *      a controlled ``threadId`` so the left sidebar can switch/resume
 *      prior conversations stored locally in localStorage.
 *   3. When the user submits the first message of a fresh thread, we
 *      capture it, wait for ``onThreadId`` to fire with the id issued
 *      by the Agent Server, and persist a small catalog entry so the
 *      sidebar can list / resume the thread later.
 */
export function ChatView({ instance, onBack, onLogout, mode }: Props) {
	const [bundle, setBundle] = useState<RunBundleResponse | null>(null);
	const [bundleError, setBundleError] = useState<string | null>(null);
	const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
	const [sidebarOpen, setSidebarOpen] = useState(false);
	const publicMode = mode === 'public';

	const history = useThreadHistory(instance.id);

	// Reset active thread whenever the user switches to a different agent
	// instance — different instance = different history bucket.
	useEffect(() => {
		setActiveThreadId(null);
	}, [instance.id]);

	// Fetch the bundle once per instance. We intentionally do NOT include
	// ``activeThreadId`` in the dependency list: swapping threads is handled
	// by the controlled ``threadId`` prop on useStream, which already knows
	// how to rehydrate state from the Agent Server's checkpointer.
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
				if (err instanceof ApiError && err.status === 401 && !publicMode) {
					onLogout();
					return;
				}
				setBundleError(err instanceof Error ? err.message : String(err));
			});
		return () => {
			cancelled = true;
		};
	}, [instance.id, onLogout, publicMode]);

	const handleSelectThread = useCallback((threadId: string) => {
		setActiveThreadId(threadId);
		setSidebarOpen(false);
	}, []);

	const handleStartNewThread = useCallback(() => {
		setActiveThreadId(null);
		setSidebarOpen(false);
	}, []);

	const handleDeleteThread = useCallback(
		(id: string) => {
			history.remove(id);
			setActiveThreadId((current) => (current === id ? null : current));
		},
		[history]
	);

	const handleMissingThread = useCallback(
		(id: string) => {
			history.remove(id);
			setActiveThreadId(null);
		},
		[history]
	);

	const handleThreadCreated = useCallback(
		(id: string, firstMessage: string) => {
			history.registerNew(id, firstMessage);
			setActiveThreadId(id);
		},
		[history]
	);

	return (
		<ChatShell
			instance={instance}
			onBack={onBack}
			onLogout={onLogout}
			publicMode={publicMode}
			onToggleSidebar={() => setSidebarOpen((v) => !v)}
			sidebarOpen={sidebarOpen}
		>
			<div className="grid h-full grid-cols-1 md:grid-cols-[260px_1fr]">
				{/* Desktop sidebar (always visible on md+). */}
				<div className="hidden h-full min-h-0 md:block">
					<HistorySidebar
						entries={history.entries}
						activeThreadId={activeThreadId}
						onSelect={handleSelectThread}
						onNewThread={handleStartNewThread}
						onRename={history.rename}
						onDelete={handleDeleteThread}
					/>
				</div>

				{/* Mobile drawer — absolute overlay toggled by the header button. */}
				{sidebarOpen && (
					<>
						<button
							type="button"
							aria-label="Close sidebar"
							onClick={() => setSidebarOpen(false)}
							className="absolute inset-0 z-30 bg-black/40 md:hidden"
						/>
						<div className="absolute inset-y-0 left-0 z-40 w-72 max-w-[80%] md:hidden">
							<HistorySidebar
								entries={history.entries}
								activeThreadId={activeThreadId}
								onSelect={handleSelectThread}
								onNewThread={handleStartNewThread}
								onRename={history.rename}
								onDelete={handleDeleteThread}
							/>
						</div>
					</>
				)}

				<div className="min-h-0">
					{bundleError ? (
						<div className="m-4 rounded-lg border border-[color:var(--color-error)]/40 bg-[color:var(--color-error)]/10 p-4 text-sm text-[color:var(--color-error)]">
							Failed to load run bundle: {bundleError}
						</div>
					) : !bundle ? (
						<p className="p-4 text-sm text-text-secondary">
							Connecting to the Agent Server…
						</p>
					) : (
						<ChatStream
							bundle={bundle}
							activeThreadId={activeThreadId}
							onThreadCreated={handleThreadCreated}
							onThreadTouched={history.touch}
							onNewThread={handleStartNewThread}
							onMissingThread={handleMissingThread}
						/>
					)}
				</div>
			</div>
		</ChatShell>
	);
}

// --------------------------------------------------------------------------
// Layout shell — header (agent meta + back/logout + sidebar toggle) + slot
// for the sidebar-grid content. Kept separate so the stream remounts
// cleanly on bundle changes.
// --------------------------------------------------------------------------
function ChatShell({
	instance,
	onBack,
	onLogout,
	publicMode,
	onToggleSidebar,
	sidebarOpen,
	children,
}: {
	instance: AgentInstance;
	onBack: () => void;
	onLogout: () => void;
	publicMode: boolean;
	onToggleSidebar: () => void;
	sidebarOpen: boolean;
	children: React.ReactNode;
}) {
	return (
		<div className="mx-auto flex h-screen max-w-5xl flex-col px-4 py-4">
			<header className="mb-3 flex items-center justify-between gap-2">
				<div className="flex min-w-0 items-center gap-2">
					<button
						type="button"
						onClick={onToggleSidebar}
						aria-label={sidebarOpen ? 'Hide history' : 'Show history'}
						aria-expanded={sidebarOpen}
						className="rounded-lg border border-border bg-surface-secondary p-2 text-text-secondary hover:border-primary hover:text-primary transition-colors md:hidden"
					>
						<HamburgerIcon />
					</button>
					<div className="min-w-0">
						<h1 className="truncate text-lg font-semibold text-text">
							{instance.display_name}
						</h1>
						<p className="truncate text-xs text-text-tertiary">
							{instance.base_agent_name}
							{instance.primary_model && ` · ${instance.primary_model}`}
						</p>
					</div>
				</div>
				<div className="flex shrink-0 gap-2">
					<ThemeToggle />
					<button
						onClick={onBack}
						className="rounded-lg border border-border bg-surface-secondary px-3 py-1.5 text-sm text-text-secondary hover:border-primary hover:text-primary transition-colors"
					>
						← Agents
					</button>
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
			<div className="relative min-h-0 flex-1 overflow-hidden rounded-xl border border-border bg-surface">
				{children}
			</div>
		</div>
	);
}

function HamburgerIcon() {
	return (
		<svg
			width="16"
			height="16"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
			aria-hidden="true"
		>
			<line x1="4" y1="6" x2="20" y2="6" />
			<line x1="4" y1="12" x2="20" y2="12" />
			<line x1="4" y1="18" x2="20" y2="18" />
		</svg>
	);
}

// --------------------------------------------------------------------------
// Stream container — isolated so useStream is only called once the
// bundle is available (hook order must be stable).
// --------------------------------------------------------------------------

function resolveApiUrl(bundleUrl: string): string {
	// Same-origin proxy switch: keeps browser CORS out of the picture in dev.
	const useProxy = import.meta.env.VITE_USE_AGENT_PROXY === 'true';
	if (useProxy && typeof window !== 'undefined') {
		return `${window.location.origin}/api/langgraph`;
	}
	return bundleUrl;
}

interface ChatStreamProps {
	bundle: RunBundleResponse;
	activeThreadId: string | null;
	onThreadCreated: (threadId: string, firstMessage: string) => void;
	onThreadTouched: (threadId: string) => void;
	onNewThread: () => void;
	onMissingThread: (threadId: string) => void;
}

function ChatStream({
	bundle,
	activeThreadId,
	onThreadCreated,
	onThreadTouched,
	onNewThread,
	onMissingThread,
}: ChatStreamProps) {
	const apiUrl = useMemo(
		() => resolveApiUrl(bundle.agent_server_url),
		[bundle.agent_server_url]
	);
	const presetPrompts = useMemo(
		() => getPresetPrompts(bundle.base_agent_name),
		[bundle.base_agent_name]
	);

	const defaultHeaders = useMemo<Record<string, string>>(() => {
		const h: Record<string, string> = {};
		if (bundle.access_token) h.Authorization = `Bearer ${bundle.access_token}`;
		return h;
	}, [bundle.access_token]);

	// Buffer for the first user message of a fresh thread — we need the
	// text to generate a sidebar title once ``onThreadId`` fires with the
	// id the Agent Server issued. Kept in a ref to avoid re-renders and
	// stale-closure bugs inside the useStream callbacks.
	const pendingFirstMessageRef = useRef<string | null>(null);

	// Latest onThreadCreated ref — prevents the useStream hook from
	// re-initialising every render just because the parent passed a new
	// function identity.
	const onThreadCreatedRef = useRef(onThreadCreated);
	useEffect(() => {
		onThreadCreatedRef.current = onThreadCreated;
	}, [onThreadCreated]);

	// NOTE: we intentionally omit the generic type argument — we don't ship
	// a compiled langchain graph type on the frontend. ``HumanMessage.isInstance``
	// / ``AIMessage.isInstance`` work at runtime regardless of the static type.
	const stream = useStream({
		apiUrl,
		assistantId: bundle.assistant_id,
		defaultHeaders,
		threadId: activeThreadId,
		onThreadId: (id: string) => {
			const pending = pendingFirstMessageRef.current;
			pendingFirstMessageRef.current = null;
			// Only register if we captured a first message locally — the
			// hook also calls onThreadId after ``switchThread``, where no
			// new conversation has started yet.
			if (pending) onThreadCreatedRef.current(id, pending);
		},
	});

	// ``toolCalls`` is exposed at runtime by the stream hook but only appears
	// in the static types when a typed graph parameter is supplied. Since we
	// don't ship compiled graph types on the frontend, we re-type it here.
	const toolCalls =
		(
			stream as unknown as {
				toolCalls?: ToolCallWithResult[];
			}
		).toolCalls ?? [];

	const handleSubmit = (text: string) => {
		if (!activeThreadId) {
			pendingFirstMessageRef.current = text;
		} else {
			onThreadTouched(activeThreadId);
		}
		stream.submit(
			{ messages: [{ type: 'human' as const, content: text }] },
			{ config: bundle.config }
		);
	};

	const handleNewThread = () => {
		pendingFirstMessageRef.current = null;
		// ``switchThread(null)`` clears the hook's internal thread ref so
		// the next submit creates a fresh thread. Parent state is updated
		// via onNewThread so the sidebar selection clears too.
		stream.switchThread(null);
		onNewThread();
	};

	// If the Agent Server reports the thread doesn't exist anymore (e.g.
	// checkpoint DB was wiped in dev), evict the stale entry from history.
	useEffect(() => {
		const err = stream.error;
		if (!err || !activeThreadId) return;
		const msg = formatStreamError(err).toLowerCase();
		if (msg.includes('404') || msg.includes('not found')) {
			onMissingThread(activeThreadId);
		}
	}, [stream.error, activeThreadId, onMissingThread]);

	const messages = stream.messages ?? [];

	return (
		<ChatContainer
			input={
				<ChatInput
					onSubmit={handleSubmit}
					disabled={stream.isLoading}
					onNewThread={messages.length > 0 ? handleNewThread : undefined}
				/>
			}
		>
			{messages.length === 0 && (
				<PresetPrompts prompts={presetPrompts} onSelect={handleSubmit} />
			)}

			{messages.map((msg, idx) => {
				const key = msg.id ?? idx;
				if (HumanMessage.isInstance(msg)) {
					return (
						<HumanBubble key={key}>
							<Markdown>{msg.text}</Markdown>
						</HumanBubble>
					);
				}
				if (AIMessage.isInstance(msg)) {
					// Tool calls attached to THIS AI message. We match by id so
					// multi-turn conversations don't render a tool card under the
					// wrong message.
					const callIds = new Set(
						(msg.tool_calls ?? [])
							.map((tc) => tc.id)
							.filter(Boolean) as string[]
					);
					const toolCallsForMessage = toolCalls.filter((tc) =>
						callIds.has(tc.call.id ?? '')
					);
					const hasText = typeof msg.text === 'string' && msg.text.length > 0;

					// Reasoning tokens: extract the model's chain-of-thought.
					//
					// Sources are provider-dependent and we accept either:
					//   1. Standardized `contentBlocks` with `type === 'reasoning'`
					//      (OpenAI o-series, Anthropic extended thinking — when the
					//       langchain adapter already normalizes them).
					//   2. `additional_kwargs.reasoning_content` — Fireworks, some
					//      DeepInfra models, vLLM-compatible servers.
					//   3. `additional_kwargs.reasoning` — a few other adapters.
					//
					// We prefer contentBlocks when present (gives us multiple
					// chunks in order) and fall back to the flat string fields.
					const reasoningFromBlocks = (
						(
							msg as unknown as {
								contentBlocks?: Array<{ type: string; reasoning?: string }>;
							}
						).contentBlocks ?? []
					)
						.filter(
							(b) =>
								b.type === 'reasoning' &&
								typeof b.reasoning === 'string' &&
								b.reasoning.trim().length > 0
						)
						.map((b) => b.reasoning as string)
						.join('');

					const additionalKwargs = (
						msg as unknown as {
							additional_kwargs?: Record<string, unknown>;
						}
					).additional_kwargs;
					const reasoningFromKwargs =
						typeof additionalKwargs?.reasoning_content === 'string'
							? (additionalKwargs.reasoning_content as string)
							: typeof additionalKwargs?.reasoning === 'string'
								? (additionalKwargs.reasoning as string)
								: '';

					const reasoning =
						reasoningFromBlocks.length > 0
							? reasoningFromBlocks
							: reasoningFromKwargs;

					// "Reasoning phase" = model is still streaming and hasn't
					// produced any text blocks yet. Once text shows up the
					// thinking bubble stops spinning.
					const isLastMessage = idx === messages.length - 1;
					const isReasoningStreaming =
						stream.isLoading && isLastMessage && !hasText;

					return (
						<div key={key} className="space-y-2">
							{reasoning.length > 0 && (
								<ThinkingBubble
									content={reasoning}
									isStreaming={isReasoningStreaming}
								/>
							)}
							{hasText && (
								<AIBubble>
									<Markdown>{msg.text}</Markdown>
								</AIBubble>
							)}
							{toolCallsForMessage.map((tc) => (
								<div key={tc.id} className="pl-9 max-w-[80%]">
									<ToolCallCard toolCall={tc} />
								</div>
							))}
						</div>
					);
				}
				return null;
			})}

			{stream.isLoading && <TypingIndicator />}

			{stream.error != null && (
				<div className="rounded-lg border border-[color:var(--color-error)]/40 bg-[color:var(--color-error)]/10 px-3 py-2 text-sm text-[color:var(--color-error)]">
					{formatStreamError(stream.error)}
				</div>
			)}
		</ChatContainer>
	);
}

function formatStreamError(err: unknown): string {
	if (!err) return 'Unknown error';
	if (err instanceof Error) return err.message;
	if (typeof err === 'string') return err;
	try {
		return JSON.stringify(err);
	} catch {
		return String(err);
	}
}

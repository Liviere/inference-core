import { useEffect, useMemo, useState } from 'react';
import { useStream } from '@langchain/react';
import type { ToolCallWithResult } from '@langchain/react';
import { HumanMessage, AIMessage } from 'langchain';

import {
	ApiError,
	getRunBundle,
	type AgentInstance,
	type RunBundleResponse,
} from '../lib/api';
import { ChatContainer } from './chat/ChatContainer';
import { AIBubble, HumanBubble, ThinkingBubble } from './chat/Bubble';
import { ChatInput } from './chat/ChatInput';
import { TypingIndicator } from './chat/TypingIndicator';
import { PresetPrompts } from './chat/PresetPrompts';
import { Markdown } from './Markdown';
import { ThemeToggle } from './ThemeToggle';
import { ToolCallCard } from './tool-cards/ToolCallCard';

interface Props {
	instance: AgentInstance;
	onBack: () => void;
	onLogout: () => void;
}

const PRESETS = [
	"What's the weather like in San Francisco?",
	'What is (25 * 4) + 130 / 2?',
	'Search for LangChain documentation',
];

/**
 * Live chat backed by @langchain/react's ``useStream`` against the
 * LangGraph Agent Server, layered on top of our run-bundle flow.
 *
 * Flow:
 *   1. Fetch the run-bundle from FastAPI (auth, configurable, URL).
 *   2. Mount ``useStream`` with the bundle's apiUrl + assistantId.
 *      Optional same-origin proxy (VITE_USE_AGENT_PROXY) rewrites
 *      the URL to ``/api/langgraph`` — avoids browser CORS in dev.
 *   3. Submit user messages with ``config: bundle.config`` so that the
 *      InstanceConfigMiddleware on the server side sees the per-user
 *      overrides (primary_model, system_prompt_override, subagent_configs).
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

	return (
		<ChatShell instance={instance} onBack={onBack} onLogout={onLogout}>
			{bundleError ? (
				<div className="m-4 rounded-lg border border-[color:var(--color-error)]/40 bg-[color:var(--color-error)]/10 p-4 text-sm text-[color:var(--color-error)]">
					Failed to load run bundle: {bundleError}
				</div>
			) : !bundle ? (
				<p className="p-4 text-sm text-text-secondary">
					Connecting to the Agent Server…
				</p>
			) : (
				<ChatStream bundle={bundle} />
			)}
		</ChatShell>
	);
}

// --------------------------------------------------------------------------
// Layout shell — header (agent meta + back/logout) + slot for the stream.
// Kept separate so the stream remounts cleanly on bundle changes.
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
					<h1 className="text-lg font-semibold text-text">
						{instance.display_name}
					</h1>
					<p className="text-xs text-text-tertiary">
						{instance.base_agent_name}
						{instance.primary_model && ` · ${instance.primary_model}`}
					</p>
				</div>
				<div className="flex gap-2">
					<ThemeToggle />
					<button
						onClick={onBack}
						className="rounded-lg border border-border bg-surface-secondary px-3 py-1.5 text-sm text-text-secondary hover:border-primary hover:text-primary transition-colors"
					>
						← Agents
					</button>
					<button
						onClick={onLogout}
						className="rounded-lg border border-border bg-surface-secondary px-3 py-1.5 text-sm text-text-secondary hover:border-primary hover:text-primary transition-colors"
					>
						Sign out
					</button>
				</div>
			</header>
			<div className="flex-1 min-h-0 rounded-xl border border-border bg-surface overflow-hidden">
				{children}
			</div>
		</div>
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

function ChatStream({ bundle }: { bundle: RunBundleResponse }) {
	const apiUrl = useMemo(
		() => resolveApiUrl(bundle.agent_server_url),
		[bundle.agent_server_url]
	);

	const defaultHeaders = useMemo<Record<string, string>>(() => {
		const h: Record<string, string> = {};
		if (bundle.access_token) h.Authorization = `Bearer ${bundle.access_token}`;
		return h;
	}, [bundle.access_token]);

	// NOTE: we intentionally omit the generic type argument — we don't ship
	// a compiled langchain graph type on the frontend. ``HumanMessage.isInstance``
	// / ``AIMessage.isInstance`` work at runtime regardless of the static type.
	const stream = useStream({
		apiUrl,
		assistantId: bundle.assistant_id,
		defaultHeaders,
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
		stream.submit(
			{ messages: [{ type: 'human' as const, content: text }] },
			{ config: bundle.config }
		);
	};

	const handleNewThread = () => {
		// ``switchThread(null)`` starts a fresh thread on the next submit.
		stream.switchThread(null);
	};

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
				<PresetPrompts prompts={PRESETS} onSelect={handleSubmit} />
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

import type { ReactNode } from 'react';
import { BotIcon, SparklesIcon, UserIcon } from '../icons';

interface BubbleProps {
	children: ReactNode;
}

function Avatar({ variant }: { variant: 'human' | 'ai' }) {
	return (
		<div
			className={
				'shrink-0 w-7 h-7 rounded-full flex items-center justify-center ' +
				(variant === 'human'
					? 'bg-surface-tertiary border border-border'
					: 'bg-surface-tertiary border border-border text-primary')
			}
		>
			{variant === 'human' ? <UserIcon /> : <BotIcon />}
		</div>
	);
}

const CHAT_TURN_TEST_ID = 'sdk-preview-chat-turn' as const;

export function HumanBubble({ children }: BubbleProps) {
	return (
		<div
			className="flex justify-end items-end gap-2"
			data-testid={CHAT_TURN_TEST_ID}
		>
			<div className="max-w-[80%] rounded-xl rounded-br-sm bg-primary-dark text-white px-4 py-3 text-sm leading-relaxed">
				{children}
			</div>
			<Avatar variant="human" />
		</div>
	);
}

export function AIBubble({ children }: BubbleProps) {
	return (
		<div
			className="flex justify-start items-end gap-2"
			data-testid={CHAT_TURN_TEST_ID}
		>
			<Avatar variant="ai" />
			<div className="max-w-[80%] rounded-xl rounded-bl-sm bg-surface-secondary border border-border text-text px-4 py-3 text-sm leading-relaxed">
				{children}
			</div>
		</div>
	);
}

/**
 * Prepared for the upcoming reasoning-tokens pattern (Phase 6). Kept in
 * the shared primitives so next patterns can reuse it without touching
 * ChatView again.
 */
interface ThinkingBubbleProps {
	content: string;
	isStreaming: boolean;
}

export function ThinkingBubble({ content, isStreaming }: ThinkingBubbleProps) {
	return (
		<div
			className="flex justify-start items-end gap-2"
			data-testid={CHAT_TURN_TEST_ID}
		>
			<Avatar variant="ai" />
			<div className="max-w-[80%] rounded-xl rounded-bl-sm bg-surface-secondary border border-border px-4 py-2.5 text-sm">
				<div className="flex items-center gap-1.5 mb-1.5 text-xs font-medium text-primary">
					<SparklesIcon
						className={`w-3.5 h-3.5 ${isStreaming ? 'animate-pulse' : ''}`}
					/>
					<span>{isStreaming ? 'Thinking…' : 'Thought'}</span>
					<span className="text-text-tertiary font-normal">
						· {content.length} chars
					</span>
				</div>
				<pre className="whitespace-pre-wrap text-xs font-mono text-text-secondary leading-relaxed m-0">
					{content}
				</pre>
			</div>
		</div>
	);
}

import { useState, type ReactNode } from 'react';
import { BotIcon, ChevronIcon, SparklesIcon, UserIcon } from '../icons';

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
 * Collapsible reasoning / chain-of-thought display.
 *
 * WHY collapsible: reasoning output from models like GPT-5 or Claude
 * extended-thinking can be long and is usually secondary to the final
 * answer. Keep it out of the way by default and let the user opt in.
 *
 * Behaviour:
 *   - While streaming  → collapsed, animated spinner, no preview (avoids
 *     layout jitter as tokens arrive, per LangChain frontend docs).
 *   - After streaming  → collapsed, shows char count + short preview.
 *   - User toggles     → expands into a monospaced <pre> with full content.
 */
interface ThinkingBubbleProps {
	content: string;
	isStreaming: boolean;
}

const REASONING_PREVIEW_LENGTH = 120;

export function ThinkingBubble({ content, isStreaming }: ThinkingBubbleProps) {
	const [expanded, setExpanded] = useState(false);

	const preview =
		content.length > REASONING_PREVIEW_LENGTH
			? content.slice(0, REASONING_PREVIEW_LENGTH).trimEnd() + '…'
			: content;

	return (
		<div
			className="flex justify-start items-end gap-2"
			data-testid={CHAT_TURN_TEST_ID}
		>
			<Avatar variant="ai" />
			<div className="max-w-[80%] rounded-xl rounded-bl-sm bg-surface-secondary border border-border px-4 py-2.5 text-sm">
				<button
					type="button"
					onClick={() => setExpanded((v) => !v)}
					aria-expanded={expanded}
					className="flex items-center gap-1.5 w-full text-left text-xs font-medium text-primary hover:text-primary-dark transition-colors"
				>
					<SparklesIcon
						className={`w-3.5 h-3.5 ${isStreaming ? 'animate-pulse' : ''}`}
					/>
					<span>
						{isStreaming
							? 'Thinking…'
							: `Thought process (${content.length} chars)`}
					</span>
					<span className="ml-auto">
						<ChevronIcon open={expanded} />
					</span>
				</button>

				{expanded && (
					<pre className="whitespace-pre-wrap text-xs font-mono text-text-secondary leading-relaxed m-0 mt-2 pt-2 border-t border-border">
						{content}
					</pre>
				)}

				{!expanded && !isStreaming && content.length > 0 && (
					<div className="mt-1 text-xs italic text-text-tertiary truncate">
						{preview}
					</div>
				)}
			</div>
		</div>
	);
}

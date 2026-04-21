import { useEffect, useRef, type ReactNode } from 'react';

interface ChatContainerProps {
	children: ReactNode;
	input: ReactNode;
	className?: string;
}

/**
 * Scrolling message column with a sticky input row at the bottom.
 *
 * WHY a shared primitive: every pattern (markdown, tool-calling, HITL…)
 * uses the same outer shell. Only the message list and input vary.
 * Auto-scroll is a naive "always jump to bottom" — sufficient for a
 * streaming chat where new tokens always append.
 */
export function ChatContainer({
	children,
	input,
	className = '',
}: ChatContainerProps) {
	const scrollRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		const el = scrollRef.current;
		if (el) el.scrollTop = el.scrollHeight;
	});

	return (
		<div className={`flex flex-col h-full bg-surface text-text ${className}`}>
			<div
				ref={scrollRef}
				data-testid="sdk-preview-messages"
				className="flex-1 overflow-y-auto px-4 py-4 space-y-4"
			>
				{children}
			</div>
			{input && <div className="border-t border-border px-4 py-3">{input}</div>}
		</div>
	);
}

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ComponentPropsWithoutRef } from 'react';

interface MarkdownProps {
	children: string;
	className?: string;
}

/**
 * Markdown renderer for AI message bubbles.
 *
 * WHY: AI messages routinely emit GFM (tables, fenced code, task lists).
 * ``react-markdown`` + ``remark-gfm`` covers that scope without pulling in
 * a full syntax-highlighter (deferred — see Phase 4+ in the plan).
 *
 * Code-block styling is handled via the `.markdown-content` CSS in
 * base.css so Vue/Svelte/HTML markdown renderers (future patterns) pick
 * up the same visuals.
 */
export function Markdown({ children, className }: MarkdownProps) {
	return (
		<div className={`markdown-content ${className ?? ''}`}>
			<ReactMarkdown
				remarkPlugins={[remarkGfm]}
				components={{
					pre: (props: ComponentPropsWithoutRef<'pre'>) => <pre {...props} />,
					code: ({
						children,
						className: codeClass,
						...rest
					}: ComponentPropsWithoutRef<'code'>) => {
						const isInline = !codeClass;
						return isInline ? (
							<code {...rest}>{children}</code>
						) : (
							<code className={codeClass} {...rest}>
								{children}
							</code>
						);
					},
				}}
			>
				{children}
			</ReactMarkdown>
		</div>
	);
}

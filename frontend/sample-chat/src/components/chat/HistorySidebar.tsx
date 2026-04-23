import { useEffect, useRef, useState, type KeyboardEvent } from 'react';

import type { ThreadHistoryEntry } from '../../lib/threadHistory';
import { formatRelative } from '../../lib/relativeTime';

interface HistorySidebarProps {
	entries: ThreadHistoryEntry[];
	activeThreadId: string | null;
	onSelect: (threadId: string) => void;
	onNewThread: () => void;
	onRename: (threadId: string, newTitle: string) => void;
	onDelete: (threadId: string) => void;
	className?: string;
}

/**
 * Left-column list of prior conversations for the current agent instance.
 *
 * WHY: Conversation state lives on the Agent Server (checkpointer) but
 * has no browsable index from the client. This sidebar keeps a local
 * catalog in sync with the Agent Server by registering entries right
 * after a new thread id is issued by `useStream.onThreadId`.
 */
export function HistorySidebar({
	entries,
	activeThreadId,
	onSelect,
	onNewThread,
	onRename,
	onDelete,
	className = '',
}: HistorySidebarProps) {
	return (
		<nav
			aria-label="Conversation history"
			className={`flex h-full flex-col border-r border-border bg-surface-secondary ${className}`}
		>
			<div className="border-b border-border p-3">
				<button
					type="button"
					onClick={onNewThread}
					className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-left text-sm font-medium text-text hover:border-primary hover:text-primary transition-colors cursor-pointer"
				>
					+ New conversation
				</button>
			</div>

			<div className="flex-1 overflow-y-auto p-2">
				{entries.length === 0 ? (
					<p className="px-2 py-4 text-xs text-text-tertiary">
						No past conversations yet. Send a message to start one.
					</p>
				) : (
					<ul className="space-y-1">
						{entries.map((entry) => (
							<li key={entry.threadId}>
								<HistoryRow
									entry={entry}
									isActive={entry.threadId === activeThreadId}
									onSelect={() => onSelect(entry.threadId)}
									onRename={(newTitle) => onRename(entry.threadId, newTitle)}
									onDelete={() => onDelete(entry.threadId)}
								/>
							</li>
						))}
					</ul>
				)}
			</div>
		</nav>
	);
}

// --------------------------------------------------------------------------
// Single-row component — isolated so edit-mode state doesn't force the
// whole list to re-render on every keystroke.
// --------------------------------------------------------------------------

interface HistoryRowProps {
	entry: ThreadHistoryEntry;
	isActive: boolean;
	onSelect: () => void;
	onRename: (newTitle: string) => void;
	onDelete: () => void;
}

function HistoryRow({
	entry,
	isActive,
	onSelect,
	onRename,
	onDelete,
}: HistoryRowProps) {
	const [isEditing, setIsEditing] = useState(false);
	const [draft, setDraft] = useState(entry.title);
	const inputRef = useRef<HTMLInputElement>(null);

	useEffect(() => {
		if (isEditing && inputRef.current) {
			inputRef.current.select();
		}
	}, [isEditing]);

	// Reset draft if the entry's title changes externally (e.g. second tab).
	useEffect(() => {
		if (!isEditing) setDraft(entry.title);
	}, [entry.title, isEditing]);

	const commitRename = () => {
		const next = draft.trim();
		if (next && next !== entry.title) onRename(next);
		else setDraft(entry.title);
		setIsEditing(false);
	};

	const cancelRename = () => {
		setDraft(entry.title);
		setIsEditing(false);
	};

	const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
		if (e.key === 'Enter') {
			e.preventDefault();
			commitRename();
		} else if (e.key === 'Escape') {
			e.preventDefault();
			cancelRename();
		}
	};

	const handleDelete = () => {
		const ok = window.confirm(
			`Remove "${entry.title}" from your local history?\n\n` +
				'The conversation itself stays on the server, but this browser ' +
				'will no longer list it.'
		);
		if (ok) onDelete();
	};

	const baseClasses =
		'group flex w-full items-center gap-1 rounded-lg px-2 py-1.5 text-left text-sm transition-colors';
	const activeClasses = isActive
		? 'bg-surface border border-primary/60 text-text'
		: 'border border-transparent text-text-secondary hover:bg-surface hover:text-text';

	return (
		<div className={`${baseClasses} ${activeClasses}`}>
			{isEditing ? (
				<input
					ref={inputRef}
					type="text"
					value={draft}
					onChange={(e) => setDraft(e.target.value)}
					onBlur={commitRename}
					onKeyDown={handleKeyDown}
					aria-label="Rename conversation"
					className="flex-1 min-w-0 rounded border border-border bg-surface px-2 py-1 text-sm text-text focus:outline-none focus:ring-2 focus:ring-primary/30"
				/>
			) : (
				<button
					type="button"
					onClick={onSelect}
					onDoubleClick={() => setIsEditing(true)}
					className="flex-1 min-w-0 cursor-pointer text-left"
					title="Click to open · double-click to rename"
				>
					<div className="truncate font-medium">{entry.title}</div>
					<div className="truncate text-xs text-text-tertiary">
						{formatRelative(entry.updatedAt)}
					</div>
				</button>
			)}

			{!isEditing && (
				<div className="flex shrink-0 items-center gap-0.5 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
					<IconButton
						label="Rename"
						onClick={() => setIsEditing(true)}
						icon={<PencilIcon />}
					/>
					<IconButton
						label="Remove from history"
						onClick={handleDelete}
						icon={<TrashIcon />}
					/>
				</div>
			)}
		</div>
	);
}

// --------------------------------------------------------------------------
// Small private icons / button — kept local to avoid bloating icons.tsx
// with sidebar-only glyphs.
// --------------------------------------------------------------------------

function IconButton({
	label,
	icon,
	onClick,
}: {
	label: string;
	icon: React.ReactNode;
	onClick: () => void;
}) {
	return (
		<button
			type="button"
			onClick={onClick}
			aria-label={label}
			title={label}
			className="rounded p-1 text-text-tertiary hover:bg-surface-secondary hover:text-primary focus:outline-none focus:ring-1 focus:ring-primary cursor-pointer"
		>
			{icon}
		</button>
	);
}

function PencilIcon() {
	return (
		<svg
			width="14"
			height="14"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
			aria-hidden="true"
		>
			<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
			<path d="m18.5 2.5 3 3L12 15l-4 1 1-4 9.5-9.5z" />
		</svg>
	);
}

function TrashIcon() {
	return (
		<svg
			width="14"
			height="14"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
			aria-hidden="true"
		>
			<polyline points="3 6 5 6 21 6" />
			<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
			<path d="M10 11v6" />
			<path d="M14 11v6" />
			<path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2" />
		</svg>
	);
}

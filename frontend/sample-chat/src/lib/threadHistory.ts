/**
 * Per-instance conversation history persisted in localStorage.
 *
 * WHY: The Agent Server's checkpointer already stores every thread's
 * state. The frontend only needs to remember a small catalog of
 * thread ids (plus a human-readable title and timestamps) so the user
 * can browse and resume prior conversations. Keeping this entirely
 * client-side avoids touching the backend schema for what is a
 * demo-level UX.
 *
 * Scoping is per agent instance (key = `chat.threads.{instanceId}`),
 * not per user. JWT contents are not decoded on the frontend, so we
 * accept the limitation that users sharing a browser profile share
 * this history. Sufficient for the sample-chat use case.
 */

export interface ThreadHistoryEntry {
	threadId: string;
	title: string;
	createdAt: number; // epoch ms
	updatedAt: number; // epoch ms
}

/** Soft cap to keep localStorage footprint bounded. */
const MAX_ENTRIES = 50;
const KEY_PREFIX = 'chat.threads.';
const TITLE_MAX_CHARS = 60;

function storageKey(instanceId: string): string {
	return `${KEY_PREFIX}${instanceId}`;
}

export function loadHistory(instanceId: string): ThreadHistoryEntry[] {
	if (typeof localStorage === 'undefined') return [];
	try {
		const raw = localStorage.getItem(storageKey(instanceId));
		if (!raw) return [];
		const parsed = JSON.parse(raw);
		if (!Array.isArray(parsed)) return [];
		return parsed.filter(isValidEntry);
	} catch {
		return [];
	}
}

export function saveHistory(
	instanceId: string,
	entries: ThreadHistoryEntry[]
): void {
	if (typeof localStorage === 'undefined') return;
	const trimmed = [...entries]
		.sort((a, b) => b.updatedAt - a.updatedAt)
		.slice(0, MAX_ENTRIES);
	try {
		localStorage.setItem(storageKey(instanceId), JSON.stringify(trimmed));
	} catch {
		/* storage disabled or quota exceeded — in-memory only */
	}
}

export function upsertEntry(
	instanceId: string,
	entry: ThreadHistoryEntry
): ThreadHistoryEntry[] {
	const current = loadHistory(instanceId);
	const idx = current.findIndex((e) => e.threadId === entry.threadId);
	const next =
		idx >= 0
			? current.map((e, i) => (i === idx ? { ...e, ...entry } : e))
			: [entry, ...current];
	saveHistory(instanceId, next);
	return next;
}

export function touchEntry(
	instanceId: string,
	threadId: string,
	now: number = Date.now()
): ThreadHistoryEntry[] {
	const current = loadHistory(instanceId);
	const idx = current.findIndex((e) => e.threadId === threadId);
	if (idx < 0) return current;
	const next = current.map((e, i) =>
		i === idx ? { ...e, updatedAt: now } : e
	);
	saveHistory(instanceId, next);
	return next;
}

export function removeEntry(
	instanceId: string,
	threadId: string
): ThreadHistoryEntry[] {
	const next = loadHistory(instanceId).filter((e) => e.threadId !== threadId);
	saveHistory(instanceId, next);
	return next;
}

export function renameEntry(
	instanceId: string,
	threadId: string,
	newTitle: string
): ThreadHistoryEntry[] {
	const title = newTitle.trim().slice(0, TITLE_MAX_CHARS) || 'Untitled';
	const current = loadHistory(instanceId);
	const next = current.map((e) =>
		e.threadId === threadId ? { ...e, title, updatedAt: Date.now() } : e
	);
	saveHistory(instanceId, next);
	return next;
}

/**
 * Derive a human-readable title from the first user message.
 *
 * Falls back to "New conversation" when the message is empty (should
 * not normally happen — we only upsert after a real submit).
 */
export function deriveTitle(firstMessage: string): string {
	const cleaned = firstMessage.replace(/\s+/g, ' ').trim();
	if (!cleaned) return 'New conversation';
	if (cleaned.length <= TITLE_MAX_CHARS) return cleaned;
	return `${cleaned.slice(0, TITLE_MAX_CHARS - 1).trimEnd()}…`;
}

export function getStorageKey(instanceId: string): string {
	return storageKey(instanceId);
}

function isValidEntry(value: unknown): value is ThreadHistoryEntry {
	if (!value || typeof value !== 'object') return false;
	const v = value as Record<string, unknown>;
	return (
		typeof v.threadId === 'string' &&
		typeof v.title === 'string' &&
		typeof v.createdAt === 'number' &&
		typeof v.updatedAt === 'number'
	);
}

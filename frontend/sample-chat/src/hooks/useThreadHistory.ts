/**
 * React hook around the localStorage-backed thread history catalog.
 *
 * WHY: Components need a reactive view of history entries that updates
 * when the user starts a new thread, renames one, or when another
 * browser tab modifies the same storage key. `useSyncExternalStore`
 * gives us tab-to-tab consistency via the native `storage` event
 * without pulling in a state library.
 */

import { useCallback, useSyncExternalStore } from 'react';

import {
	deriveTitle,
	getStorageKey,
	loadHistory,
	removeEntry,
	renameEntry,
	touchEntry,
	upsertEntry,
	type ThreadHistoryEntry,
} from '../lib/threadHistory';

type Listener = () => void;

// Local pub/sub so same-tab mutations notify subscribers immediately
// (the `storage` event only fires for *other* tabs).
const listeners = new Set<Listener>();

function notify(): void {
	listeners.forEach((l) => l());
}

function subscribe(listener: Listener): () => void {
	listeners.add(listener);
	const onStorage = (e: StorageEvent) => {
		// Any thread-history key counts; individual hook instances filter
		// by instanceId via their getSnapshot.
		if (e.key === null || e.key.startsWith('chat.threads.')) listener();
	};
	window.addEventListener('storage', onStorage);
	return () => {
		listeners.delete(listener);
		window.removeEventListener('storage', onStorage);
	};
}

// Cache snapshots per instance so useSyncExternalStore's referential
// equality check doesn't trigger infinite renders. The snapshot is
// invalidated whenever we write to storage via the mutating helpers.
const snapshotCache = new Map<string, ThreadHistoryEntry[]>();

function getSnapshot(instanceId: string): ThreadHistoryEntry[] {
	const cached = snapshotCache.get(instanceId);
	if (cached) return cached;
	const fresh = loadHistory(instanceId).sort(
		(a, b) => b.updatedAt - a.updatedAt
	);
	snapshotCache.set(instanceId, fresh);
	return fresh;
}

function invalidate(instanceId: string): void {
	snapshotCache.delete(instanceId);
	notify();
}

export interface UseThreadHistoryResult {
	entries: ThreadHistoryEntry[];
	/** Create an entry for a freshly-issued thread id (no-op if already present). */
	registerNew: (threadId: string, firstMessage: string) => void;
	/** Bump updatedAt on an existing entry (ignored if unknown). */
	touch: (threadId: string) => void;
	/** Rename an entry to a user-provided title. */
	rename: (threadId: string, newTitle: string) => void;
	/** Remove an entry from history (Agent Server checkpoint is untouched). */
	remove: (threadId: string) => void;
}

export function useThreadHistory(
	instanceId: string | null
): UseThreadHistoryResult {
	// When no instance is selected we return a stable empty-result shape.
	const entries = useSyncExternalStore(
		subscribe,
		() => (instanceId ? getSnapshot(instanceId) : EMPTY),
		() => EMPTY
	);

	const registerNew = useCallback(
		(threadId: string, firstMessage: string) => {
			if (!instanceId || !threadId) return;
			const existing = loadHistory(instanceId).find(
				(e) => e.threadId === threadId
			);
			if (existing) return;
			const now = Date.now();
			upsertEntry(instanceId, {
				threadId,
				title: deriveTitle(firstMessage),
				createdAt: now,
				updatedAt: now,
			});
			invalidate(instanceId);
		},
		[instanceId]
	);

	const touch = useCallback(
		(threadId: string) => {
			if (!instanceId || !threadId) return;
			touchEntry(instanceId, threadId);
			invalidate(instanceId);
		},
		[instanceId]
	);

	const rename = useCallback(
		(threadId: string, newTitle: string) => {
			if (!instanceId || !threadId) return;
			renameEntry(instanceId, threadId, newTitle);
			invalidate(instanceId);
		},
		[instanceId]
	);

	const remove = useCallback(
		(threadId: string) => {
			if (!instanceId || !threadId) return;
			removeEntry(instanceId, threadId);
			invalidate(instanceId);
		},
		[instanceId]
	);

	return { entries, registerNew, touch, rename, remove };
}

// Module-level singleton so getServerSnapshot / empty-instance paths
// keep returning the same array reference across renders.
const EMPTY: ThreadHistoryEntry[] = [];

// Keep the unused import lint-safe in case someone trims helpers above.
export { getStorageKey };

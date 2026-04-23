/**
 * Format an epoch-ms timestamp as a short relative label ("5m ago",
 * "yesterday", "Apr 22").
 *
 * WHY a local helper: avoid adding a date library for a single string
 * used in the history sidebar. Deliberately coarse — precision is
 * not useful when scanning a list of conversations.
 */

const SECOND = 1000;
const MINUTE = 60 * SECOND;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

export function formatRelative(ms: number, now: number = Date.now()): string {
	const diff = now - ms;
	if (!Number.isFinite(diff) || diff < 0) return 'just now';
	if (diff < MINUTE) return 'just now';
	if (diff < HOUR) {
		const m = Math.floor(diff / MINUTE);
		return `${m}m ago`;
	}
	if (diff < DAY) {
		const h = Math.floor(diff / HOUR);
		return `${h}h ago`;
	}
	if (diff < 2 * DAY) return 'yesterday';
	if (diff < 7 * DAY) {
		const d = Math.floor(diff / DAY);
		return `${d}d ago`;
	}
	// Older than a week: show a short calendar date.
	const date = new Date(ms);
	return date.toLocaleDateString(undefined, {
		month: 'short',
		day: 'numeric',
	});
}

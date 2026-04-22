/**
 * Shared types and small helpers for tool-calling cards.
 *
 * WHY: Keeps card files free of duplicate type imports. The generic args
 * type lives here because we deliberately avoid the ``ToolCallFromTool``
 * helper from ``@langchain/react`` — our tools live in Python, so there is
 * no TypeScript schema to feed the helper. Callers provide the expected
 * args shape via a plain generic parameter on each card.
 */

import type { ToolMessage } from 'langchain';
import type { ToolCallWithResult as BaseToolCallWithResult } from '@langchain/react';

export type ToolCallState = 'pending' | 'completed' | 'error';

export type ToolCall = BaseToolCallWithResult;

/**
 * Safely parse a possibly-JSON string. If parsing fails the original
 * value is returned unchanged; callers then fall back to a string view.
 */
export function tryParseJSON(value: unknown): unknown {
	if (typeof value === 'string') {
		try {
			return JSON.parse(value);
		} catch {
			return value;
		}
	}
	return value;
}

export type { ToolMessage };

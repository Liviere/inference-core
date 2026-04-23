import { parse } from 'yaml';

import presetPromptsYaml from '../config/preset-prompts.yaml?raw';

interface AgentPromptEntry {
	prompts?: unknown;
}

interface PresetPromptConfig {
	agents?: Record<string, AgentPromptEntry>;
}

/**
 * Normalizes arbitrary YAML input into a safe string array.
 *
 * WHY: configuration is user-editable, so the UI should degrade to an empty
 * state instead of throwing when a prompt entry is malformed.
 */
function normalizePrompts(value: unknown): string[] {
	if (!Array.isArray(value)) return [];
	return value.filter((item): item is string => typeof item === 'string');
}

/**
 * Parses the YAML preset configuration once at module load.
 *
 * WHY: presets are static frontend configuration. Parsing eagerly keeps the
 * render path simple and guarantees missing or broken config falls back to an
 * empty list for every agent.
 */
function loadPresetPromptConfig(): Map<string, string[]> {
	try {
		const parsed = parse(presetPromptsYaml) as PresetPromptConfig | null;
		const entries = Object.entries(parsed?.agents ?? {});
		return new Map(
			entries.map(([agentName, entry]) => [
				agentName,
				normalizePrompts(entry?.prompts),
			])
		);
	} catch (error) {
		console.warn('Failed to parse preset prompt config', error);
		return new Map();
	}
}

const presetPromptsByAgent = loadPresetPromptConfig();

/**
 * Returns preset prompts for a given agent type.
 *
 * WHY: the empty-thread view should be configurable per base agent name while
 * keeping the fallback behavior explicit: no config means no suggestions.
 */
export function getPresetPrompts(agentName?: string | null): string[] {
	if (!agentName) return [];
	return presetPromptsByAgent.get(agentName) ?? [];
}
/**
 * Thin typed wrapper around the inference-core REST API.
 *
 * WHY: The frontend touches three endpoint families during the chat lifecycle:
 *   1. /auth/login          — exchange username+password for a JWT
 *   2. /agent-instances     — list user instances for the picker
 *   3. /run-bundle          — get the handshake payload for useStream
 *
 * All calls go through Vite's /api proxy in dev so we share an origin with
 * the backend and avoid CORS for the inference-core API surface.  The Agent
 * Server is reached separately by the langgraph-sdk client.
 */

import { clearToken, getToken } from './auth';

export interface LoginResponse {
	access_token: string;
	token_type: string;
}

export interface AgentInstance {
	id: string;
	instance_name: string;
	display_name: string;
	base_agent_name: string;
	description?: string | null;
	primary_model?: string | null;
	is_default: boolean;
	is_deepagent: boolean;
	is_active: boolean;
}

export interface AgentInstanceListResponse {
	instances: AgentInstance[];
	total: number;
}

export interface RunBundleResponse {
	instance_id: string;
	instance_name: string;
	display_name: string;
	base_agent_name: string;
	description?: string | null;
	assistant_id: string;
	agent_server_url: string;
	access_token?: string | null;
	config: { configurable: Record<string, unknown> };
	is_remote: boolean;
}

export type AccessMode = 'public' | 'user' | 'superuser';

export interface AccessModeResponse {
	mode: AccessMode;
}

class ApiError extends Error {
	status: number;
	body?: unknown;
	constructor(message: string, status: number, body?: unknown) {
		super(message);
		this.status = status;
		this.body = body;
	}
}

async function request<T>(
	path: string,
	init: RequestInit = {},
	{ auth = true }: { auth?: boolean } = {}
): Promise<T> {
	const headers = new Headers(init.headers);
	if (!headers.has('Content-Type') && init.body) {
		headers.set('Content-Type', 'application/json');
	}
	if (auth) {
		const token = getToken();
		if (token) headers.set('Authorization', `Bearer ${token}`);
	}

	const resp = await fetch(`/api/v1${path}`, { ...init, headers });
	if (resp.status === 401) {
		// Token expired or rejected — drop it so the UI returns to login.
		clearToken();
	}
	if (!resp.ok) {
		let body: unknown;
		try {
			body = await resp.json();
		} catch {
			body = await resp.text();
		}
		throw new ApiError(
			`Request to ${path} failed (${resp.status})`,
			resp.status,
			body
		);
	}
	if (resp.status === 204) return undefined as T;
	return (await resp.json()) as T;
}

export async function login(
	username: string,
	password: string
): Promise<LoginResponse> {
	return request<LoginResponse>(
		'/auth/login',
		{ method: 'POST', body: JSON.stringify({ username, password }) },
		{ auth: false }
	);
}

export async function getAccessMode(): Promise<AccessModeResponse> {
	// Unauthenticated on purpose: we call this before the user has any
	// token so the UI can decide whether a login screen is needed at all.
	return request<AccessModeResponse>('/auth/access-mode', {}, { auth: false });
}

export async function listAgentInstances(): Promise<AgentInstanceListResponse> {
	return request<AgentInstanceListResponse>('/agent-instances');
}

export async function getRunBundle(
	instanceId: string,
	sessionId?: string
): Promise<RunBundleResponse> {
	const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
	return request<RunBundleResponse>(
		`/agent-instances/${instanceId}/run-bundle${qs}`
	);
}

export { ApiError };

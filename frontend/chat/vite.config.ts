import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

/**
 * Dev server proxies:
 *   - /api          → FastAPI backend (JWT, run-bundle, agent-instances)
 *   - /api/langgraph → LangGraph Agent Server (direct — no FastAPI hop)
 *
 * The /api/langgraph proxy is optional: when VITE_USE_AGENT_PROXY=true
 * the frontend rewrites bundle.agent_server_url to same-origin, avoiding
 * browser CORS in dev without changing the backend contract.
 */
export default defineConfig(({ mode }) => {
	const env = loadEnv(mode, process.cwd(), '');
	const agentTarget = env.VITE_AGENT_SERVER_URL || 'http://127.0.0.1:2024';
	const backendTarget = env.VITE_BACKEND_URL || 'http://localhost:8000';
	return {
		plugins: [react(), tailwindcss()],
		server: {
			port: 5173,
			proxy: {
				// Order matters: langgraph first so /api/langgraph doesn't hit /api.
				'/api/langgraph': {
					target: agentTarget,
					changeOrigin: true,
					rewrite: (p) => p.replace(/^\/api\/langgraph/, ''),
				},
				'/api': {
					target: backendTarget,
					changeOrigin: true,
				},
			},
		},
	};
});

import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

/**
 * Dev server proxies /api → FastAPI backend (default 8000).
 *
 * The Agent Server (port 2024) is reached DIRECTLY by useStream — bypasses
 * the proxy on purpose so we exercise the same CORS + JWT path as production.
 */
export default defineConfig(({ mode }) => {
	const env = loadEnv(mode, process.cwd(), '');
	return {
		plugins: [react(), tailwindcss()],
		server: {
			port: 5173,
			proxy: {
				'/api': {
					target: env.VITE_BACKEND_URL || 'http://localhost:8000',
					changeOrigin: true,
				},
			},
		},
	};
});

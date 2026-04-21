/// <reference types="vite/client" />

interface ImportMetaEnv {
	readonly VITE_BACKEND_URL?: string;
	readonly VITE_AGENT_SERVER_URL?: string;
	readonly VITE_USE_AGENT_PROXY?: string;
}

interface ImportMeta {
	readonly env: ImportMetaEnv;
}

# Inference Core — Chat (Frontend MVP)

React + Vite chat UI that talks to the **LangGraph Agent Server** via
[`useStream`](https://github.com/langchain-ai/langgraph) while using the
inference-core FastAPI backend for authentication and per-user agent
configuration.

This iteration adds a fuller MVP shell around the original handshake flow:

- light/dark theme toggle with persisted preference
- markdown + GFM rendering for AI and user bubbles
- preset starter prompts on empty threads
- optional same-origin Vite proxy for Agent Server calls in local development
- richer chat shell components instead of a single monolithic view

## Stack

- Vite + React 19 + TypeScript
- Tailwind v4 (via `@tailwindcss/vite`)
- `@langchain/react` (`useStream`)
- `react-markdown` + `remark-gfm`

## Architecture

```
Browser                                 Backend
─────────────                           ────────────────────────────
LoginForm  ───POST /api/v1/auth/login──▶  FastAPI  (returns JWT)
                │
                ▼ store JWT in localStorage
InstanceSelector ─GET /api/v1/agent-instances─▶ FastAPI
                │
                ▼ user picks an instance
ChatView   ───GET /agent-instances/{id}/run-bundle──▶ FastAPI
                │      ↳ {assistant_id, agent_server_url,
                │         access_token, config.configurable, …}
                ▼
useStream  ───POST /threads /runs (SSE) ──▶ LangGraph Agent Server
                │                           (validates JWT via
                └─ optional /api/langgraph  langgraph_auth.py)
                   same-origin Vite proxy
```

By default this frontend uses the Vite dev proxy for both `/api` and
`/api/langgraph`, so the browser can stay same-origin during local
development. If you disable that proxy, the browser reaches the Agent Server
directly and must be allowed by `langgraph.json → http.cors`.

## Setup

```bash
cd frontend/chat
cp .env.example .env       # adjust backend / Agent Server URLs if needed
npm install                # or pnpm install / yarn
npm run dev                # http://localhost:5173
```

Relevant env vars:

- `VITE_BACKEND_URL` — FastAPI base URL for the `/api` dev proxy.
- `VITE_AGENT_SERVER_URL` — LangGraph Agent Server URL behind `/api/langgraph`.
- `VITE_USE_AGENT_PROXY=true` — rewrites `bundle.agent_server_url` to
   `${window.location.origin}/api/langgraph`, avoiding browser CORS in dev.

In separate terminals:

```bash
# 1. FastAPI (issues JWTs, serves /agent-instances + /run-bundle)
poetry run uvicorn inference_core.main_factory:create_application \
  --factory --reload --port 8000

# 2. LangGraph Agent Server (runs the actual graphs)
poetry run langgraph dev --no-browser
```

## Auth flow

1. User signs in — backend returns `{access_token}`.
2. Token is stored in `localStorage` and attached as `Authorization: Bearer …`
   to every `/api/*` request.
3. When the user opens a chat, the backend's `/run-bundle` endpoint echoes
   that same token in `bundle.access_token`.
4. The frontend hands it to `useStream` via `defaultHeaders`, so the
   Agent Server sees the same JWT and validates it through
   `langgraph_auth.py`.
5. When `VITE_USE_AGENT_PROXY=true`, the browser calls the Vite dev server at
   `/api/langgraph` and Vite forwards those requests to the Agent Server.

## UI Surface

- `LoginForm` and `InstanceSelector` now share the same theme-aware visual
  tokens and expose a theme toggle.
- `ChatView` renders a structured chat shell with reusable bubble, input,
  typing-indicator, and prompt-preset components.
- Message bubbles render markdown using GFM, so tables, fenced code blocks,
  lists, and inline code display correctly.
- The input exposes a "new thread" action once a conversation exists.

## What this MVP does NOT cover

- Refresh-token rotation (token simply expires and the user re-logs in)
- Tool-call cards and reasoning bubbles
- HITL / `interrupt()` resumes
- Time travel / thread history sidebar
- Streaming of subagent / supervisor branches with their own UI

These are deliberate Phase 4+ deferrals — the scope here is "prove the
useStream → Agent Server path works end-to-end with our auth and
configurable overrides."

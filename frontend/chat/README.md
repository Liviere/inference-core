# Inference Core — Chat (Frontend MVP)

Minimal React + Vite chat UI that talks to the **LangGraph Agent Server**
directly via [`useStream`](https://github.com/langchain-ai/langgraph) while
using the inference-core FastAPI backend for authentication and per-user
agent configuration.

## Stack

- Vite + React 18 + TypeScript
- Tailwind v4 (via `@tailwindcss/vite`)
- `@langchain/langgraph-sdk/react` (`useStream`)

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
                                            (validates JWT via
                                             langgraph_auth.py)
```

The frontend never goes through FastAPI for streaming — it reaches the
Agent Server directly. CORS is allowed in `langgraph.json → http.cors`
for `http://localhost:5173`.

## Setup

```bash
cd frontend/chat
cp .env.example .env       # adjust VITE_BACKEND_URL if needed
npm install                # or pnpm install / yarn
npm run dev                # http://localhost:5173
```

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

## What this MVP does NOT cover

- Refresh-token rotation (token simply expires and the user re-logs in)
- Tool-call cards, reasoning bubbles, markdown rendering
- HITL / `interrupt()` resumes
- Time travel / thread history sidebar
- Streaming of subagent / supervisor branches with their own UI

These are deliberate Phase 4+ deferrals — the scope here is "prove the
useStream → Agent Server path works end-to-end with our auth and
configurable overrides."

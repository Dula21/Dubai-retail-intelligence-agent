# DRI Agent frontend (Next.js)

A Next.js 14 (App Router, TypeScript) interface for the reorder agent: scan, review the reasoning trace, approve / modify /
reject. No UI library; plain CSS with light and dark themes.

## Run it

With the rest of the stack (Docker):

```powershell
docker compose -f docker-compose.yml -f docker-compose.frontend.yml up -d --build
# open http://localhost:3000
```

Locally, with the API already on :8000:

```powershell
cd frontend
npm install
npm run dev
```

`BACKEND_URL` (default `http://localhost:8000`) tells the proxy where the API is. In Docker it is `http://api:8000`.

## Pages

- `/` recommendations: run or replay a scan, filter by status, open one to see the numbers, caveats, full reasoning trace and
  history, and record a decision. A decision is final; the agent's own quantity is kept beside the human's.
- `/ask` the chat agent (English or Arabic), with the engine's order quantities shown separately from the model's prose.

## Decision: same-origin proxy

The browser calls `/api/agent/*`, a Next.js route handler that forwards to the FastAPI backend server to server.
Alternative: call the API directly from the browser (needs CORS on the API). Why this way: no backend change, the backend URL
never reaches the browser, and a scan that fits dozens of Prophet models is not cut short by a dev-proxy timeout. Trade-off:
one extra hop. Only an allow-list of paths is forwarded (`recommendations`, `scan`, `query`, `health`).

## Scale and deployment notes

- The UI loads the whole recommendation list (up to 500 rows) and filters in the browser. Fine at SME scale; at larger scale
  it needs server-side pagination and filtering.
- No authentication: the "Your name" field on a decision is self-declared, so it is an audit label, not an identity. Before
  any real use, put the app behind login and take the actor from the session.
- Hosting on a free tier: this app is a standard Node server (Docker image provided). A Hugging Face Space for the API plus a
  static or Node host for the UI both fit the free-tier rule; set `BACKEND_URL` to the API's public address.

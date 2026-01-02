# rust-local-rag frontend

Minimal React (Vite) UI for the local RAG server.

## Quick start

1) Start the backend (from repo root):

```bash
make run
```

2) Start the frontend (in another terminal):

```bash
cd frontend
npm install
npm run dev
```

The dev server proxies `POST /search` to `http://127.0.0.1:3046/search`.


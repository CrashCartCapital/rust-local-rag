# AGENTS.md — rust-local-rag

Guidance for AI coding agents (Codex CLI, Claude Code, etc.) working in this repository.

Synthesized from:
- `CLAUDE.md`
- `/Users/ryanpappal/My Drive/Obsidian Vaults/CCC_2025-10-24/CODE_VAULT/configs/GENERAL_GUIDANCE.md`
- `/Users/ryanpappal/My Drive/Obsidian Vaults/CCC_2025-10-24/CODE_VAULT/configs/MCP_TOOLS_REF.md`

---

## Instruction Priority (when conflicts occur)

1. System/runtime instructions (highest authority)
2. User request in the current session
3. This `AGENTS.md`
4. `CLAUDE.md`
5. `README.md` / `docs/*` / other repo docs

If your environment provides a sandbox/approval model, obey it. Don’t run destructive commands unless explicitly requested.

---

## Project Summary

`rust-local-rag` is a Rust MCP server that provides **local RAG** (Retrieval-Augmented Generation) over a directory of PDFs:

- Extracts PDF text locally (primary: `lopdf`; fallback: `pdftotext` when available)
- Chunks text (sentence-aware), embeds chunks via **local Ollama**, persists embeddings locally (per-model indexes)
- Serves MCP tools + job orchestration for background indexing to avoid timeouts
- Also exposes HTTP endpoints used by the TUI and the evaluation harness

---

## Repo Layout (key paths)

- `src/main.rs`: entrypoint; env + logging; initializes engine + job system; starts server
- `src/mcp_server.rs`: MCP tool definitions + Streamable HTTP server + health/eval endpoints
- `src/rag_engine.rs`: chunking, persistence, retrieval, MMR diversification, scoring + reranking orchestration
- `src/embeddings.rs`: Ollama embeddings client; batch embedding; query LRU cache
- `src/reranker.rs`: optional Ollama-based reranker; prompt override via `PROMPTS_DIR/prompts/reranker.txt`
- `src/job_manager.rs`: SQLite job persistence; atomic “single active job” semantics
- `src/worker.rs`: background worker supervisor; resumable jobs; per-document locking
- `src/bin/rag_tui/*`: TUI client for the HTTP endpoints
- `docs/*`: setup/usage/eval specs and debugging notes
- `eval/*`: Python evaluation harness (talks to HTTP endpoints)
- `.mcp.json`: local MCP wiring (includes this server + `mcpjungle-ccc`)
- `.cargo/config.toml`: cargo aliases; **warnings are errors** via `-D warnings`

---

## Build, Run, Test (common commands)

Preferred (repo-provided):
- `make run` (dev logging)
- `make ci` (check + lint + test + build)
- `make fmt`, `make clippy`, `make test`

Cargo aliases (from `.cargo/config.toml`):
- `cargo c` (check), `cargo cc` (clippy), `cargo ccd` (clippy -D warnings), `cargo f` (fmt), `cargo t` (test)

TUI (via `justfile`):
- `just tui` (runs `rag-tui` against `http://localhost:3046`)
- `just up` (build server, run in background, then launch TUI)

---

## Runtime Configuration (env vars)

Core:
- `DATA_DIR` (default `./data`)
- `DOCUMENTS_DIR` (default `./documents`)
- `LOG_DIR` (default `/var/log/rust-local-rag` if writable else `./logs`)
- `LOG_LEVEL` (default `info`)
- `LOG_MAX_MB` (default `5`)
- `DEV` / `DEVELOPMENT` (prefer console logs)
- `CONSOLE_LOGS` (force console logs)

Ollama:
- `OLLAMA_URL` (default `http://localhost:11434`)
- `OLLAMA_EMBEDDING_MODEL` (default `nomic-embed-text`)
- `OLLAMA_RERANK_MODEL` (default `llama3.1`; reranker init is non-fatal if unavailable)
- `PROMPTS_DIR` (default `./prompts`) — prompt override at `prompts/reranker.txt`

MCP/HTTP:
- `MCP_HTTP_BIND` (default `127.0.0.1:3046`)
- `MCP_HTTP_ENDPOINT` (default `/mcp`)

Retrieval weights (global defaults; may be overridden per-query):
- `RAG_EMBEDDING_WEIGHT` (default `0.7`)
- `RAG_LEXICAL_WEIGHT` (default `0.3`)
- `RAG_RERANKER_WEIGHT` (default `0.7`)
- `RAG_INITIAL_SCORE_WEIGHT` (default `0.3`)

---

## Engineering Guardrails (important invariants)

Warnings are errors:
- `.cargo/config.toml` sets `-D warnings`. Keep builds/clippy warning-free.

Async + blocking work:
- Do not block Tokio runtime. Use `tokio::task::spawn_blocking` for CPU-bound / blocking work (PDF extraction, embedding calls, heavy parsing).

Locks:
- The shared engine is `Arc<RwLock<RagEngine>>`. Keep write locks short.
- Indexing uses per-document locking; don’t reintroduce “hold write lock for entire reindex” behavior.

Jobs:
- Reindexing runs as background jobs (SQLite persistence + supervisor).
- Preserve “single active reindex job” semantics (atomic check-and-create).
- Poison-pill behavior: a single document failure should not abort the whole job.

Persistence:
- Indexes are model-partitioned (`chunks_{sanitized_model}.json`), plus SHA-256 document hashes.
- If you change persistence format/semantics, add a migration path and update docs.

---

## Testing & Evaluation

Rust:
- `make test` / `cargo test`
- TUI tests: `cargo test --bin rag-tui`

Evaluation harness (`eval/`):
- Requires Python `>=3.11` (see `eval/pyproject.toml`)
- Start server first: `make run`
- Run eval: `python -m eval.run evaluate --config baseline -v`

If you change search/scoring/reranking behavior, update:
- tests where appropriate
- `docs/RAG_EVALUATION_FRAMEWORK_SPEC.md` (if metrics/semantics change)
- `eval/configs/*` (if config surface changes)

---

## AI Workflow: Pragmatism + Minimalism (from GENERAL_GUIDANCE)

Default stance:
- Implement only what’s requested; no scope creep.
- Prefer the simplest single-user, local-first solution.
- Avoid “enterprise” architecture unless explicitly requested.

Use ensemble validation for non-trivial work (recommended for):
- logic changes, multi-file edits, refactors
- async/concurrency changes
- persistence/scoring/reranker changes

Pragmatism Checklist (aim to pass 6/8 before implementing big recommendations):
1. Solves a real user problem now
2. Defends against a real threat model
3. Simplest viable approach first
4. Addresses a current blocker
5. Low maintenance burden
6. macOS-first is acceptable here
7. Can’t be deferred easily
8. No simpler 80/20 alternative exists

---

## AI Ensemble + Tooling (from MCP_TOOLS_REF)

If the CCC MCP toolchain is available:

- Gemini: **ONLY** `model="gemini-3-pro-preview"` (ignore older references in `CLAUDE.md`)
- Qwen: `model="qwen3-coder-plus"` as the backup consultant
- Codex bridge: always set `timeout` (recommend `420–600s` for heavy tasks)

Failure protocol (timeouts):
- Retry once; then retry with a much narrower prompt; then fall back to Qwen for that round only.

Consultant models are analysis-only:
- Don’t ask them to call tools; use them to review/validate plans and changes.

---

## CCC / mcpjungle Operational Requirement (from CLAUDE.md)

🚨 **Never restart `mcpjungle` from any directory other than `~/03_CODE`.**

Correct pattern:
```bash
pkill -f "mcpjungle.*start"
sleep 2
cd ~/03_CODE
nohup mcpjungle start > /tmp/mcpjungle.log 2>&1 &
```

---

## ccc-code-mode Workflow Pattern (if present in your environment)

When using ccc-code-mode tools:
1. Always `search_workflows` before creating new workflows
2. If found: `execute_workflow`
3. If not found: use `execute_code` (one-off) and optionally `save_workflow` (reusable)


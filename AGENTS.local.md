## Project Summary

`rust-local-rag` is a Rust MCP server that provides **local RAG** (Retrieval-Augmented Generation) over a directory of PDFs:

- Extracts PDF text locally (primary: `lopdf`; fallback: `pdftotext` when available)
- Chunks text (sentence-aware), embeds chunks via **local Ollama**, persists embeddings locally (per-model indexes)
- Serves MCP tools + job orchestration for background indexing to avoid timeouts
- Also exposes HTTP endpoints used by the TUI and the evaluation harness

---

## Repo Layout

See **[docs/development.md](docs/development.md)** for the canonical repository layout and build instructions.

---

## Runtime Configuration (env vars)

See **[docs/configuration.md](docs/configuration.md)** for the canonical list of environment variables and configuration options.

---

## Engineering Guardrails (important invariants)

**Warnings are errors**:
- `.cargo/config.toml` sets `-D warnings`. Keep builds/clippy warning-free.

**Async + blocking work**:
- Do not block Tokio runtime. Use `tokio::task::spawn_blocking` for CPU-bound / blocking work (PDF extraction, embedding calls, heavy parsing).

**Locks**:
- The shared engine is `Arc<RwLock<RagEngine>>`. Keep write locks short.
- Indexing uses per-document locking; don't reintroduce "hold write lock for entire reindex" behavior.

**Jobs**:
- Reindexing runs as background jobs (SQLite persistence + supervisor).
- Preserve "single active reindex job" semantics (atomic check-and-create).
- Poison-pill behavior: a single document failure should not abort the whole job.

**Persistence**:
- Indexes are model-partitioned (`chunks_{sanitized_model}.json`), plus SHA-256 document hashes.
- If you change persistence format/semantics, add a migration path and update docs.

---

## Testing & Evaluation

**Rust tests**:
- `make test` / `cargo test`
- TUI tests: `cargo test --bin rag-tui`

**Evaluation harness** (`eval/`):
- Requires Python `>=3.11` (see `eval/pyproject.toml`)
- Start server first: `make run`
- Run eval: `python -m eval.run evaluate --config baseline -v`

If you change search/scoring/reranking behavior, update:
- tests where appropriate
- `docs/RAG_EVALUATION_FRAMEWORK_SPEC.md` (if metrics/semantics change)
- `eval/configs/*` (if config surface changes)

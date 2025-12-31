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

## Repo Layout

See **[docs/development.md](docs/development.md)** for the canonical repository layout and build instructions.

---

## Runtime Configuration (env vars)

See **[docs/configuration.md](docs/configuration.md)** for the canonical list of environment variables and configuration options.

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

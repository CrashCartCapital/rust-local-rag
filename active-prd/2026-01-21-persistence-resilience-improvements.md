# PRD: Persistence, Resilience & Configuration Hardening

**Project**: rust-local-rag  
**Created**: 2026-01-21  
**Status**: Draft (Revised to match current codebase)  
**Authors**: Claude Code (Sonnet 4.5), Codex CLI (GPT-5.2)  
**Reviewed With**: CRASH MCP reasoning chains + Gemini (`gemini-3-pro-preview`)  
**Based On**:
- `docs/architecture.md` (current architecture + persistence model)
- `analysis/20260114_operational_stability_hardening_PRD_v1.0.md` (approved stability PRD)
- Current implementation + tests (`src/worker.rs`, `src/job_manager.rs`, `src/embeddings.rs`, `tests/rag_persistence.rs`)

---

## Executive Summary

Several resilience goals that originally motivated this PRD are already achieved in the current codebase:
- Atomic JSON writes for `chunks_{model}.json` (temp file + rename)
- Corrupt index handling on load (`needs_reindex=true`, continue running)
- Restart resilience: the supervisor resumes pending/in-progress reindex jobs
- Incremental add/modify: per-document SHA-256 skip prevents re-embedding unchanged PDFs

This revised PRD focuses on the remaining correctness and maintainability gaps:
1. **Ghost results**: PDFs deleted from disk remain in the index indefinitely.
2. **Embedding dimension ambiguity**: embedding dimension is not reliably known/persisted/validated.
3. **Persistence API fragmentation**: `rag-core` has `PersistenceBackend`/`EngineState`, but the server bypasses it with a parallel persistence path.
4. **Job/index reconciliation UX**: after crashes, job state can lag reality even when the index is consistent.

**Impact**: Removes ghost results, prevents dimension-mismatch runtime failures, simplifies persistence, and culminates in a mandatory SQLite-backed index for durable, incremental writes.

---

## Problem Statement

### P0 Problems (Correctness / Silent Corruption)

**Problem 1: Orphaned / “Ghost” Documents After Deletion**

**Current behavior**: Reindex discovers PDFs and upserts them, but does not prune index entries for PDFs that no longer exist on disk.

**Failure mode**:
1. User deletes `some.pdf` from `DOCUMENTS_DIR`
2. User runs `start_reindex` (or restarts and resumes)
3. Index still contains chunks for `some.pdf`
4. Search results include a document that no longer exists (“ghost result”)

**User impact**: Silent correctness failure (search returns results from deleted files). This breaks user trust in a local-first RAG tool.

---

**Problem 2: Embedding Dimension Is Not Reliably Known or Validated**

**Current behavior**:
- Connectivity and model existence are validated at startup.
- The embedding dimension is not established reliably for Ollama today (dimension reporting is effectively unknown).
- Persisted index format does not explicitly include `embedding_dim`, preventing robust fail-fast validation.

**Failure mode**:
- If the embedding dimension changes for a model identifier (or backend behavior changes), search can hit dimension mismatch errors at runtime and `get_stats`/health reporting can be misleading.

**User impact**: Confusing runtime failures and unclear remediation (“why did search break?”).

---

### P1 Problems (Maintainability / Operational UX)

**Problem 3: Persistence API Fragmentation**

`rag-core` provides feature-gated persistence (`PersistenceBackend`, `EngineState`, `JsonFileBackend`) including migration helpers and metadata (schema version, model id, inferred embedding dim). The server currently persists via a parallel JSON format, creating:
- duplicate implementations
- duplicated migration logic
- inconsistent metadata surface
- higher future cost for adding new backends (binary/SQLite)

**User impact**: Slower iteration and higher risk of future persistence regressions.

---

**Problem 4: Job/Index Reconciliation Edge Cases**

Jobs live in SQLite (`jobs.db`), while index lives in JSON (`chunks_{model}.json`). While the system already resumes jobs and persists per-document, edge cases remain:
- crash after index state is saved but before job is marked completed
- multiple stale pending/in-progress jobs left from dev/test cycles
- job progress UX resets on resume (not a correctness issue, but confusing)

**User impact**: “Zombie job” status confusion and harder debugging.

---

## Goals / Non-Goals

### Goals
- The index reflects disk truth: **deleted PDFs do not appear in search results** after sync/reindex.
- Embedding configuration is validated with reliable dimension info; mismatches become clear, actionable states.
- The persistence layer is simplified: one canonical path using `rag-core` persistence abstractions.
- Restart behavior is predictable and observable (job status and index state converge quickly).

### Non-Goals (Out of Scope)
- Real-time file watching (FSEvents/inotify): sync remains explicit (reindex/sync call) or startup-driven.
- Sub-document incremental patching: unit of update remains “whole document if hash changes”.
- Distributed persistence (S3, remote DBs, vector DBs) — local-first remains the constraint.
- Mixed-dimension indexes — still one embedding model per index file.

---

## Requirements

### R1: Canonical Persistence API + Index Metadata (P0)

**Functional Requirements**:
- FR1.1: The server MUST use `rag-core`’s `PersistenceBackend` abstraction for all index save/load operations (default: `JsonFileBackend`).
- FR1.2: The persisted state MUST include: `schema_version`, `embedding_model_id`, `embedding_dim`, `needs_reindex`, `document_hashes`, and `chunks`.
- FR1.3: Save MUST be atomic (temp file + rename). If a save fails, the prior index MUST remain valid.
- FR1.4: Backward compatibility: existing `chunks_{model}.json` files MUST load, with `embedding_dim` inferred when missing.

**Non-Functional Requirements**:
- NFR1.1: Load+parse for 100K chunks SHOULD complete within 5 seconds on a typical dev laptop.
- NFR1.2: Search read performance MUST NOT regress noticeably (≤5%).
- NFR1.3: No new external runtime dependencies by default; alternative backends must be feature-gated.

**Success Criteria**:
- Upgrading from an existing `chunks_{model}.json` does not break startup; the index loads or transitions to a safe “needs reindex” state.
- There is one canonical persistence path in the server (no parallel save/load formats).

---

### R2: Embedding Dimension Discovery + Validation (P0)

**Functional Requirements**:
- FR2.1: The embedding backend MUST provide a non-zero, correct embedding dimension at runtime.
- FR2.2: On startup, the server MUST validate that persisted `embedding_dim` matches the current backend dimension when loading an existing index.
- FR2.3: If dimensions mismatch, the system MUST enter a safe state that prevents using incompatible embeddings:
  - mark `needs_reindex=true`
  - avoid serving vector search from incompatible loaded chunks (either clear them or treat as invalid until reindex)
  - emit an actionable message (“model changed; reindex required”)

**Non-Functional Requirements**:
- NFR2.1: Validation SHOULD complete within 5 seconds.
- NFR2.2: Errors MUST include remediation (reindex or revert model).

**Success Criteria**:
- Switching to a backend/model with a different dimension never causes a runtime panic; the system clearly indicates reindex is required.

---

### R3: Deletion Synchronization (P0)

**Functional Requirements**:
- FR3.1: Any sync/reindex operation MUST detect PDFs deleted from `DOCUMENTS_DIR` and remove their chunks from the index.
- FR3.2: Removal MUST update vector + lexical indexes and `document_hashes` consistently.
- FR3.3: The user must get a summary of deletions (documents removed + chunks removed).

**Non-Functional Requirements**:
- NFR3.1: Deletion pruning SHOULD be O(number of indexed documents) and complete within seconds for typical corpora.
- NFR3.2: Deletion pruning MUST NOT require a full rebuild of embeddings for unchanged documents.

**Success Criteria**:
- Delete a PDF from disk → run `start_reindex` → searches never return that document again.

---

### R4: Job/Index Reconciliation + Resume UX (P1)

**Functional Requirements**:
- FR4.1: On startup, the supervisor MUST reconcile resumable jobs with index state (e.g., if no work remains, mark job completed with a clear message).
- FR4.2: Resume MUST be idempotent and MUST NOT duplicate chunks.
- FR4.3: Multiple stale jobs must not block server startup indefinitely (guardrail: limit to one active reindex job, mark older ones failed).

**Success Criteria**:
- Crash mid-reindex → restart → job resumes and completes without duplicate chunks, and job status reflects completion.

---

### R5: Mandatory SQLite Index Persistence (Phase 5 / P1)

SQLite index persistence is a planned milestone, not an optional optimization. After the immediate correctness fixes (deletion pruning, embedding-dimension validation, and job reconciliation), we will migrate index persistence from JSON snapshots to a SQLite-backed store.

This is motivated by the current write pattern: reindexing saves the index after each processed document. As the corpus grows, repeatedly rewriting `chunks_{model}.json` becomes the dominant cost and increases the “crash window” during long saves.

**Functional Requirements**:
- FR5.1: The system MUST store index state in SQLite tables (chunks + document hashes + engine metadata) as the primary persistence format.
- FR5.2: The index MUST support multiple embedding models concurrently (preserving model switching) by partitioning all index rows by `model_id` (or equivalent).
- FR5.3: Per-document updates MUST be atomic via a single SQL transaction: delete old chunks for the document, insert new chunks, upsert the document hash, and update model metadata.
- FR5.4: The job system MUST record `model_id` in the job payload so resumes are model-safe (a job created under model A must not resume under model B without an explicit decision).
- FR5.5: On first startup after the migration ships, the system MUST auto-migrate any existing JSON index (`chunks_{model}.json`) into SQLite, keep a backup, and mark migration completion in SQLite so the migration is idempotent.
- FR5.6: SQLite becomes the default and required source of truth; JSON remains supported only as an import/export + emergency fallback for a defined transition window.

**Non-Functional Requirements**:
- NFR5.1: SQLite writes MUST NOT occur while holding the engine write lock (preserve the existing “short write lock” invariant). DB I/O happens outside the lock; in-memory updates happen after commit.
- NFR5.2: SQLite configuration MUST be enforced per connection: WAL mode, `busy_timeout`, `foreign_keys=ON`, and an explicit `synchronous` level.
- NFR5.3: Reindexing MUST remain resilient to crashes mid-document: no partially-written document state is visible after restart.

**Success Criteria**:
- Reindexing does not rewrite a monolithic JSON file after each document; it performs per-document SQL transactions.
- Kill/restart mid-reindex does not corrupt the DB; on restart the system resumes safely without duplicate chunks.
- Model switching cannot silently reuse an incompatible job; the resume path is model-aware and emits an actionable message.

---

## Design

### D1: Canonical JSON Persistence via `rag-core` `EngineState` (Recommended)

**Decision**: Adopt `rag-core`’s persistence abstractions as the single source of truth for index persistence.

Why this is pragmatic:
- The repo already ships `EngineState` + `JsonFileBackend` (feature enabled in the server).
- Legacy migration already exists (legacy state can be converted; `embedding_dim` can be inferred from stored chunks).
- Avoids introducing a new DB schema and `sqlx` usage into `rag-core` prematurely.

**Design notes**:
- Use `JsonFileBackend` for `{data_dir}/chunks_{sanitized_model}.json`.
- Persist `embedding_dim` explicitly (infer it when missing on legacy loads).
- Keep atomic temp-file write + rename pattern.

---

### D2: Embedding Dimension Discovery

**Decision**: Determine embedding dimension at runtime on startup and cache it for the process lifetime.

Recommended approach:
- Embed a fixed canary text once at startup (e.g., `"The quick brown fox jumps over the lazy dog"`).
- Dimension = returned vector length.
- Store this value in the embedding service so `EmbeddingBackend::dimension()` returns a real number.

Optional (P1): persist canary embedding to detect drift in weights (cosine similarity warning threshold).

---

### D3: Deletion Sync (“Prune” Step)

**Decision**: Treat deletion sync as part of the existing reindex flow to avoid extra MCP surface area.

Algorithm sketch:
1. Discover current PDFs in `DOCUMENTS_DIR`.
2. Build set of current document basenames.
3. Compare to the set of indexed documents (`document_hashes` keys or `engine.list_documents()`).
4. For any indexed doc missing from disk: call `rag-core`’s `remove_document`.
5. Save index state after pruning (and again after processing changed docs).

This preserves incremental behavior while eliminating ghost results.

---

### D4: Job/Index Reconciliation

**Decision**: Use restart-time reconciliation rather than transactionally unifying jobs + index (simpler, adequate for local-first).

Implementation sketch:
- On startup, for each resumable reindex job:
  - validate payload/documents dir
  - run a quick filesystem scan
  - if nothing needs processing (all docs unchanged + deletions pruned), mark job completed with message “No changes detected; index already up to date.”
  - otherwise, resume as today

---

### D5: SQLite Index Store (Mandatory, Phase 5)

This PRD commits to a full SQLite-backed index store after the immediate correctness fixes ship.

#### Decision Matrix (Architecture)

| Option | What it is | Pros | Cons |
|---|---|---|---|
| A | Keep JSON snapshots | Simple; already works | O(N) rewrite per save; scale pain; larger crash window |
| B | Store whole `EngineState` as a blob in SQLite | Transactional; easy migration | Still O(N) write; mostly “JSON-in-a-DB” |
| C (Chosen) | Granular SQLite tables (documents + chunks) | ACID per-doc updates; fast incremental writes; constraints/indexes | More schema + migration work |

#### Chosen Approach

- Implement a **server-level** `SqliteIndexStore` using the existing `sqlx` SQLite stack.
- Use a **single SQLite DB file** in `DATA_DIR` for jobs + index tables (shared pool, WAL).
- Partition all index data by `model_id` to preserve the current “one index per model” behavior without needing separate files.
- Keep `rag-core` as the in-memory search engine; SQLite is the durable store of record. The engine is hydrated from SQLite on startup and updated in-memory after successful commits.

#### Concrete Schema (Draft)

```sql
-- Model metadata (one row per embedding model)
CREATE TABLE IF NOT EXISTS rag_models (
    model_id TEXT PRIMARY KEY NOT NULL,
    embedding_dim INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    needs_reindex INTEGER NOT NULL DEFAULT 0,
    canary_embedding BLOB,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

-- Document-level hashes (one row per document per model)
CREATE TABLE IF NOT EXISTS rag_documents (
    model_id TEXT NOT NULL,
    document_name TEXT NOT NULL,
    document_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (model_id, document_name),
    FOREIGN KEY (model_id) REFERENCES rag_models(model_id) ON DELETE CASCADE
);

-- Chunk store (many rows per document per model)
CREATE TABLE IF NOT EXISTS rag_chunks (
    model_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    document_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,          -- f32 little-endian bytes
    page_number INTEGER NOT NULL,
    section TEXT,
    metadata_json TEXT NOT NULL,      -- serde_json of ChunkMetadata
    tags_json TEXT NOT NULL,          -- serde_json of HashSet<String>
    resolution TEXT NOT NULL,         -- enum as string
    parent_id TEXT,
    PRIMARY KEY (model_id, chunk_id),
    FOREIGN KEY (model_id, document_name)
        REFERENCES rag_documents(model_id, document_name)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_model_doc
    ON rag_chunks(model_id, document_name);
```

Notes:
- We intentionally align identity keys with current behavior (`document_name` is a basename today) to minimize behavioral changes.
- Embeddings are stored as raw bytes (no extra serialization dependency required); conversion is `Vec<f32> <-> Vec<u8>` via `to_le_bytes`/`from_le_bytes`.

#### Transaction Boundaries (Per-Document, Atomic)

For each processed document:
1. Begin transaction
2. `DELETE FROM rag_documents WHERE (model_id, document_name)=...` (cascades chunks)
3. `INSERT INTO rag_documents (...)`
4. `INSERT INTO rag_chunks (...)` for each chunk
5. Update `jobs.progress` and/or `jobs.status` if desired (same DB, same tx)
6. Commit
7. Apply the same prepared document to the in-memory `rag-core` engine (outside the transaction)

This preserves atomicity (no partial document state), while keeping engine locks short by performing DB I/O outside the engine write lock.

#### Migration Strategy (JSON → SQLite, Idempotent)

- On first run with the SQLite backend:
  - Load existing JSON index (`chunks_{model}.json`) into memory
  - Import into SQLite inside a transaction (insert model metadata + documents + chunks)
  - Rename JSON to `.migrated.bak` (do not delete)
  - Persist a “migration complete” marker in SQLite (e.g., in `rag_models.schema_version` or a dedicated `rag_meta` table)
- If migration is interrupted, restart must be able to retry safely (either full rollback due to transaction, or detect partial import via the marker and re-run).

---

## Implementation Tasks

### Phase 1: Persistence Unification + Metadata (P0)

**Task 1.1: Make `rag-core` persistence canonical**
- [ ] Adopt `EngineState` + `JsonFileBackend` as the server’s save/load path.
- [ ] Remove/avoid the parallel persistence path in the server.
- [ ] Ensure legacy JSON indexes still load via migration (no user action).

**Task 1.2: Persist and expose `embedding_dim`**
- [ ] Ensure loaded state has a valid `embedding_dim` (infer from chunks if needed).
- [ ] Persist `embedding_dim` explicitly on saves.
- [ ] Expose `embedding_dim` in `get_stats`/health outputs.

**Acceptance Criteria**:
- [ ] Existing users upgrade seamlessly; either index loads or `needs_reindex=true` with clear messaging.
- [ ] The server has one persistence implementation path (no duplication).

---

### Phase 2: Deletion Synchronization (P0)

**Task 2.1: Implement deletion pruning in reindex flow**
- [ ] Before embedding/upserting, prune indexed docs missing from disk.
- [ ] Record count of removed documents and removed chunks.
- [ ] Save state after pruning.

**Task 2.2: Add tests**
- [ ] Integration test: create an index with doc A → delete A on disk → run reindex → doc A not returned in results and not listed.

**Acceptance Criteria**:
- [ ] Ghost documents never appear after reindex/sync.

---

### Phase 3: Dimension Validation + Drift Detection (P1)

**Task 3.1: Implement real embedding dimension**
- [ ] Embed canary text on startup; set `EmbeddingBackend::dimension()` accordingly (non-zero).

**Task 3.2: Validate on load**
- [ ] Compare persisted `embedding_dim` to current backend `dimension()` when an index is present.
- [ ] On mismatch: enter safe state (`needs_reindex=true`, do not use incompatible embeddings for search).
- [ ] Provide clear remediation message.

**Task 3.3 (Optional): Canary drift warning**
- [ ] Persist canary embedding vector in state.
- [ ] On startup, compare cosine similarity; warn if below threshold.

**Acceptance Criteria**:
- [ ] No dimension mismatch runtime panics; mismatch is detected and handled.

---

### Phase 4: Job/Index Reconciliation Improvements (P1)

**Task 4.1: Reconcile resumable jobs**
- [ ] On startup, verify payload; if invalid, mark job failed.
- [ ] If job is resumable but no work remains, mark completed with message.
- [ ] Ensure multiple stale jobs do not block server startup (limit/guardrail).

**Acceptance Criteria**:
- [ ] “Zombie jobs” self-heal on restart; statuses converge quickly to truth.

---

### Phase 5: SQLite Index Persistence (Mandatory)

This phase is mandatory, but intentionally scheduled after Phases 1–4 so we first ship correctness fixes with minimal moving parts.

**Task 5.1: Database file + schema + migrations**
- [ ] Keep using the existing `jobs.db` file as the single SQLite database for both jobs + index tables (avoid a risky filename migration).
- [ ] Add schema migrations for `rag_models`, `rag_documents`, `rag_chunks` (and any meta tables).
- [ ] Ensure migrations run at startup (single place, deterministic), and PRAGMAs are enforced per connection.

**Task 5.2: Implement `SqliteIndexStore` (server crate)**
- [ ] `init()` / `ensure_schema()` (migrations + PRAGMAs)
- [ ] `load_model(model_id)` → stream chunks into in-memory engine on startup
- [ ] `upsert_document_atomic(model_id, document_name, document_hash, chunks)` (single transaction)
- [ ] `delete_document(model_id, document_name)` / `prune_missing_documents(model_id, current_docs)`
- [ ] Store/update `embedding_dim`, `needs_reindex`, optional `canary_embedding` in `rag_models`

**Task 5.3: Integrate with reindex + job system**
- [ ] Update reindex job payload to include `{ documents_dir, model_id, embedding_dim }` (JSON).
- [ ] On resume: if payload model differs from current model, mark job failed with remediation (“re-run reindex under current model”).
- [ ] Replace per-document JSON saves with per-document SQL transactions; update job progress in the same DB.
- [ ] Ensure deletion pruning happens in SQLite as well as in-memory (no ghost docs).

**Task 5.4: JSON → SQLite migration (one-time, idempotent)**
- [ ] If legacy `chunks_{model}.json` exists, import into SQLite within a transaction.
- [ ] Write a migration-complete marker (and schema version) in SQLite.
- [ ] Rename legacy JSON to `.migrated.bak` (never delete automatically).
- [ ] Migration can be safely re-run (no duplicates; idempotent marker prevents double-import).

**Task 5.5: De-risking tests (must-have)**
- [ ] **Per-document atomicity**: fault-inject mid-transaction → no partial doc state.
- [ ] **Kill/restart**: crash mid-reindex → restart → DB integrity OK; job resumes/settles; no duplicates.
- [ ] **Multi-model isolation**: index same doc under two models → no cross-contamination.
- [ ] **DB busy/locks**: concurrent `get_job_status` polling during reindex → no `SQLITE_BUSY` surfaced (retries/backoff).
- [ ] **Hydration correctness**: index → restart → search returns same docs; delete → restart → doc stays deleted.
- [ ] **Migration**: JSON fixture → import → counts match; backup exists; re-run import is a no-op.

**Acceptance Criteria**:
- [ ] SQLite is the primary index store for all runs (post-migration).
- [ ] Index updates are per-document transactions; no monolithic JSON rewrites during reindex.
- [ ] The system remains resilient to crashes and restarts; no corruption and no duplicate chunk growth.

---

## Testing Strategy

### Unit Tests
- Embedding dimension discovery (canary embed length)
- Dimension mismatch handling transitions to safe state
- Deletion diff logic (added/modified/deleted classification)
- EngineState migration from legacy persisted formats
- Embedding BLOB encoding/decoding round-trip (f32 <-> little-endian bytes)
- SQLite schema/migration marker idempotency

### Integration Tests
- **Ghost deletion**: index doc → delete file → reindex → doc removed everywhere
- **Restart resume**: crash mid-reindex → restart → completes without duplication
- **Dimension mismatch**: load index with dim A → backend dim B → safe state + clear message
- **JSON → SQLite migration**: import legacy JSON → counts + hashes match; backup exists; rerun is no-op
- **Per-document atomicity**: fail mid-transaction → no partial document state visible after restart
- **Multi-model isolation**: two models indexed concurrently → no cross-contamination
- **DB lock resilience**: poll jobs/status during indexing → no surfaced SQLITE_BUSY (retries/backoff)

### Acceptance Tests
- Run eval framework baseline after changes: no meaningful quality regressions.
- Verify typical “add a PDF” workflow remains incremental (only changed docs re-embedded).

---

## Rollout Strategy

### Phase 1-2 (P0)
- Roll out as default behavior (no feature flags), since changes are backward-compatible and correctness-focused.
- Keep legacy index backups where appropriate during migration.

### Phase 5 (SQLite, Mandatory)
- Roll out SQLite as the default index store with automatic JSON → SQLite migration and automatic `.migrated.bak` backups.
- Provide a short-lived emergency escape hatch (one release window): allow forcing legacy JSON read-only mode if SQLite migration fails unexpectedly.
- After the transition window: remove the escape hatch; keep JSON support only for import/export and backups.

---

## Success Metrics

### Primary (Correctness)
- **Zero ghost results** after deleting PDFs and running reindex/sync.
- **Zero dimension mismatch panics**; mismatches produce clear “reindex required” state.

### Secondary (UX / Ops)
- Reindex resumes safely after crash/restart without duplicating chunks.
- Job statuses converge to accurate state after restart (no long-lived “in progress” confusion).

---

## Risk Mitigation

### Risk 1: Startup/sync overhead due to deletion diff
**Mitigation**: Compare basenames and hashes without extracting PDF text unless needed; keep deletion prune O(docs).

### Risk 2: Legacy migration edge cases
**Mitigation**: Keep legacy backups; add explicit migration tests; on parse failures, fall back to “needs reindex” rather than crash.

### Risk 3: SQLite migration or locking regressions
**Mitigation**:
- Keep transactions strictly per-document (short write locks) and enforce WAL/PRAGMAs per connection.
- Add retries/backoff (or `busy_timeout`) for `SQLITE_BUSY` to avoid surfacing transient lock errors to users.
- After heavy reindexing, consider an explicit WAL checkpoint (`wal_checkpoint(TRUNCATE)`) to prevent unbounded `jobs.db-wal` growth.
- Size the SQLite pool to avoid starving job/status reads during indexing.
- Ship migration idempotency + rollback backups (`.migrated.bak`).

---

## Open Questions

1. **Failure mode preference for dimension mismatch**: fail server startup vs start in “needs reindex” mode?
   - Recommendation: start but block vector search until reindex (keeps MCP tools usable).

2. **Deletion prune trigger**: run only on `start_reindex`, or also on startup?
   - Recommendation: run on reindex; optionally run on startup if index is loaded and DOCUMENTS_DIR is accessible.

3. **SQLite DB filename**: keep `jobs.db` (now multi-purpose) vs rename to `rag.db`?
   - Recommendation: keep `jobs.db` as the single DB file for now (lower risk; no WAL/rename edge cases). Consider a rename in a follow-on PRD only if it becomes operationally valuable.

4. **Document identity**: should the durable key remain basename-only (`document_name`) or become a stable relative path under `DOCUMENTS_DIR` to avoid collisions?
   - Recommendation: keep basename for Phase 5 to avoid user-facing behavior changes; revisit as a follow-on hardening task.

---

## Appendix

### A1: Concrete Repo Anchors

- Index persistence and corruption handling: `crates/rag-core/src/engine.rs`, `tests/rag_persistence.rs`
- Persistence abstractions: `crates/rag-core/src/persistence.rs`
- Job persistence + single-active-reindex semantics: `src/job_manager.rs`
- Resume behavior and per-document saves: `src/worker.rs`
- Embedding startup validation (connectivity/model existence): `src/embeddings.rs`

---

## Conclusion

This revised PRD focuses first on closing the remaining correctness gaps (deletion sync, dimension validation, job reconciliation), then delivers a **mandatory** Phase 5 migration to a granular SQLite index store for durable, incremental persistence. It keeps the system local-first and solo-dev friendly while eliminating the highest-impact sources of silent index drift and scaling pain.

**Document Status**: Draft → Ready for implementation after self-review  
**Review Date**: 2026-01-21  
**Approver**: Self (Solo Developer)  

# Product Requirements Document: rust-local-rag Wave 2 (Modularity + Reliability)

**Version**: 2.0  
**Date**: 2026-01-09  
**Authors**: Codex CLI (AI-assisted), Gemini-3-pro-preview (consultant)  
**Status**: Draft for Implementation

---

## Executive Summary

Wave 1 delivered centralized tuning config, removed known crash hazards, and added targeted tests. Wave 2 focuses on **maintainability without behavior change**:

- **Modularity**: split the monolithic `src/mcp_server.rs` into cohesive modules while preserving all MCP/HTTP contracts.
- **Reliability**: reduce complexity in the worker reindex loop and add explicit **error-path** tests (not just happy-path).
- **Pragmatism**: avoid deep architectural rewrites (e.g., `rag-core` trait object refactors); keep this Wave implementable in **1–2 PRs**.

**Definition of Done (Wave 2)**:
- Code is modularized/decomposed as specified, with no behavior regressions.
- `cargo test --workspace` passes.
- Builds remain warning-free under repo policy (`-D warnings`).

---

## Guiding Principles

- **Modularity**: separate concerns (MCP tools vs HTTP handlers vs models/serialization).
- **Extensibility**: make it easy to add a new tool/endpoint without touching unrelated code.
- **Relevance**: prioritize changes that reduce bug surface area and developer friction.
- **Pragmatism**: preserve existing behavior and public interfaces; avoid speculative abstraction.
- **Operational safety**: do not block Tokio; keep locks short; preserve job semantics and poison-pill behavior.

---

## Scope

### In Scope (Wave 2)

1. **MCP/HTTP modularization**
2. **Worker reindex decomposition**
3. **Error-path integration tests**
4. **Targeted removal of risky `unwrap()` in production paths**

### Out of Scope (Explicit)

- Any changes to retrieval/scoring semantics in `rag-core` (including making `Rerank` object-safe).
- Persistence format changes (index layout, hashes) or migrations.
- New user-facing features (new tools, new endpoints, new CLI flags).
- A new shared `OllamaClient` abstraction (defer to a later wave).
- Coverage tooling targets (tarpaulin, blanket % goals).

---

# PART 1: REQUIREMENTS

## R1: Modularize MCP/HTTP Server (MUST)

### R1.1: Module Split (MUST)

**Problem**: `src/mcp_server.rs` is ~1000+ LOC and mixes:
- MCP tool handlers
- HTTP handlers + routing
- request/response models
- formatting helpers
- tests

**Requirement**: Introduce a `src/mcp/` module tree and move code by responsibility:

```
src/mcp/
├── mod.rs          # public entrypoints + re-exports
├── tools.rs        # MCP #[tool] handlers
├── http.rs         # Axum HTTP handlers + router
├── models.rs       # request/response structs (serde/schemars)
├── responses.rs    # shared response structs + conversions (MCP + HTTP)
└── formatting.rs   # shared human-readable formatting helpers (MCP search output)
```

Keep `src/mcp_server.rs` as a **thin compatibility shim** (re-export entrypoints/types from `src/mcp/`).

**Acceptance Criteria**:
- [ ] `rust_local_rag::mcp_server::start_mcp_server(...)` keeps the same signature and behavior.
- [ ] Existing public type paths used by tests/TUI remain valid via re-exports (e.g., `rust_local_rag::mcp_server::SearchRequest`).
- [ ] MCP tool names and schemas remain compatible: `search_documents`, `list_documents`, `get_stats`, `start_reindex`, `get_job_status`, `calibrate_reranker`.
- [ ] HTTP routes remain unchanged: `/healthz`, `/health`, `/readyz`, `/search`, `/stats`, `/reindex`, `/jobs/active`, `/jobs/{job_id}`, plus MCP endpoint path `MCP_HTTP_ENDPOINT`.
- [ ] Shared response structs live in one place and are used by both HTTP and MCP handlers where applicable (job status + reindex start).
- [ ] No cyclic module dependencies; `models.rs` remains “data only”.
- [ ] Builds warning-free with repo’s `-D warnings` policy.

### R1.2: Size + Ownership Constraints (SHOULD)

**Acceptance Criteria**:
- [ ] Each file in `src/mcp/` is ≤ ~350 LOC (excluding tests).
- [ ] Adding a new MCP tool requires changes only in `src/mcp/tools.rs` and (if needed) `src/mcp/models.rs`.

---

## R2: Decompose Worker Reindex Logic (MUST)

**Problem**: `WorkerSupervisor::reindex_documents` is long and mixes discovery, processing, progress, and finalization, increasing regression risk.

**Requirement**: Extract helpers with narrow responsibilities, keeping behavior stable:

Suggested extraction (names may vary):
- `discover_pdfs(documents_dir: &Path) -> Result<Vec<PathBuf>>`
- `process_single_pdf(...) -> Result<PerDocOutcome>`
- `update_job_progress(...) -> Result<()>`
- `finalize_reindex(...) -> Result<()>`

**Non-negotiable invariants** (do not regress):
- Do not block Tokio runtime (use `spawn_blocking` for CPU-bound work).
- Keep `Arc<RwLock<RagEngine>>` write locks short; do not hold for entire reindex.
- Preserve “single active reindex job” semantics.
- Preserve “poison-pill” behavior: a single document failure does not abort the whole job.

**Acceptance Criteria**:
- [ ] Functional behavior is unchanged (existing integration tests remain green).
- [ ] `reindex_documents` becomes primarily orchestration (target ≤ ~200 LOC).
- [ ] Any heavy work stays outside write locks; no new long-held locks introduced.

---

## R3: Error-Path Test Coverage (MUST)

**Problem**: Current tests primarily validate happy paths; key failure behaviors can regress silently.

**Requirement**: Add a small, stable set of integration tests focused on failure modes that matter operationally.

### R3.1: Corrupt PDF Does Not Abort Reindex (MUST)

**Acceptance Criteria**:
- [ ] A reindex job with a mix of valid PDFs and one corrupt/invalid file completes without panicking.
- [ ] The job records progress and completes (or completes with a recorded failure), and valid documents are still indexed.

### R3.2: Reranker Timeout Falls Back (MUST)

**Acceptance Criteria**:
- [ ] When the reranker request exceeds configured timeout, search still returns results (fallback to initial/embedding scoring).
- [ ] The test does not depend on a real Ollama server (use mock server + artificial delay).

### R3.3: Job Dedup Race Remains Correct (SHOULD)

**Acceptance Criteria**:
- [ ] Concurrency test for “single active reindex job” continues to pass.

---

## R4: Production Panic Hygiene (SHOULD)

**Problem**: `unwrap()`/panic paths in production code can crash the server for recoverable conditions.

**Requirement**: Perform a targeted pass over production code paths touched by Wave 2 refactors:
- Replace risky `unwrap()` with `?`, `ok_or_else`, or explicit error responses.
- Allow `unwrap()` only where invariants are proven and truly non-recoverable; prefer `expect("...")` with a descriptive message in those cases.

**Acceptance Criteria**:
- [ ] No new production `unwrap()` introduced as part of Wave 2.
- [ ] The refactor does not alter external API behavior (errors remain user-readable and actionable).

---

# PART 2: DESIGN (LIGHTWEIGHT)

## D1: Dependency Rules for `src/mcp/`

- `models.rs` contains only data structures (serde/schemars).
- `responses.rs` contains shared response structs and `From` conversions.
- `tools.rs` depends on `models.rs` and `responses.rs`.
- `http.rs` depends on `responses.rs` and may share minimal “application state” types with tools, but avoids importing MCP macros directly.
- `mod.rs` re-exports stable entrypoints.

## D2: Compatibility Shim

Keep `src/mcp_server.rs` as a stable import path that re-exports:
- request structs used externally (e.g., `SearchRequest`)
- `start_mcp_server`

This allows internal modularization with minimal downstream churn.

---

# PART 3: TASKS (1–2 PRs)

## PR A: Modularization + Worker Decomposition

1. Create `src/mcp/` modules and move code by responsibility (tools/http/models/responses).
2. Keep `src/mcp_server.rs` as a re-export/shim to preserve public paths.
3. Decompose `WorkerSupervisor::reindex_documents` into helper functions (no behavioral changes).
4. Run full unit + integration tests; fix regressions.

## PR B: Error-Path Tests + Targeted Panic Hygiene

5. Add integration tests for corrupt PDF and reranker timeout fallback.
6. Targeted removal of production `unwrap()` introduced/encountered in the refactor paths.
7. Run full test suite; verify warning-free build.

---

# PART 4: RISKS + MITIGATIONS

1. **Risk**: MCP/HTTP behavior changes during modularization.  
   **Mitigation**: Preserve router wiring and tool handler signatures; add regression tests; keep the shim and re-exports.

2. **Risk**: Accidental circular dependencies in new `src/mcp/` modules.  
   **Mitigation**: Enforce the dependency rules (D1); keep models data-only.

3. **Risk**: Worker refactor reintroduces long-held write locks.  
   **Mitigation**: Explicitly scope locks; keep heavy work outside locks; rely on existing lock instrumentation/tests.

4. **Risk**: Error-path tests become flaky (timing).  
   **Mitigation**: Use deterministic mocks and generous time margins; avoid relying on system load.

5. **Risk**: Rollback complexity if a refactor goes sideways.  
   **Mitigation**: Keep PR A mechanical (move-only) where possible; keep PR B behavior changes limited to tests/error handling; use small commits.

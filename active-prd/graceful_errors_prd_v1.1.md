# Graceful Error Handling PRD

**Project**: rust-local-rag
**Version**: v1.1
**Date**: 2026-01-17
**Status**: ACTIVE

---

## Executive Summary

This PRD replaces the rejected "Operational Stability Hardening" v1.0, which was found to solve imagined problems rather than real user needs. v1.1 is a focused effort delivering **tangible stability improvements** aligned with actual usage patterns.

**Core Goal**: Replace panics with helpful error messages in user-facing code paths, so users see actionable guidance instead of crashes.

### What Changed from v1.0

| v1.0 (Rejected) | v1.1 (This PRD) |
|-----------------|-----------------|
| 3 phases, enterprise scope | Single focused effort |
| Security hardening, chaos testing | Graceful errors only |
| Job watchdog, drift monitoring | Per-document failure isolation |
| Memory optimization (Arc refactor) | Deferred until evidence of OOM |
| 10 functional requirements | 4 focused requirements |

### Ensemble Validation

**Reviewers**: CRASH MCP, Codex, Gemini 3 Pro Preview
**Consensus**: v1.0's speculative enterprise patterns inappropriate for local single-user tool. This minimal alternative delivers concrete value.

---

## Requirements

### 1.1 Functional Requirements

| ID | Requirement | Priority | Testable Outcome |
|----|-------------|----------|------------------|
| **FR-1** | Replace `unwrap`/`expect` in user-facing paths with structured errors | P0 | Config errors, PDF failures, search issues return `Result` with helpful messages |
| **FR-2** | Per-document failure isolation during indexing | P0 | One bad PDF doesn't abort entire reindex; job completes with failure summary |
| **FR-3** | Unified `RagError` type with actionable messages | P1 | Errors include: what failed, likely cause, suggested fix |
| **FR-4** | Basic mismatch detection (index/model) | P2 | Startup warns if index model differs from configured model |

### 1.2 Non-Functional Requirements

| ID | Requirement | Testable Outcome |
|----|-------------|------------------|
| **NFR-1** | Zero functional regressions | All existing tests pass |
| **NFR-2** | No new dependencies | Cargo.toml unchanged (use existing `thiserror`, `anyhow`) |
| **NFR-3** | No performance degradation | Search latency unchanged (error paths are exceptional) |

### 1.3 Explicit Non-Goals

These items from v1.0 are **intentionally excluded** as over-engineering for a local single-user tool:

- Security hardening (path traversal, symlink blocking) - user has filesystem permissions anyway
- Job watchdog / zombie monitoring - no evidence of zombie jobs in git history
- Model drift detection / MLOps monitoring - static local embeddings, not live models
- Memory optimization (Arc<str>, Arc<[f32]>) - no OOM reports; measure before optimizing
- Chaos engineering / failure injection suite - inappropriate for local desktop tool
- Batch size limits - defensive but no evidence of issues

---

## Design

### 2.1 Error Architecture

**Pattern**: Use `thiserror` for typed errors, `anyhow` for context propagation.

```rust
// src/error.rs (new file)
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RagError {
    #[error("Configuration error: {message}\n  Cause: {cause}\n  Fix: {fix}")]
    Config { message: String, cause: String, fix: String },

    #[error("PDF extraction failed for '{filename}': {reason}\n  Fix: {fix}")]
    PdfExtraction { filename: String, reason: String, fix: String },

    #[error("Search failed: {message}\n  Fix: {fix}")]
    Search { message: String, fix: String },

    #[error("Embedding service error: {message}")]
    Embedding { message: String },

    #[error("Index mismatch: stored model '{stored}' differs from configured '{configured}'\n  Fix: Run reindex or update OLLAMA_EMBEDDING_MODEL")]
    ModelMismatch { stored: String, configured: String },

    #[error("Document processing failed: {failures_count} of {total_count} documents failed\n  Details: {details}")]
    PartialIndexFailure { failures_count: usize, total_count: usize, details: String },
}
```

### 2.2 Per-Document Failure Isolation

**Current behavior**: Unknown (needs verification)
**Target behavior**: Indexing continues when individual documents fail; summary reports failures.

```rust
// src/worker.rs - reindex_documents modification
pub async fn reindex_documents(&self) -> Result<IndexingSummary> {
    let mut successes = Vec::new();
    let mut failures = Vec::new();

    for pdf_path in discover_pdfs(&self.documents_dir)? {
        match self.process_single_document(&pdf_path).await {
            Ok(chunk_count) => successes.push((pdf_path, chunk_count)),
            Err(e) => {
                tracing::warn!("Failed to process {}: {}", pdf_path.display(), e);
                failures.push((pdf_path, e.to_string()));
                // Continue processing other documents
            }
        }
    }

    Ok(IndexingSummary { successes, failures })
}
```

### 2.3 User-Facing Error Paths

**Audit scope** - these paths need `unwrap`/`expect` replacement:

| Path | File | Current Risk | Fix |
|------|------|--------------|-----|
| Config loading | `src/config.rs` | Panic on missing env | Return `RagError::Config` |
| PDF extraction | `src/rag_engine.rs` | Panic on extraction failure | Return `RagError::PdfExtraction` |
| Search execution | `src/rag_engine.rs` | Panic on embedding failure | Return `RagError::Search` |
| Index loading | `src/rag_engine.rs` | Panic on JSON parse error | Return `RagError::Config` |
| TUI channel recv | `src/bin/rag_tui/main.rs` | Panic on channel close | Graceful exit with message |

### 2.4 Mismatch Detection (Lightweight)

On `RagEngine::load_from_disk()`, compare stored model with configured:

```rust
if let Some(stored_model) = persisted_chunks.model.as_ref() {
    if stored_model != &self.embedding_model {
        tracing::warn!(
            "Index was built with model '{}' but current model is '{}'. Reindex recommended.",
            stored_model,
            self.embedding_model
        );
        // Don't auto-flag needs_reindex - just inform user
    }
}
```

---

## Tasks

### Section: Error Types
- [x] Task 1: Create RagError Type
  - **Description**: Create `src/error.rs` with `RagError` enum using `thiserror`.
  - **Acceptance Criteria**:
    - `RagError` variants for: Config, PdfExtraction, Search, Embedding, ModelMismatch, PartialIndexFailure
    - Each variant includes actionable fix suggestion
    - Unit tests for error formatting
  - **Files**:
    - Create: `src/error.rs`
    - Modify: `src/lib.rs` (add `pub mod error;`)

### Section: Configuration
- [x] Task 2: Replace Panics in Config Loading
  - **Description**: Audit `src/config.rs` and replace `unwrap`/`expect` with `Result<Config, RagError>`.
  - **Acceptance Criteria**:
    - Missing required env vars return `RagError::Config` with fix suggestion
    - Invalid values (non-numeric ports, etc.) return descriptive errors
    - Test: intentionally missing `DATA_DIR` returns helpful error
  - **Files**:
    - Modify: `src/config.rs`
    - Modify: `src/main.rs` (handle Config error at startup)

### Section: RagEngine (PDF + Index)
- [x] Task 3: Replace Panics in PDF/Index Operations
  - **Description**: Audit `src/rag_engine.rs` for `unwrap`/`expect` in:
    - PDF extraction (`add_document`)
    - Index loading (`load_from_disk`)
    - Index saving (`save_to_disk`)
  - **Acceptance Criteria**:
    - PDF extraction errors return `RagError::PdfExtraction` with filename
    - Index JSON parse errors return `RagError::Config` with path
    - Test: corrupt index file returns helpful error
  - **Files**:
    - Modify: `src/rag_engine.rs`

- [x] Task 5: Model Mismatch Warning
  - **Description**: Add lightweight mismatch detection to `load_from_disk()`.
  - **Acceptance Criteria**:
    - Startup logs warning if stored model != configured model
    - Warning includes remediation: "Run reindex or update OLLAMA_EMBEDDING_MODEL"
    - Test: create index with model A, load with model B, verify warning
  - **Files**:
    - Modify: `src/rag_engine.rs`

### Section: Indexing Jobs
- [x] Task 4: Per-Document Failure Isolation
  - **Description**: Modify `src/worker.rs` so indexing continues when individual documents fail.
  - **Acceptance Criteria**:
    - One bad PDF doesn't abort entire reindex
    - Job completes with `IndexingSummary` containing successes and failures
    - MCP `get_job_status` returns failure details
    - Test: place intentionally corrupt PDF in test directory, verify other PDFs indexed
  - **Files**:
    - Modify: `src/worker.rs`
    - Modify: `src/job_manager.rs` (store failure details in job record)

### Section: TUI
- [x] Task 6: TUI Graceful Exit
  - **Description**: Handle channel receiver errors in TUI without panic.
  - **Acceptance Criteria**:
    - TUI exits gracefully on server disconnect
    - User sees "Connection lost. Please restart." instead of panic
    - Test: manual verification
  - **Files**:
    - Modify: `src/bin/rag_tui/main.rs`

### Section: Tests
- [ ] Task 7: Add Regression Tests
  - **Description**: Create focused tests for common failure modes.
  - **Acceptance Criteria**:
    - Test: bad PDF file (0 bytes, non-PDF) handled gracefully
    - Test: missing Ollama (connection refused) handled gracefully
    - Test: corrupt index JSON handled gracefully
    - All tests in `tests/error_handling.rs`
  - **Files**:
    - Create: `tests/error_handling.rs`

---

## Testing Strategy

PRD Size: Medium (7 tasks, user-facing stability changes).

- Unit: `src/error.rs` formatting + config parsing error cases.
- Integration: `tests/error_handling.rs` covers corrupt PDF, corrupt index JSON, and missing Ollama error propagation.
- E2E: N/A (covered by integration tests + manual TUI disconnect check).

---

## Addenda

### Task Dependencies

```
Task 1 (RagError) ─┬─> Task 2 (Config)
                   ├─> Task 3 (PDF/Index)
                   ├─> Task 4 (Per-Doc Isolation)
                   └─> Task 7 (Tests)

Task 5 (Mismatch Warning) ─> Independent
Task 6 (TUI Exit) ─> Independent
```

---

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| User-facing panic paths | 0 | `grep -r "unwrap\|expect" src/ --include="*.rs"` in user-facing functions |
| Reindex with bad PDF | Continues | Test places corrupt PDF, verifies other docs indexed |
| Error messages | Actionable | Each error includes: what failed, cause, fix suggestion |
| Test coverage | No reduction | `cargo test` passes, no new untested paths |
| Performance | No change | Search latency benchmarks unchanged |

---

### Deferred to Future Work

These items may be revisited if evidence emerges:

| Item | Trigger for Reconsideration |
|------|----------------------------|
| Security hardening (path traversal) | Evidence of exploit or multi-tenant deployment |
| Job watchdog | Evidence of zombie jobs in production |
| Memory optimization | User reports OOM or profiling shows hotspot |
| Chaos engineering tests | Moving to production/CI environment |
| Batch size limits | Evidence of memory issues with large batches |

---

### Validation Gate

Before implementation, verify:

1. **Existing unwraps**: Run `grep -r "unwrap\|expect" src/ --include="*.rs"` to audit current state
2. **Current behavior**: Test what happens with corrupt PDF today (panic? graceful?)
3. **Test infrastructure**: Verify `tests/` directory structure supports new test file

---

**END OF PRD v1.1**

*Next Steps*:
1. Create branch: `git checkout -b feat/graceful-errors`
2. Run validation gate (grep audit, behavior test)
3. Begin Task 1: Create `src/error.rs`
4. Execute tasks sequentially, mark complete in this file
5. Run `cargo test` after each task to verify no regressions
